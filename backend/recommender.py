import json
import logging
from typing import Optional

from groq import Groq

from .config import (
    CANDIDATE_POOL_SIZE,
    GROQ_API_KEY,
    GROQ_PARSE_MODEL,
    GROQ_RERANK_MODEL,
    RESULT_COUNT,
)
from .embedder import Embedder
from .filters import apply_hard_filters
from .models import ParsedIntent, Recommendation

logger = logging.getLogger(__name__)

# Short prompt — 8b is fast and the task is mechanical structured extraction.
# Giving it more tokens won't improve JSON quality; temperature 0.1 handles that.
_PARSE_SYSTEM = """\
Extract movie/TV search intent from a natural language query.

Return ONLY valid JSON — no markdown fences, no explanation:
{
  "mood": "dark|uplifting|tense|funny|romantic|unsettling|contemplative|etc or null",
  "themes": ["thematic keywords relevant to the query"],
  "genres": ["matching genres, e.g. Dramas, Thrillers, Horror Movies, Comedies, Documentaries, Action & Adventure, Sci-Fi & Fantasy, Crime TV Shows, Romantic Movies"],
  "content_type": "Movie" or "TV Show" or null,
  "era_preference": "classic" or "80s" or "90s" or "2000s" or "2010s" or "recent" or null,
  "enriched_query": "expanded query for semantic search — add tone, emotional texture, similar titles, thematic associations"
}"""

# Rerank prompt for the 70b model — more explicit rubric because the task
# requires reasoning about fit, not just extraction
_RERANK_SYSTEM = """\
You are a film critic and recommendation expert. Given a user query and a candidate list from semantic search, select and rank the best 5.

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {
    "title": "exact title from the candidates list",
    "match_reason": "one sentence explaining specifically why this matches the query",
    "confidence": <float 0.0-1.0>
  }
]

Confidence guide: 0.9+ exceptional match, 0.7–0.89 strong, 0.5–0.69 decent, below 0.5 weak."""


class Recommender:
    def __init__(
        self,
        embedder: Embedder,
        embeddings,  # np.ndarray loaded once at startup
        titles_index: list[dict],
    ) -> None:
        self.embedder = embedder
        self.embeddings = embeddings
        self.titles_index = titles_index
        # Only instantiate the Groq client if a key is present — lets fast mode
        # work in environments with no API key configured
        self._groq: Optional[Groq] = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    def recommend(self, query: str, pipeline_mode: str) -> dict:
        if pipeline_mode == "fast":
            return self._fast(query)
        elif pipeline_mode == "balanced":
            return self._balanced(query)
        else:
            return self._full(query)

    # -------------------------------------------------------------------------
    # Pipeline modes
    # -------------------------------------------------------------------------

    def _fast(self, query: str) -> dict:
        """Raw query → embedding → cosine top 5. Zero Groq calls."""
        query_vec = self.embedder.embed(query)
        indices = self.embedder.top_k(query_vec, self.embeddings, k=RESULT_COUNT)
        return {
            "interpreted_as": None,
            "recommendations": [
                self._title_to_rec(self.titles_index[i]) for i in indices
            ],
        }

    def _balanced(self, query: str) -> dict:
        """Stage 1 intent parse → enriched query → cosine top 5. One Groq call."""
        intent = self._parse_intent(query)
        search_query = intent.enriched_query if intent else query

        query_vec = self.embedder.embed(search_query)
        indices = self.embedder.top_k(query_vec, self.embeddings, k=RESULT_COUNT)
        return {
            "interpreted_as": intent,
            "recommendations": [
                self._title_to_rec(self.titles_index[i]) for i in indices
            ],
        }

    def _full(self, query: str) -> dict:
        """Stage 1 parse → hard filter → cosine top 20 → Stage 3 rerank. Two Groq calls."""
        intent = self._parse_intent(query)
        search_query = intent.enriched_query if intent else query

        # Apply content_type and era constraints before cosine search —
        # reduces corpus to valid candidates only, so rankings are over relevant titles
        filtered_titles, filtered_embeddings = apply_hard_filters(
            self.titles_index,
            self.embeddings,
            content_type=intent.content_type if intent else None,
            era_preference=intent.era_preference if intent else None,
        )

        query_vec = self.embedder.embed(search_query)
        indices = self.embedder.top_k(
            query_vec, filtered_embeddings, k=CANDIDATE_POOL_SIZE
        )
        candidates = [filtered_titles[i] for i in indices]

        # Reranker can fail gracefully — cosine order is already a reasonable fallback
        reranked = self._rerank(query, intent, candidates)
        recommendations = reranked or [
            self._title_to_rec(c) for c in candidates[:RESULT_COUNT]
        ]

        return {"interpreted_as": intent, "recommendations": recommendations}

    # -------------------------------------------------------------------------
    # LLM stages
    # -------------------------------------------------------------------------

    def _parse_intent(self, query: str) -> Optional[ParsedIntent]:
        """
        Stage 1: llama3-8b parses the raw query into structured intent.
        Returns None on any failure — callers fall back to the raw query string.
        """
        if not self._groq:
            return None

        try:
            response = self._groq.chat.completions.create(
                model=GROQ_PARSE_MODEL,
                messages=[
                    {"role": "system", "content": _PARSE_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences in case the model ignores the "no fences" instruction
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
            return ParsedIntent(
                mood=data.get("mood"),
                themes=data.get("themes") or [],
                genres=data.get("genres") or [],
                content_type=data.get("content_type"),
                era_preference=data.get("era_preference"),
                enriched_query=data.get("enriched_query") or query,
            )

        except Exception as exc:
            logger.warning("Intent parsing failed, using raw query: %s", exc)
            return None

    def _rerank(
        self,
        query: str,
        intent: Optional[ParsedIntent],
        candidates: list[dict],
    ) -> Optional[list[Recommendation]]:
        """
        Stage 3: llama3-70b reranks the cosine top-20 and adds match_reason +
        confidence. Returns None on failure so the caller can fall back cleanly.
        """
        if not self._groq or not candidates:
            return None

        candidate_lines = "\n".join(
            f"{i + 1}. {c['title']} ({c['type']}, {c.get('release_year', '?')}): "
            f"{c['description'][:150]}"
            for i, c in enumerate(candidates)
        )

        intent_summary = (
            f"mood={intent.mood}, themes={intent.themes}, genres={intent.genres}"
            if intent
            else "no structured intent available"
        )

        user_msg = (
            f'User query: "{query}"\n'
            f"Interpreted as: {intent_summary}\n\n"
            f"Candidates:\n{candidate_lines}\n\n"
            f"Select and rank the top 5 most relevant titles."
        )

        try:
            response = self._groq.chat.completions.create(
                model=GROQ_RERANK_MODEL,
                messages=[
                    {"role": "system", "content": _RERANK_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            reranked_data = json.loads(raw)

            # Title lookup tolerates minor punctuation differences from the model
            title_lookup = {c["title"].lower(): c for c in candidates}
            results: list[Recommendation] = []

            for item in reranked_data[:RESULT_COUNT]:
                key = item.get("title", "").lower()
                meta = title_lookup.get(key)

                if meta is None:
                    # Soft match — model sometimes drops leading "The" or adds punctuation
                    for k, v in title_lookup.items():
                        if key in k or k in key:
                            meta = v
                            break

                if meta:
                    results.append(
                        Recommendation(
                            title=meta["title"],
                            type=meta["type"],
                            genres=meta["genres"],
                            description=meta["description"],
                            release_year=meta.get("release_year", 0),
                            match_reason=item.get("match_reason"),
                            confidence=float(item.get("confidence", 0.7)),
                        )
                    )

            return results or None

        except Exception as exc:
            logger.warning("Reranking failed, falling back to cosine order: %s", exc)
            return None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _title_to_rec(title_data: dict) -> Recommendation:
        return Recommendation(
            title=title_data["title"],
            type=title_data["type"],
            genres=title_data["genres"],
            description=title_data["description"],
            release_year=title_data.get("release_year", 0),
        )
