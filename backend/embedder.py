import numpy as np
from sentence_transformers import SentenceTransformer

from .config import CANDIDATE_POOL_SIZE, EMBEDDING_MODEL


class Embedder:
    def __init__(self) -> None:
        # Model is downloaded once at build time via precompute.py and cached.
        # Subsequent loads hit the local cache — no network call at startup.
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> np.ndarray:
        # normalize_embeddings=True means cosine similarity reduces to dot product —
        # same ranking, avoids the unnecessary sqrt at query time
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=64,
        )

    def top_k(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int = CANDIDATE_POOL_SIZE,
    ) -> list[int]:
        """
        Return indices of the k most similar titles, sorted descending by similarity.

        argpartition is O(n) vs argsort's O(n log n). At 8k titles the wall-clock
        difference is small, but it's the right tool: we only need the top-k
        ordered, not all n sorted.
        """
        k = min(k, len(corpus_embeddings))
        scores = corpus_embeddings @ query_embedding
        top_indices = np.argpartition(scores, -k)[-k:]
        return sorted(top_indices.tolist(), key=lambda i: float(scores[i]), reverse=True)
