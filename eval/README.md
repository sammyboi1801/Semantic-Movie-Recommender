# Evaluation Suite

Automated evaluation across all three pipeline modes using 20 curated test queries.

## Running

Start the server first (or use the Docker container):

```bash
# Option A: local dev
uvicorn backend.main:app --host 0.0.0.0 --port 8080

# Option B: Docker
docker build -t movie-recommender .
docker run -p 8080:80 movie-recommender
```

Then run the eval:

```bash
cd movie-recommender/
python eval/evaluate.py
```

Options:

```
--url     Server URL (default: http://localhost:8080)
--mode    Run one mode only: fast | balanced | full
--timeout Request timeout in seconds (default: 45)
```

## Query Set

20 queries across 4 types:

| Type | Count | Description |
|------|-------|-------------|
| `vague_mood` | 6 | Mood-based, no explicit genre — e.g. "something unsettling that stays with you" |
| `semi_structured` | 6 | Partial constraints — e.g. "dark comedy from the 2010s" |
| `explicit_reference` | 6 | Reference-based — e.g. "movies like Inception" |
| `edge_case` | 2 | Gibberish and whitespace-only input |

## Scoring

**Precision@5**: fraction of top-5 results that score ≥ 1 on the graded relevance scale.

- **2 (strong)**: genre exactly matches the query's expected tone (e.g. "Horror Movies" for a horror query)
- **1 (weak)**: genre is plausible but not ideal (e.g. "Dramas" for a horror query)
- **0 (miss)**: genre is irrelevant or contradicts the query

Genre matching is case-insensitive substring — "TV Thrillers" matches "Thriller".

Edge cases are excluded from aggregate averages (E1 tests graceful handling; E2 tests HTTP 422).

**Confidence calibration**: Spearman ρ between the reranker's confidence scores and graded relevance scores across all full-mode results. A positive ρ means the model is more confident when it's actually right.

## What the Table Shows

```
ID    Type              fast  balanced      full   vs fast  vs balanced
----------------------------------------------------------------------
A1    vague_mood       0.400     0.600     0.800    +0.400      +0.200
...
AVG   (excl. edge)     0.467     0.567     0.700    +0.233      +0.133
```

The delta columns quantify how much the additional Groq calls improve precision over the baseline cosine search.
