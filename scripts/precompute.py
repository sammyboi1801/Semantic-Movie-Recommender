"""
Precompute title embeddings from netflix_data.csv.

Run once before starting the server — the Dockerfile does this at build time
so the container boots with embeddings already on disk.

Output:
  data/embeddings.npy      — float32 matrix of shape (n_titles, 384)
  data/titles_index.json   — list of title metadata dicts, index-aligned with embeddings
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path so we can import from the backend package
# without installing the package
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import (
    CSV_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    TITLES_INDEX_PATH,
)
from backend.embedder import Embedder


def build_embedding_text(row: pd.Series) -> str:
    """
    Construct the string we embed for each title.

    Description carries the most semantic weight — it's dense natural language
    that encodes tone, plot, and mood. Genre tags anchor the embedding in the
    right neighborhood of the latent space. Director/cast are included when
    present because talent-specific queries ("movies directed by Scorsese") need
    them, but they're often null so we check before including.

    Limiting cast to the first 4 names avoids padding the embedding with 20
    names that dilute the signal from description.
    """
    parts = [
        row["title"],
        row["type"],
        f"Genres: {row['listed_in']}",
        row["description"],
    ]

    if row.get("director"):
        parts.append(f"Director: {row['director']}")

    if row.get("cast"):
        cast_preview = ", ".join(str(row["cast"]).split(",")[:4]).strip()
        parts.append(f"Cast: {cast_preview}")

    return ". ".join(p for p in parts if p)


def main() -> None:
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows loaded")

    # Surface null patterns before we clean them — useful to know which fields
    # can't be relied on (director is null ~30% of the time in this dataset)
    null_counts = df.isnull().sum()
    noisy = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(noisy):
        print(f"  Null counts: {noisy.to_dict()}")

    # description is essential for embedding quality — drop the handful of rows
    # that are missing it rather than embedding an empty string
    before = len(df)
    df = df.dropna(subset=["description"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing description")

    # Fill remaining nulls so string operations below don't explode
    df["listed_in"] = df["listed_in"].fillna("")
    df["director"] = df["director"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["release_year"] = (
        pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    )

    print(f"  {len(df):,} titles will be embedded")

    # Build the index that gets served at recommendation time.
    # Storing genres as a pre-split list avoids repeated .split() on every request.
    titles_index = [
        {
            "title": row["title"],
            "type": row["type"],
            "genres": [g.strip() for g in row["listed_in"].split(",") if g.strip()],
            "description": row["description"],
            "release_year": int(row["release_year"]),
            "director": row["director"],
            "duration": str(row.get("duration", "")),
        }
        for _, row in df.iterrows()
    ]

    texts = [build_embedding_text(row) for _, row in df.iterrows()]

    print(f"\nComputing embeddings with {EMBEDDING_MODEL} ...")
    print("  (This downloads the model on first run — ~80MB, cached afterwards)")
    embedder = Embedder()
    embeddings = embedder.embed_batch(texts)

    print(f"\n  Embeddings shape: {embeddings.shape}")
    print(f"  Memory: {embeddings.nbytes / 1_048_576:.1f} MB")

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(str(EMBEDDINGS_PATH), embeddings)
    with open(TITLES_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(titles_index, f, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {EMBEDDINGS_PATH}")
    print(f"  {TITLES_INDEX_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
