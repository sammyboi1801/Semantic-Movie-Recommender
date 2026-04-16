from typing import Optional

import numpy as np

from .config import ERA_RANGES, FILTER_FLOOR


def apply_hard_filters(
    titles_index: list[dict],
    embeddings: np.ndarray,
    content_type: Optional[str],
    era_preference: Optional[str],
) -> tuple[list[dict], np.ndarray]:
    """
    Slice the embedding matrix to only titles that satisfy hard constraints
    before cosine search runs.

    Filtering here rather than post-retrieval is intentional: cosine search
    over a restricted corpus means the candidates it returns are all valid,
    not just the top-20 before we throw half away. The ranking stays honest.
    """
    mask = np.ones(len(titles_index), dtype=bool)

    if content_type in ("Movie", "TV Show"):
        mask &= np.array([t["type"] == content_type for t in titles_index])

    if era_preference and era_preference in ERA_RANGES:
        low, high = ERA_RANGES[era_preference]
        mask &= np.array([
            low <= t.get("release_year", 0) <= high
            for t in titles_index
        ])

    indices = np.where(mask)[0]

    # If the filters are too aggressive (e.g. "classic TV shows" on a mostly-modern
    # dataset), silently fall back to the full corpus rather than returning a tiny set.
    if len(indices) < FILTER_FLOOR:
        return titles_index, embeddings

    return [titles_index[i] for i in indices], embeddings[indices]
