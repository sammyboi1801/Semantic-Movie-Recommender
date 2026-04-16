from typing import Optional

from pydantic import BaseModel, field_validator


class RecommendRequest(BaseModel):
    query: str
    pipeline_mode: str = "full"

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v.strip()

    @field_validator("pipeline_mode")
    @classmethod
    def pipeline_mode_valid(cls, v: str) -> str:
        if v not in {"fast", "balanced", "full"}:
            raise ValueError("pipeline_mode must be 'fast', 'balanced', or 'full'")
        return v


class ParsedIntent(BaseModel):
    """Structured representation of what the user is actually looking for."""
    mood: Optional[str] = None
    themes: list[str] = []
    genres: list[str] = []
    # None means "don't filter by type" — distinct from an explicit Movie/TV Show request
    content_type: Optional[str] = None
    era_preference: Optional[str] = None
    enriched_query: str


class Recommendation(BaseModel):
    title: str
    type: str
    genres: list[str]
    description: str
    release_year: int
    # Only present in balanced/full modes where a reranker or enrichment ran
    match_reason: Optional[str] = None
    confidence: Optional[float] = None


class RecommendResponse(BaseModel):
    query: str
    interpreted_as: Optional[ParsedIntent] = None
    recommendations: list[Recommendation]
    pipeline_used: str


class HealthResponse(BaseModel):
    status: str
    titles_loaded: int
