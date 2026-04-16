import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from .config import EMBEDDINGS_PATH, FRONTEND_DIR, TITLES_INDEX_PATH
from .embedder import Embedder
from .models import HealthResponse, RecommendRequest, RecommendResponse
from .recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load everything once at startup. The heavy work (embedding computation,
    model download) was done at Docker build time — this just reads files
    from disk and warms the SentenceTransformer from local cache.
    """
    logger.info("Loading pre-computed embeddings from %s", EMBEDDINGS_PATH)
    embeddings = np.load(str(EMBEDDINGS_PATH))

    with open(TITLES_INDEX_PATH, encoding="utf-8") as f:
        titles_index = json.load(f)

    logger.info(
        "Loaded %d titles, embeddings shape %s", len(titles_index), embeddings.shape
    )

    embedder = Embedder()
    app.state.recommender = Recommender(embedder, embeddings, titles_index)
    app.state.titles_count = len(titles_index)

    logger.info("Startup complete, ready to serve.")
    yield


app = FastAPI(
    title="Movie Recommender",
    description="Three-stage NL→semantic→rerank recommendation pipeline over Netflix catalog",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", titles_loaded=request.app.state.titles_count)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
    result = request.app.state.recommender.recommend(body.query, body.pipeline_mode)
    return RecommendResponse(
        query=body.query,
        interpreted_as=result["interpreted_as"],
        recommendations=result["recommendations"],
        pipeline_used=body.pipeline_mode,
    )


# Explicit route for "/" so the API routes above aren't shadowed by a catch-all mount.
# We only have one HTML file so a FileResponse is simpler than StaticFiles.
@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
