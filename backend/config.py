import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Resolve paths relative to the package so this works whether run directly
# or via uvicorn from the project root
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

CSV_PATH = DATA_DIR / "netflix_data.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
TITLES_INDEX_PATH = DATA_DIR / "titles_index.json"

# Local embedding model — no API key, runs on CPU, ~80MB download
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq: small model for parsing (latency matters, task is structured extraction),
# large model for reranking (reasoning quality matters more than speed)
GROQ_PARSE_MODEL = "llama-3.1-8b-instant"
GROQ_RERANK_MODEL = "llama-3.3-70b-versatile"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

DEFAULT_PIPELINE_MODE = "full"

# top-20 gives the reranker enough variety to surface non-obvious matches
# without blowing the 70b model's prompt context with too many candidates
CANDIDATE_POOL_SIZE = 20
RESULT_COUNT = 5

# If hard filters leave fewer than this many titles, silently relax them.
# Prevents degenerate searches like "classic TV shows" from returning nothing.
FILTER_FLOOR = 50

# Map era labels to release year ranges
ERA_RANGES: dict[str, tuple[int, int]] = {
    "classic": (1900, 1979),
    "80s": (1980, 1989),
    "90s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "recent": (2020, 2099),
}
