FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch must be installed before sentence-transformers, otherwise pip
# will resolve the CUDA build (~2GB) as the default. The CPU build is ~300MB.
RUN pip install --no-cache-dir \
    torch==2.11.0+cpu \
    torchvision==0.26.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies separately so pip can resolve against
# the already-installed CPU torch rather than pulling CUDA variants
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data and source before running precompute — these layers are cached
# unless the CSV or backend code changes, which keeps rebuild times fast
COPY data/netflix_data.csv ./data/netflix_data.csv
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# Compute embeddings and download the sentence-transformers model at build time.
# The container starts in <10 seconds because all heavy work is done here.
RUN python scripts/precompute.py

# Frontend and .env change more often than the ML pipeline — place them last
# so a UI edit doesn't invalidate the precompute cache layer
COPY frontend/ ./frontend/
COPY .env .env

EXPOSE 80

# Run from /app so "backend.main" resolves as a package import
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "80"]
