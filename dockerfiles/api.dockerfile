# API Docker image for serving model predictions
# Usage:
#   docker build -t rep-geom-api:latest -f dockerfiles/api.dockerfile .
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models rep-geom-api:latest

# Use official uv image with Python 3.12 (includes uv pre-installed)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration first (for layer caching)
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies only (not the project itself)
RUN uv sync --no-dev --locked --no-cache --no-install-project

# Copy source code
COPY src/ src/

# Install the project (dependencies are already cached)
RUN uv sync --no-dev --locked --no-cache

# Create necessary directories (API searches models/ and outputs/ by default)
RUN mkdir -p models outputs logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV MODEL_DIRS=/app/models:/app/outputs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
ENTRYPOINT ["uv", "run", "uvicorn"]
CMD ["representation_geometry.api:app", "--host", "0.0.0.0", "--port", "8000"]
