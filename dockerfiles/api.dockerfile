# API Docker image for serving model predictions and drift monitoring
# Usage:
#   docker build -t rep-geom-api:latest -f dockerfiles/api.dockerfile .
#   docker run -p 8000:8000 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/outputs:/app/outputs \
#     -v $(pwd)/api_logs:/app/api_logs \
#     -v $(pwd)/data:/app/data \
#     rep-geom-api:latest
#
# Endpoints:
#   - /health           - API health check
#   - /predict          - Run inference (logs predictions for drift monitoring)
#   - /predictions/stats - View prediction statistics
#   - /monitoring/drift - Check for data drift
#   - /monitoring/health - Monitoring health status

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

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Install the project (dependencies are already cached)
RUN uv sync --no-dev --locked --no-cache

# Create necessary directories
# - models/outputs: model checkpoints
# - api_logs: prediction database for drift monitoring
# - data/raw: reference datasets (CIFAR-10/STL-10)
RUN mkdir -p models outputs api_logs data/raw

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV MODEL_DIRS=/app/models:/app/outputs
ENV PREDICTION_LOG_DIR=/app/api_logs
ENV PREDICTION_LOG_ENABLED=true
ENV REFERENCE_DATA_DIR=/app/data/raw

# Volumes for persistent data
VOLUME ["/app/models", "/app/outputs", "/app/api_logs", "/app/data"]

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI server
ENTRYPOINT ["uv", "run", "uvicorn"]
CMD ["representation_geometry.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-config", "configs/api/logging.conf", "--access-log"]
