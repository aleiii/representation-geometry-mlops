# Training Docker image for neural representation geometry experiments
# Usage:
#   docker build -t rep-geom-train:latest -f dockerfiles/train.dockerfile .
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs rep-geom-train:latest
#
# All outputs (logs, checkpoints, configs, wandb) are saved to outputs/{date}/{time}/

# Use official uv image with Python 3.12 (includes uv pre-installed)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration first (for layer caching)
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies only (not the project itself)
# Note: We use CPU-only PyTorch for smaller image size
# For GPU support, modify pyproject.toml to use CUDA wheels
RUN uv sync --no-dev --locked --no-cache --no-install-project

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Install the project (dependencies are already cached)
RUN uv sync --no-dev --locked --no-cache

# Create necessary directories
RUN mkdir -p data/raw data/processed outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV HYDRA_FULL_ERROR=1

# Default command: train with MLP on CIFAR-10
# Override with: docker run rep-geom-train:latest model=resnet18
ENTRYPOINT ["uv", "run", "rep-geom-train"]
CMD ["model=mlp", "data=cifar10"]
