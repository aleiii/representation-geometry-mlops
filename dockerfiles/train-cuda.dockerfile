# CUDA Training Docker image for neural representation geometry experiments
# Usage:
#   docker build -t rep-geom-train:cuda -f dockerfiles/train-cuda.dockerfile .
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs rep-geom-train:cuda
#
# All outputs (logs, checkpoints, configs, wandb) are saved to outputs/{date}/{time}/

# Stage 1: Get Python from official image
FROM python:3.12-slim AS python-base

# Stage 2: Get uv from official image
FROM ghcr.io/astral-sh/uv:latest AS uv

# Stage 3: NVIDIA CUDA runtime with Python and uv
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Copy Python from official image
COPY --from=python-base /usr/local /usr/local

# Copy uv from official image
COPY --from=uv /uv /uvx /usr/local/bin/

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project configuration first (for layer caching)
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies with CUDA PyTorch
RUN uv sync --no-dev --locked --no-cache --no-install-project \
    --index pytorch-cpu=https://download.pytorch.org/whl/cu124

# Copy source code
COPY src/ src/
COPY configs/ configs/

# Install the project (dependencies are already cached)
RUN uv sync --no-dev --locked --no-cache \
    --index pytorch-cpu=https://download.pytorch.org/whl/cu124

# Create necessary directories
RUN mkdir -p data/raw data/processed outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV HYDRA_FULL_ERROR=1

# Default command: train with MLP on CIFAR-10
# Override with: docker run --gpus all rep-geom-train:cuda model=resnet18
ENTRYPOINT ["uv", "run", "rep-geom-train"]
CMD ["model=mlp", "data=cifar10"]
