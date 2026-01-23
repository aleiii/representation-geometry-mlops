# Neural Representation Geometry Stability

> Reproducible experiments studying representation geometry stability across random seeds.

## Overview

The goal of this project is to investigate **the stability of internal representations in neural networks under varying training seeds**. We aim to answer the core question: *To what extent does the geometry of learned representations change across runs, even when performance remains constant?*

We analyze models trained on two datasets:

- **CIFAR-10**: A standard benchmark with 32×32 natural images across 10 classes
- **STL-10**: A more complex dataset with 96×96 images and fewer labeled examples

By comparing the geometry of internal activations across models trained with different seeds, we quantify representational stability layer-by-layer.

The models used are:

- A **Multi-Layer Perceptron (MLP)** to provide a baseline
- A **ResNet-18** convolutional neural network for deeper representation analysis

## MLOps Stack

| Category | Tools |
|----------|-------|
| Package Management | [uv](https://github.com/astral-sh/uv) |
| Training Framework | [PyTorch Lightning](https://lightning.ai/) |
| Configuration | [Hydra](https://hydra.cc/) |
| Experiment Tracking | [Weights & Biases](https://wandb.ai/) |
| Data Versioning | [DVC](https://dvc.org/) with GCS backend |
| Code Quality | [Ruff](https://github.com/astral-sh/ruff), [pre-commit](https://pre-commit.com/) |
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Monitoring | [Prometheus](https://prometheus.io/), [Evidently](https://www.evidentlyai.com/) |
| Containerization | Docker (CPU, CUDA, API images) |
| CI/CD | GitHub Actions |
| Cloud | Google Cloud Platform (Compute Engine, Cloud Run, Artifact Registry, Cloud Storage) |
| Task Runner | [Invoke](https://www.pyinvoke.org/) |

## Project Structure

```txt
├── .github/                  # GitHub Actions workflows
│   ├── dependabot.yaml
│   └── workflows/
│       ├── cml-data.yaml         # CML data reports on PRs
│       ├── data-change.yaml      # DVC data change detection
│       ├── docker-build.yaml     # Docker image builds
│       ├── linting.yaml          # Ruff linting
│       ├── model-registry-change.yaml  # W&B model registry webhook
│       ├── pre-commit-update.yaml      # Pre-commit autoupdate
│       └── tests.yaml            # Unit tests
├── .dvc/                     # DVC configuration
├── configs/                  # Hydra configuration files
│   ├── config.yaml               # Main config
│   ├── api/                      # API configs
│   ├── data/                     # Dataset configs (cifar10, stl10)
│   ├── experiment/               # Experiment configs
│   └── model/                    # Model configs (mlp, resnet18)
├── data/                     # Data directory (managed by DVC)
│   └── raw.dvc
├── dockerfiles/              # Docker images
│   ├── api.dockerfile            # FastAPI serving
│   ├── train.dockerfile          # CPU training
│   └── train-cuda.dockerfile     # GPU training
├── docs/                     # MkDocs documentation
├── reports/                  # Project reports and figures
├── src/                      # Source code
│   └── representation_geometry/
│       ├── api.py                # FastAPI application
│       ├── data.py               # Data loading utilities
│       ├── dataset.py            # Dataset statistics
│       ├── drift.py              # Drift detection with Evidently
│       ├── evaluate.py           # Model evaluation
│       ├── model.py              # Model architectures (MLP, ResNet-18)
│       ├── registry.py           # W&B model registry integration
│       ├── train.py              # Training script
│       ├── utils.py              # Utilities
│       └── visualize.py          # Visualization tools
├── tests/                    # Test suite
│   ├── performancetests/         # Locust load tests
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_drift.py
│   └── test_model.py
├── cloudbuild.yaml           # GCP Cloud Build config
├── pyproject.toml            # Project dependencies (managed by uv)
├── tasks.py                  # Invoke task definitions
└── uv.lock                   # Locked dependencies
```

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized training/serving)
- NVIDIA GPU + CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/representation-geometry-mlops.git
   cd representation-geometry-mlops
   ```

2. Install dependencies:
   ```bash
   uv sync --dev
   ```

3. Download datasets:
   ```bash
   uv run invoke get-data
   ```

4. (Optional) Pull data from DVC remote:
   ```bash
   dvc pull
   ```

### CUDA Support

For GPU training on Linux with CUDA 12.4:
```bash
uv sync --dev --reinstall --index pytorch-cuda=https://download.pytorch.org/whl/cu124
```

## Usage

### Training

Run training experiments using invoke commands:

```bash
# Train MLP on CIFAR-10
uv run invoke baseline-mlp-cifar10

# Train ResNet-18 on CIFAR-10
uv run invoke baseline-resnet18-cifar10

# Train MLP on STL-10
uv run invoke baseline-mlp-stl10

# Train ResNet-18 on STL-10
uv run invoke baseline-resnet18-stl10

# Full comparison across all models/datasets with multiple seeds
uv run invoke full-comparison
```

Or use the CLI directly with Hydra overrides:

```bash
# Basic training
uv run rep-geom-train experiment=baseline_resnet18_cifar10

# Override parameters
uv run rep-geom-train experiment=baseline_mlp_cifar10 trainer.max_epochs=20 seed=123

# Multi-run sweep
uv run rep-geom-train -m seed=42,123,456 model=mlp,resnet18 data=cifar10,stl10
```

### API

Start the FastAPI server locally:

```bash
uv run uvicorn representation_geometry.api:app --host 0.0.0.0 --port 8000
```

Available endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with device info |
| `GET /models` | List available models |
| `POST /predict` | Single image prediction |
| `POST /predict/batch` | Batch predictions |
| `GET /metrics` | Prometheus metrics |
| `GET /monitoring/drift` | Drift detection report |
| `GET /docs` | Interactive API documentation |

### Docker

Build Docker images:

```bash
# Build CPU training and API images
uv run invoke docker-build

# Build with CUDA support
uv run invoke docker-build --cuda
```

Run training in Docker:

```bash
# Run ResNet-18 CIFAR-10 training
uv run invoke docker-baseline-resnet18-cifar10

# Run full comparison
uv run invoke docker-full-comparison
```

Run API in Docker:

```bash
docker run -p 8000:8000 rep-geom-api:latest
```

## Configuration

Configuration is managed with Hydra. Key configuration groups:

- **model**: `mlp`, `resnet18`
- **data**: `cifar10`, `stl10`
- **experiment**: Pre-defined experiment configurations

Example configuration override:

```bash
uv run rep-geom-train \
  model=resnet18 \
  data=stl10 \
  trainer.max_epochs=50 \
  seed=42 \
  wandb.project=my-project
```

## Testing

Run tests with coverage:

```bash
uv run invoke test
```

Run tests without coverage:

```bash
uv run invoke test --no-coverage
```

Run load tests with Locust:

```bash
locust -f tests/performancetests/locustfile.py --host http://localhost:8000
```

## Code Quality

Run linting:

```bash
uv run invoke lint
```

Run formatting check:

```bash
uv run invoke format
```

Fix issues automatically:

```bash
uv run invoke lint --fix
uv run invoke format --no-check
```

Pre-commit hooks are configured for automatic checks on commit.

## Deployment

### Cloud Run

Deploy the API to Google Cloud Run:

```bash
# Build and push to Artifact Registry
gcloud builds submit --config cloudbuild.yaml

# Deploy to Cloud Run
gcloud run deploy rep-geom-api \
  --image europe-west1-docker.pkg.dev/PROJECT_ID/rep-geom/rep-geom-api:latest \
  --region europe-west1 \
  --allow-unauthenticated
```

### Compute Engine (GPU Training)

For GPU training on GCP:

1. Create a VM instance with GPU (e.g., NVIDIA L4, g2-standard-4)
2. SSH into the instance and clone the repository
3. Build the CUDA Docker image:
   ```bash
   uv run invoke docker-build --cuda
   ```
4. Run training:
   ```bash
   uv run invoke docker-baseline-resnet18-cifar10
   ```

## Monitoring

The project includes monitoring capabilities:

- **Prometheus Metrics**: Request latency, prediction confidence, inference time
- **Drift Detection**: Image feature extraction and comparison using Evidently
- **GCP Cloud Monitoring**: Alerting based on custom metrics

Check drift status:

```bash
uv run rep-geom-drift check
```

Generate reference data:

```bash
uv run rep-geom-drift generate-reference
```

## Documentation

Build and serve documentation:

```bash
# Build docs
uv run invoke build-docs

# Serve docs locally
uv run invoke serve-docs
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).

This project was developed as part of the [02476 MLOps course](https://skaftenicki.github.io/dtu_mlops/) at DTU.
