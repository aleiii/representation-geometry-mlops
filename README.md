# Neural Representation Geometry Stability

> Reproducible experiments studying representation geometry stability across random seeds.

## Overview

The goal of this project is to investigate **the stability of internal representations in neural networks under varying training seeds**. We are going to understand the core question: To what extent does the geometry of learned representations change across runs, even when performance remains constant? 

We will analyze models trained on two datasets:

- **CIFAR-10**: A standard benchmark with 32×32 natural images across 10 classes
- **STL-10**: A more complex dataset with 96×96 images and fewer labeled examples

By comparing the geometry of internal activations across models trained with different seeds, we aim to quantify representational stability layer-by-layer.

Initially, the models used will be:

- A **Multi-Layer Perceptron (MLP)** to provide a baseline
- A **ResNet-18** convolutional neural network for deeper and more realistic representation analysis

The project uses PyTorch as the core framework and incorporates PyTorch Lightning for modular training. Experiments will be evaluated using metrics such as **Centered Kernel Alignment (CKA)**, **SVCCA**, and **Procrustes distance** to capture representation similarity.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file (dependencies managed by uv)
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
