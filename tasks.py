import os
from shlex import quote

from invoke import Context, task

WINDOWS = os.name == "nt"
DEFAULT_FULL_COMPARISON_SEEDS = "42,123,456,789,1011"
DEFAULT_DOCKER_IMAGE = "rep-geom-train:cuda"
DEFAULT_DOCKER_GPUS = "all"
DEFAULT_DOCKER_DATA_DIR = "data"
DEFAULT_DOCKER_OUTPUTS_DIR = "outputs"
DEFAULT_DOCKER_MULTIRUN_DIR = "multirun"


def _run(ctx: Context, cmd: str) -> None:
    ctx.run(cmd, echo=True, pty=not WINDOWS)


def _append_args(cmd: str, args: str) -> str:
    return f"{cmd} {args}" if args else cmd


def _run_docker_train(ctx: Context, train_args: str, detach: bool = False, multirun_dir: str | None = None) -> None:
    data_host = os.path.abspath(DEFAULT_DOCKER_DATA_DIR)
    outputs_host = os.path.abspath(DEFAULT_DOCKER_OUTPUTS_DIR)
    multirun_host = os.path.abspath(multirun_dir) if multirun_dir else None

    cmd = (
        f"docker run {'-d' if detach else '--rm'}"
        f" --gpus {quote(DEFAULT_DOCKER_GPUS)}"
        " -e WANDB_API_KEY"
        " -e WANDB_ENTITY"
        " -e WANDB_PROJECT"
        f" -v {quote(data_host)}:/app/data"
        f" -v {quote(outputs_host)}:/app/outputs"
        + (f" -v {quote(multirun_host)}:/app/multirun" if multirun_host else "")
        + f" {quote(DEFAULT_DOCKER_IMAGE)}"
    )
    _run(ctx, _append_args(cmd, train_args))


# Project commands
@task
def get_data(
    ctx: Context,
    datasets: str = "cifar10,stl10",
    data_dir: str = "data/raw",
) -> None:
    """Download datasets via torchvision."""
    normalized = [dataset.strip() for dataset in datasets.split(",") if dataset.strip()]
    if not normalized:
        raise ValueError("Provide at least one dataset (e.g. datasets=cifar10).")

    cmd = f"uv run rep-geom-data --data-dir {quote(data_dir)}"
    for dataset in normalized:
        cmd += f" --dataset {quote(dataset)}"

    _run(ctx, cmd)


@task
def data_report(
    ctx: Context,
    data_dir: str = "data/raw",
    out_dir: str = "cml-data",
    sample_size: int = 48,
) -> None:
    """Generate a dataset statistics report."""
    cmd = f"uv run rep-geom-report --data-dir {quote(data_dir)} --out-dir {quote(out_dir)} --sample-size {sample_size}"
    _run(ctx, cmd)


@task
def baseline_mlp_cifar10(ctx: Context) -> None:
    """Train MLP on CIFAR-10 using the baseline experiment config."""
    _run(ctx, "uv run rep-geom-train experiment=baseline_mlp_cifar10")


@task
def baseline_mlp_stl10(ctx: Context) -> None:
    """Train MLP on STL-10 using the baseline experiment config."""
    _run(ctx, "uv run rep-geom-train experiment=baseline_mlp_stl10")


@task
def baseline_resnet18_cifar10(ctx: Context) -> None:
    """Train ResNet-18 on CIFAR-10 using the baseline experiment config."""
    _run(ctx, "uv run rep-geom-train experiment=baseline_resnet18_cifar10")


@task
def baseline_resnet18_stl10(ctx: Context) -> None:
    """Train ResNet-18 on STL-10 using the baseline experiment config."""
    _run(ctx, "uv run rep-geom-train experiment=baseline_resnet18_stl10")


@task
def full_comparison(ctx: Context) -> None:
    """Run full comparison across models/datasets with a multi-seed sweep."""
    cmd = (
        "uv run rep-geom-train -m experiment=full_comparison "
        f"model=mlp,resnet18 data=cifar10,stl10 seed={quote(DEFAULT_FULL_COMPARISON_SEEDS)}"
    )
    _run(ctx, cmd)


@task
def test(ctx: Context, coverage: bool = True, report: bool = True, args: str = "") -> None:
    """Run tests (optionally with coverage)."""
    if coverage:
        cmd = "uv run coverage run -m pytest tests/"
    else:
        cmd = "uv run pytest tests/"

    _run(ctx, _append_args(cmd, args))

    if coverage and report:
        _run(ctx, "uv run coverage report -m -i")


@task
def lint(ctx: Context, fix: bool = False) -> None:
    """Run ruff checks."""
    cmd = "uv run ruff check src/ tests/"
    if fix:
        cmd += " --fix"
    _run(ctx, cmd)


@task
def format(ctx: Context, check: bool = True) -> None:
    """Run ruff formatter."""
    cmd = "uv run ruff format src/ tests/"
    if check:
        cmd += " --check"
    _run(ctx, cmd)


@task
def docker_build(ctx: Context, progress: str = "plain", cuda: bool = False) -> None:
    """Build docker images."""
    _run(
        ctx,
        f"docker build -t rep-geom-train:latest -f dockerfiles/train.dockerfile . --progress={progress}",
    )
    _run(
        ctx,
        f"docker build -t rep-geom-api:latest -f dockerfiles/api.dockerfile . --progress={progress}",
    )
    if cuda:
        _run(
            ctx,
            f"docker build -t rep-geom-train:cuda -f dockerfiles/train-cuda.dockerfile . --progress={progress}",
        )


@task
@task
def docker_baseline_mlp_cifar10(ctx: Context) -> None:
    """Run MLP CIFAR-10 baseline in Docker."""
    _run_docker_train(ctx, "experiment=baseline_mlp_cifar10")


@task
def docker_baseline_mlp_stl10(ctx: Context) -> None:
    """Run MLP STL-10 baseline in Docker."""
    _run_docker_train(ctx, "experiment=baseline_mlp_stl10")


@task
def docker_baseline_resnet18_cifar10(ctx: Context) -> None:
    """Run ResNet-18 CIFAR-10 baseline in Docker."""
    _run_docker_train(ctx, "experiment=baseline_resnet18_cifar10")


@task
def docker_baseline_resnet18_stl10(ctx: Context) -> None:
    """Run ResNet-18 STL-10 baseline in Docker."""
    _run_docker_train(ctx, "experiment=baseline_resnet18_stl10")


@task
def docker_full_comparison(ctx: Context) -> None:
    """Run full comparison in Docker with a multi-seed sweep."""
    args = (
        "-m experiment=full_comparison "
        f"model=mlp,resnet18 data=cifar10,stl10 seed={quote(DEFAULT_FULL_COMPARISON_SEEDS)}"
    )
    _run_docker_train(ctx, args, multirun_dir=DEFAULT_DOCKER_MULTIRUN_DIR)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    _run(ctx, "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build")


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    _run(ctx, "uv run mkdocs serve --config-file docs/mkdocs.yaml")
