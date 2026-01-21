import os
from shlex import quote

from invoke import Context, task

WINDOWS = os.name == "nt"


def _run(ctx: Context, cmd: str) -> None:
    ctx.run(cmd, echo=True, pty=not WINDOWS)


def _append_args(cmd: str, args: str) -> str:
    return f"{cmd} {args}" if args else cmd


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
def train(ctx: Context, args: str = "") -> None:
    """Train model via rep-geom-train. Pass Hydra overrides via args."""
    _run(ctx, _append_args("uv run rep-geom-train", args))


@task
def full_comparison(
    ctx: Context,
    seeds: str = "42,123,456,789,1011",
    args: str = "",
) -> None:
    """Run the full_comparison experiment across models, datasets, and seeds."""
    cmd = (
        f"uv run rep-geom-train -m experiment=full_comparison model=mlp,resnet18 data=cifar10,stl10 seed={quote(seeds)}"
    )
    _run(ctx, _append_args(cmd, args))


@task
def evaluate(ctx: Context, args: str = "") -> None:
    """Evaluate a trained model via rep-geom-evaluate."""
    _run(ctx, _append_args("uv run rep-geom-evaluate", args))


@task
def visualize(ctx: Context, args: str = "") -> None:
    """Visualize representations via rep-geom-visualize."""
    _run(ctx, _append_args("uv run rep-geom-visualize", args))


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
def docker_run_train(
    ctx: Context,
    image: str = "rep-geom-train:cuda",
    gpus: str = "all",
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    detach: bool = False,
    args: str = "",
) -> None:
    """Run the training container with W&B env passthrough."""
    data_host = os.path.abspath(data_dir)
    outputs_host = os.path.abspath(outputs_dir)

    cmd = (
        f"docker run {'-d' if detach else '--rm'}"
        f" --gpus {quote(gpus)}"
        " -e WANDB_API_KEY"
        " -e WANDB_ENTITY"
        " -e WANDB_PROJECT"
        f" -v {quote(data_host)}:/app/data"
        f" -v {quote(outputs_host)}:/app/outputs"
        f" {quote(image)}"
    )
    _run(ctx, _append_args(cmd, args))


@task
def docker_run_full_comparison(
    ctx: Context,
    image: str = "rep-geom-train:cuda",
    gpus: str = "all",
    data_dir: str = "data",
    outputs_dir: str = "outputs",
    seeds: str = "42,123,456,789,1011",
    detach: bool = False,
    args: str = "",
) -> None:
    """Run the full_comparison experiment in Docker (W&B env passthrough)."""
    base_args = f"-m experiment=full_comparison model=mlp,resnet18 data=cifar10,stl10 seed={quote(seeds)}"
    docker_args = _append_args(base_args, args)
    docker_run_train(
        ctx,
        image=image,
        gpus=gpus,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        detach=detach,
        args=docker_args,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    _run(ctx, "uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build")


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    _run(ctx, "uv run mkdocs serve --config-file docs/mkdocs.yaml")
