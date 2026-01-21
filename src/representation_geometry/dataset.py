"""Dataset reporting utilities (used for CML reports)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer
from torchvision import datasets

matplotlib.use("Agg")


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    split: str
    num_samples: int
    class_names: list[str]
    class_counts: list[int]


def _save_image_grid(images: Iterable[np.ndarray], labels: Iterable[str], out_path: Path, ncols: int = 8) -> None:
    images_list = list(images)
    labels_list = list(labels)
    if len(images_list) == 0:
        return

    n = len(images_list)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 1.8))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= n:
            continue
        ax.imshow(images_list[idx])
        ax.set_title(labels_list[idx], fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_class_hist(summary: DatasetSummary, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    xs = np.arange(len(summary.class_names))
    ax.bar(xs, summary.class_counts)
    ax.set_xticks(xs)
    ax.set_xticklabels(summary.class_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(f"{summary.name} ({summary.split}) class distribution")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _cifar10_summaries(data_dir: Path) -> tuple[DatasetSummary, DatasetSummary]:
    train = datasets.CIFAR10(root=data_dir, train=True, download=False)
    test = datasets.CIFAR10(root=data_dir, train=False, download=False)

    class_names = list(train.classes)
    train_counts = np.bincount(np.array(train.targets), minlength=len(class_names)).tolist()
    test_counts = np.bincount(np.array(test.targets), minlength=len(class_names)).tolist()

    return (
        DatasetSummary(
            name="CIFAR-10",
            split="train",
            num_samples=len(train),
            class_names=class_names,
            class_counts=train_counts,
        ),
        DatasetSummary(
            name="CIFAR-10",
            split="test",
            num_samples=len(test),
            class_names=class_names,
            class_counts=test_counts,
        ),
    )


def _stl10_summaries(data_dir: Path) -> tuple[DatasetSummary, DatasetSummary]:
    train = datasets.STL10(root=data_dir, split="train", download=False)
    test = datasets.STL10(root=data_dir, split="test", download=False)

    class_names = list(train.classes)
    train_counts = np.bincount(np.array(train.labels), minlength=len(class_names)).tolist()
    test_counts = np.bincount(np.array(test.labels), minlength=len(class_names)).tolist()

    return (
        DatasetSummary(
            name="STL-10",
            split="train",
            num_samples=len(train),
            class_names=class_names,
            class_counts=train_counts,
        ),
        DatasetSummary(
            name="STL-10",
            split="test",
            num_samples=len(test),
            class_names=class_names,
            class_counts=test_counts,
        ),
    )


def dataset_statistics(data_dir: str | Path, out_dir: str | Path, sample_size: int = 48) -> None:
    """Create a small dataset report suitable for posting in PRs via CML."""
    data_dir_path = Path(data_dir)
    out_dir_path = Path(out_dir)
    figures_dir = out_dir_path / "figures"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cifar_train, cifar_test = _cifar10_summaries(data_dir_path)
    stl_train, stl_test = _stl10_summaries(data_dir_path)

    _save_class_hist(cifar_train, figures_dir / "cifar10_train_class_dist.png")
    _save_class_hist(cifar_test, figures_dir / "cifar10_test_class_dist.png")
    _save_class_hist(stl_train, figures_dir / "stl10_train_class_dist.png")
    _save_class_hist(stl_test, figures_dir / "stl10_test_class_dist.png")

    cifar_raw = datasets.CIFAR10(root=data_dir_path, train=True, download=False)
    stl_raw = datasets.STL10(root=data_dir_path, split="train", download=False)

    cifar_imgs = []
    cifar_lbls = []
    for i in range(min(sample_size, len(cifar_raw))):
        img, lbl = cifar_raw[i]
        cifar_imgs.append(np.asarray(img))
        cifar_lbls.append(str(cifar_raw.classes[lbl]))
    _save_image_grid(cifar_imgs, cifar_lbls, figures_dir / "cifar10_samples.png", ncols=8)

    stl_imgs = []
    stl_lbls = []
    for i in range(min(sample_size, len(stl_raw))):
        img, lbl = stl_raw[i]
        stl_imgs.append(np.asarray(img))
        stl_lbls.append(str(stl_raw.classes[int(lbl)]))
    _save_image_grid(stl_imgs, stl_lbls, figures_dir / "stl10_samples.png", ncols=8)

    report_path = out_dir_path / "report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Dataset report (CML)",
                "",
                f"Data directory: `{data_dir_path}`",
                "",
                "## CIFAR-10",
                f"- Train samples: {cifar_train.num_samples}",
                f"- Test samples: {cifar_test.num_samples}",
                "",
                "![](figures/cifar10_train_class_dist.png)",
                "![](figures/cifar10_test_class_dist.png)",
                "",
                "![](figures/cifar10_samples.png)",
                "",
                "## STL-10",
                f"- Train samples: {stl_train.num_samples}",
                f"- Test samples: {stl_test.num_samples}",
                "",
                "![](figures/stl10_train_class_dist.png)",
                "![](figures/stl10_test_class_dist.png)",
                "",
                "![](figures/stl10_samples.png)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


report_app = typer.Typer(add_completion=False, no_args_is_help=True)


@report_app.command()
def report(
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Directory containing raw datasets.")] = Path("data/raw"),
    out_dir: Annotated[Path, typer.Option("--out-dir", help="Output directory for the report.")] = Path("cml-data"),
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", help="Number of sample images per dataset."),
    ] = 48,
) -> None:
    """Generate a dataset statistics report."""
    dataset_statistics(data_dir, out_dir, sample_size=sample_size)


def main() -> None:
    report_app()


if __name__ == "__main__":
    main()
