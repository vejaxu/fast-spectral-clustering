"""Data loading, evaluation, formatting, and plotting utilities for FastSC."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fastsc-matplotlib")

import h5py
import matplotlib
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


KEY_PAIRS = (
    ("data", "class"),
    ("data", "label"),
    ("fea", "gt"),
    ("X", "gtlabels"),
)


@dataclass
class Dataset:
    name: str
    path: Path
    features: np.ndarray
    visualization_features: np.ndarray
    labels: np.ndarray
    n_samples: int
    n_features: int
    n_clusters: int


def _find_keys(keys: set[str], filename: str) -> tuple[str, str]:
    for feature_key, label_key in KEY_PAIRS:
        if feature_key in keys and label_key in keys:
            return feature_key, label_key
    raise ValueError(f"Cannot identify features/labels in {filename}: {sorted(keys)}")


def _orient(features: np.ndarray, labels: np.ndarray, filename: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"{filename}: expected a 2-D feature matrix, got {features.shape}")
    if features.shape[0] == labels.size:
        oriented = features
    elif features.shape[1] == labels.size:
        oriented = features.T
    else:
        raise ValueError(
            f"{filename}: feature shape {features.shape} does not match {labels.size} labels"
        )
    _, encoded = np.unique(labels, return_inverse=True)
    return np.asarray(oriented), encoded.astype(np.int64, copy=False)


def minmax_normalize(features: np.ndarray) -> np.ndarray:
    """Column-wise Min-Max normalization with safe handling of constant columns."""
    matrix = np.array(features, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    minima = matrix.min(axis=0)
    ranges = matrix.max(axis=0) - minima
    ranges[ranges == 0] = 1.0
    matrix -= minima
    matrix /= ranges
    return matrix


def load_dataset(path: Path | str, name: str | None = None) -> Dataset:
    """Load any D-Spec MATLAB dataset and Min-Max normalize its features."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            feature_key, label_key = _find_keys(set(handle.keys()), path.name)
            features = np.asarray(handle[feature_key])
            labels = np.asarray(handle[label_key])
    else:
        keys = {key for key, _, _ in whosmat(path)}
        feature_key, label_key = _find_keys(keys, path.name)
        payload = loadmat(path, variable_names=[feature_key, label_key])
        features = payload[feature_key]
        labels = payload[label_key]

    features, labels = _orient(features, labels, path.name)
    raw = np.asarray(features, dtype=np.float32)
    n_samples, n_features = map(int, raw.shape)
    return Dataset(
        name=name or path.stem,
        path=path,
        features=minmax_normalize(raw),
        visualization_features=raw,
        labels=labels,
        n_samples=n_samples,
        n_features=n_features,
        n_clusters=int(np.unique(labels).size),
    )


def align_labels_hungarian(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Map predicted cluster IDs to true IDs using maximum-overlap assignment."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted labels must have the same shape")

    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    pred_values, pred_inverse = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((pred_values.size, true_values.size), dtype=np.int64)
    np.add.at(contingency, (pred_inverse, true_inverse), 1)
    pred_rows, true_columns = linear_sum_assignment(-contingency)
    mapping = {
        pred_values[row]: true_values[column]
        for row, column in zip(pred_rows, true_columns)
    }

    aligned = np.empty(y_pred.shape, dtype=np.int64)
    unmatched_start = int(true_values.min()) - pred_values.size - 1
    for offset, value in enumerate(pred_values):
        aligned[y_pred == value] = mapping.get(value, unmatched_start - offset)
    return aligned


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    aligned = align_labels_hungarian(y_true, y_pred)
    return {
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, aligned, average="macro", zero_division=0)),
        "aligned_labels": aligned,
    }


def format_value(value: float) -> str:
    rounded = Decimal(str(float(value))).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    return f"{rounded:.4f}"


def _sample_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.size <= limit:
        return np.arange(labels.size)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    unique, counts = np.unique(labels, return_counts=True)
    remaining = limit
    for position, (label, count) in enumerate(zip(unique, counts)):
        if position == unique.size - 1:
            take = remaining
        else:
            take = max(1, int(round(limit * int(count) / labels.size)))
            take = min(take, remaining - (unique.size - position - 1))
        candidates = np.flatnonzero(labels == label)
        chosen = rng.choice(candidates, size=min(take, candidates.size), replace=False)
        selected.append(chosen)
        remaining -= chosen.size
    indices = np.concatenate(selected)
    if indices.size < limit:
        unused = np.setdiff1d(np.arange(labels.size), indices, assume_unique=False)
        indices = np.concatenate(
            (indices, rng.choice(unused, size=limit - indices.size, replace=False))
        )
    return np.sort(indices[:limit])


def _project(values: np.ndarray, seed: int) -> np.ndarray:
    if values.shape[1] == 1:
        return np.column_stack((values[:, 0], np.zeros(values.shape[0])))
    if values.shape[1] == 2:
        return values
    if values.shape[0] < 3:
        return values[:, :2]
    return TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate=200.0,
        perplexity=min(30.0, float(values.shape[0] - 1)),
    ).fit_transform(values)


def _scatter(points: np.ndarray, labels: np.ndarray, path: Path) -> None:
    unique = np.unique(labels)
    if unique.size <= 10:
        colors = [plt.get_cmap("tab10")(index) for index in range(unique.size)]
    elif unique.size <= 20:
        colors = [plt.get_cmap("tab20")(index) for index in range(unique.size)]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, unique.size))

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111)
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap=ListedColormap(colors),
        alpha=0.7,
        s=15,
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_clustering(
    features: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    output_dir: Path | str,
    seed: int = 42,
    sample_size: int = 5000,
) -> None:
    """Save true-label and raw predicted-cluster plots using one shared projection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = _sample_indices(true_labels, sample_size, seed)
    values = np.asarray(features)[indices]
    projected = _project(values, seed)
    _scatter(projected, np.asarray(predicted_labels)[indices], output_dir / "clustering_result.jpg")
    _scatter(projected, np.asarray(true_labels)[indices], output_dir / "true_labels.jpg")
