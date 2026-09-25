#!/usr/bin/env python3
"""FastSC parameter search and fixed-parameter reproduction entry point."""

from __future__ import annotations

import os

# Parallelism is across datasets. Bound numerical libraries inside each worker.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utils import clustering_metrics, format_value, load_dataset, plot_clustering


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = (PROJECT_ROOT / "../data/D-Spec").resolve()
LARGE_DATA_ROOT = (PROJECT_ROOT / "../data/large_data").resolve()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results"
SEEDS = (42, 3407, 4079, 2024, 0)
DEFAULT_NEIGHBORS = (3, 5, 10, 15, 20, 30)
DEFAULT_T_CONST = (1, 2, 5, 10, 15, 20, 30)
DATASETS = {
    "spiral": DATA_ROOT / "spiral.mat",
    "4C": DATA_ROOT / "4C.mat",
    "AC": DATA_ROOT / "AC.mat",
    "RingG": DATA_ROOT / "RingG.mat",
    "complex9": DATA_ROOT / "complex9.mat",
    "cure-t2-4k": DATA_ROOT / "cure-t2-4k.mat",
    "landsat": DATA_ROOT / "landsat.mat",
    "spam": DATA_ROOT / "spam.mat",
    "waveform3": DATA_ROOT / "waveform3.mat",
    "pendigits": DATA_ROOT / "pendigits.mat",
    "USPS": DATA_ROOT / "USPS.mat",
    "letters": DATA_ROOT / "letters.mat",
    "MNIST": DATA_ROOT / "mnist.mat",
    "skin": DATA_ROOT / "skin.mat",
    "covertype": DATA_ROOT / "covertype.mat",
    "one_gaussian_10_one_line_5_2": DATA_ROOT / "one_gaussian_10_one_line_5_2.mat",
    "sparse_3_dense_3_dense_3": DATA_ROOT / "sparse_3_dense_3_dense_3.mat",
    "sparse_8_dense_1_dense_1": DATA_ROOT / "sparse_8_dense_1_dense_1.mat",
    "data_TB1M": LARGE_DATA_ROOT / "data_TB1M.mat",
    "data_SF2M": LARGE_DATA_ROOT / "data_SF2M.mat",
    "data_CC5M": LARGE_DATA_ROOT / "data_CC5M.mat",
    "data_CG10M": LARGE_DATA_ROOT / "data_CG10M.mat",
    "data_Flower20M": LARGE_DATA_ROOT / "data_Flower20M.mat",
}


def embedding_dimension(n_clusters: int) -> int:
    return max(2, math.ceil(math.log2(n_clusters)))


def iteration_count(n_samples: int, n_clusters: int, t_const: int) -> int:
    log_factor = max(1, math.ceil(math.log2(n_samples / n_clusters)))
    return int(t_const * log_factor)


def query_neighbors(features: np.ndarray, count: int, n_jobs: int = 1) -> np.ndarray:
    """Return exact neighbour indices, excluding each query point itself."""
    if count >= features.shape[0]:
        raise ValueError(f"neighbors={count} must be smaller than n={features.shape[0]}")
    # On very large datasets sklearn's dimensionality heuristic can select a
    # quadratic brute-force query even when an exact KD-tree is faster and far
    # more scalable (notably covertype). Both backends compute exact Euclidean
    # neighbours; this changes only the search data structure.
    algorithm = "kd_tree" if features.shape[0] >= 100_000 else "auto"
    model = NearestNeighbors(n_neighbors=count, algorithm=algorithm, n_jobs=n_jobs)
    model.fit(features)
    # X=None makes sklearn exclude the fitted sample itself, including when
    # duplicate observations are present.
    return np.asarray(model.kneighbors(return_distance=False), dtype=np.int64)


def build_knn_graph(indices: np.ndarray, count: int) -> scipy.sparse.csr_matrix:
    n_samples = indices.shape[0]
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), count)
    columns = indices[:, :count].reshape(-1)
    directed = scipy.sparse.csr_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(n_samples, n_samples),
    )
    adjacency = directed.maximum(directed.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def normalized_signless_operator(
    adjacency: scipy.sparse.csr_matrix,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Return M=(I+D^-1/2 A D^-1/2)/2 and the D^-1/2 diagonal."""
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(degrees <= 0):
        raise ValueError("The k-NN graph contains isolated vertices")
    inv_sqrt = 1.0 / np.sqrt(degrees)
    normalized = adjacency.multiply(inv_sqrt[:, None]).multiply(inv_sqrt[None, :])
    identity = scipy.sparse.eye(adjacency.shape[0], dtype=np.float32, format="csr")
    return ((normalized + identity) * 0.5).tocsr(), inv_sqrt


def fastsc_labels(
    operator: scipy.sparse.csr_matrix,
    inv_sqrt_degree: np.ndarray,
    n_clusters: int,
    t_const: int,
    seed: int,
) -> np.ndarray:
    """Algorithm 2: power method, D^-1/2 Y, then k-means."""
    vectors = np.random.RandomState(seed).normal(
        size=(operator.shape[0], embedding_dimension(n_clusters))
    )
    for _ in range(iteration_count(operator.shape[0], n_clusters, t_const)):
        vectors = operator @ vectors
    embedding = inv_sqrt_degree[:, None] * vectors
    return KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=1,
        random_state=seed,
    ).fit_predict(embedding)


def _write_search_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("neighbors", "t_const", "iterations", "nmi", "ari", "f1", "time")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for metric in ("nmi", "ari", "f1", "time"):
                output[metric] = format_value(output[metric])
            writer.writerow(output)


def search_dataset(
    name: str,
    path: Path,
    output_dir: Path,
    neighbor_grid: tuple[int, ...],
    t_const_grid: tuple[int, ...],
    knn_jobs: int,
) -> dict[str, object]:
    """Select parameters with seed 42, ranking by NMI, then ARI and F1."""
    search_started = perf_counter()
    dataset = load_dataset(path, name=name)
    all_neighbors = query_neighbors(dataset.features, max(neighbor_grid), knn_jobs)
    dimension = embedding_dimension(dataset.n_clusters)
    target_steps = {
        iteration_count(dataset.n_samples, dataset.n_clusters, value): value
        for value in t_const_grid
    }
    print(
        f"[{name}] search n={dataset.n_samples} d={dataset.n_features} "
        f"k={dataset.n_clusters} l={dimension}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    for neighbors in neighbor_grid:
        graph_started = perf_counter()
        adjacency = build_knn_graph(all_neighbors, neighbors)
        operator, inv_sqrt = normalized_signless_operator(adjacency)
        vectors = np.random.RandomState(42).normal(
            size=(dataset.n_samples, dimension)
        )
        for step in range(1, max(target_steps) + 1):
            vectors = operator @ vectors
            if step not in target_steps:
                continue
            t_const = target_steps[step]
            labels = KMeans(
                n_clusters=dataset.n_clusters,
                init="k-means++",
                n_init=1,
                random_state=42,
            ).fit_predict(inv_sqrt[:, None] * vectors)
            metrics = clustering_metrics(dataset.labels, labels)
            row = {
                "neighbors": int(neighbors),
                "t_const": int(t_const),
                "iterations": int(step),
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
                "f1": metrics["f1"],
                "time": perf_counter() - graph_started,
            }
            rows.append(row)
            score = (
                float(row["nmi"]),
                float(row["ari"]),
                float(row["f1"]),
                -int(neighbors),
                -int(t_const),
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "row": row,
                    "labels": np.asarray(labels, dtype=np.int64).copy(),
                    "edges": int(adjacency.nnz // 2),
                }
                print(
                    f"[{name}] best neighbors={neighbors} t_const={t_const} "
                    f"NMI={format_value(row['nmi'])} ARI={format_value(row['ari'])} "
                    f"F1={format_value(row['f1'])}",
                    flush=True,
                )

    if best is None:
        raise RuntimeError(f"No successful search candidate for {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_search_csv(output_dir / "search.csv", rows)
    best_row = best["row"]
    best_labels = best["labels"]
    np.save(output_dir / "best_labels_seed42.npy", best_labels)
    np.savetxt(
        output_dir / "best_labels_seed42.csv",
        best_labels,
        fmt="%d",
        delimiter=",",
        header="label",
        comments="",
    )
    plot_clustering(
        dataset.visualization_features,
        dataset.labels,
        best_labels,
        output_dir / "plots",
        seed=42,
    )
    parameters = {
        "dataset": name,
        "neighbors": int(best_row["neighbors"]),
        "t_const": int(best_row["t_const"]),
        "iterations": int(best_row["iterations"]),
        "embedding_dim": dimension,
        "selection_seed": 42,
        "selection_metric": "nmi_then_ari_then_f1_then_lower_complexity",
        "search_nmi": float(best_row["nmi"]),
        "search_ari": float(best_row["ari"]),
        "search_f1": float(best_row["f1"]),
        "undirected_edges": int(best["edges"]),
        "knn_algorithm": "kd_tree" if dataset.n_samples >= 100_000 else "auto",
        "search_seconds": perf_counter() - search_started,
    }
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2, ensure_ascii=False)
    return parameters


def reproduce_dataset(
    name: str,
    path: Path,
    output_dir: Path,
    parameters: dict[str, object],
    knn_jobs: int,
) -> dict[str, object]:
    """Run fixed best parameters with five seeds and summarize mean/std."""
    neighbors = int(parameters["neighbors"])
    t_const = int(parameters["t_const"])
    runs: list[dict[str, object]] = []
    metadata = None
    runs_path = output_dir / "runs.csv"
    if runs_path.is_file():
        with runs_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                runs.append(
                    {
                        "seed": int(row["seed"]),
                        "nmi": float(row["nmi"]),
                        "ari": float(row["ari"]),
                        "f1": float(row["f1"]),
                        "time": float(row["time"]),
                    }
                )
    completed_seeds = {int(run["seed"]) for run in runs}
    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"[{name}] seed={seed} using checkpoint", flush=True)
            continue
        started = perf_counter()
        dataset = load_dataset(path, name=name)
        indices = query_neighbors(dataset.features, neighbors, knn_jobs)
        adjacency = build_knn_graph(indices, neighbors)
        operator, inv_sqrt = normalized_signless_operator(adjacency)
        labels = fastsc_labels(operator, inv_sqrt, dataset.n_clusters, t_const, seed)
        total_seconds = perf_counter() - started
        metrics = clustering_metrics(dataset.labels, labels)
        metadata = dataset
        runs.append(
            {
                "seed": int(seed),
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
                "f1": metrics["f1"],
                "time": total_seconds,
            }
        )
        _write_runs(runs_path, runs)
        print(
            f"[{name}] seed={seed} NMI={format_value(metrics['nmi'])} "
            f"ARI={format_value(metrics['ari'])} F1={format_value(metrics['f1'])} "
            f"time={format_value(total_seconds)}",
            flush=True,
        )

    if metadata is None:
        metadata = load_dataset(path, name=name)
    _write_runs(runs_path, runs)

    if len(runs) != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} runs for {name}, found {len(runs)}")

    summary: dict[str, object] = {
        "dataset": name,
        "n_samples": metadata.n_samples,
        "n_features": metadata.n_features,
        "n_clusters": metadata.n_clusters,
        "best_params": {
            "neighbors": neighbors,
            "t_const": t_const,
            "embedding_dim": int(parameters["embedding_dim"]),
            "iterations": int(parameters["iterations"]),
        },
    }
    for metric in ("nmi", "ari", "f1", "time"):
        values = np.asarray([run[metric] for run in runs], dtype=np.float64)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=0))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def _write_runs(path: Path, runs: list[dict[str, object]]) -> None:
    """Checkpoint completed seed runs so an interrupted job can resume."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("seed", "nmi", "ari", "f1", "time"),
            lineterminator="\n",
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "seed": run["seed"],
                    "nmi": format_value(run["nmi"]),
                    "ari": format_value(run["ari"]),
                    "f1": format_value(run["f1"]),
                    "time": format_value(run["time"]),
                }
            )



def run_dataset(
    name: str,
    path_string: str,
    output_root_string: str,
    mode: str,
    neighbor_grid: tuple[int, ...],
    t_const_grid: tuple[int, ...],
    skip_existing: bool,
    knn_jobs: int,
) -> dict[str, object] | None:
    path = Path(path_string)
    output_dir = Path(output_root_string) / name
    parameter_path = output_dir / "best_params.json"
    summary_path = output_dir / "summary.json"

    if mode in {"search", "all"}:
        if skip_existing and parameter_path.is_file():
            with parameter_path.open(encoding="utf-8") as handle:
                parameters = json.load(handle)
            print(f"[{name}] using existing search result", flush=True)
        else:
            parameters = search_dataset(
                name, path, output_dir, neighbor_grid, t_const_grid, knn_jobs
            )
    else:
        if not parameter_path.is_file():
            raise FileNotFoundError(f"Run search first: {parameter_path}")
        with parameter_path.open(encoding="utf-8") as handle:
            parameters = json.load(handle)

    if mode in {"reproduce", "all"}:
        if skip_existing and summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return reproduce_dataset(name, path, output_dir, parameters, knn_jobs)
    return None


def aggregate(output_root: Path, destination: Path) -> None:
    fields = (
        "dataset",
        "n_samples",
        "n_features",
        "n_clusters",
        "nmi_mean",
        "nmi_std",
        "ari_mean",
        "ari_std",
        "f1_mean",
        "f1_std",
        "time_mean",
        "time_std",
        "best_params",
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in DATASETS:
            summary_path = output_root / name / "summary.json"
            if not summary_path.is_file():
                continue
            with summary_path.open(encoding="utf-8") as source:
                summary = json.load(source)
            row = {key: summary[key] for key in fields}
            row["best_params"] = json.dumps(
                row["best_params"], ensure_ascii=False, separators=(",", ":")
            )
            for key in fields:
                if key.endswith(("_mean", "_std")):
                    row[key] = format_value(row[key])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("search", "reproduce", "all"))
    parser.add_argument("--datasets", nargs="*", choices=tuple(DATASETS), default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--knn-jobs",
        type=int,
        default=4,
        help="threads used by each exact nearest-neighbour query",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--neighbors-grid", nargs="+", type=int, default=list(DEFAULT_NEIGHBORS)
    )
    parser.add_argument(
        "--t-const-grid", nargs="+", type=int, default=list(DEFAULT_T_CONST)
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse existing best_params.json and summary.json files",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.knn_jobs < 1:
        parser.error("--jobs and --knn-jobs must be positive")
    if any(value < 1 for value in args.neighbors_grid + args.t_const_grid):
        parser.error("search grids must contain positive integers")
    args.neighbors_grid = tuple(sorted(set(args.neighbors_grid)))
    args.t_const_grid = tuple(sorted(set(args.t_const_grid)))
    return args


def main() -> int:
    args = parse_args()
    selected = args.datasets or list(DATASETS)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_dataset,
                name,
                str(DATASETS[name]),
                str(output_root),
                args.mode,
                args.neighbors_grid,
                args.t_const_grid,
                args.skip_existing,
                args.knn_jobs,
            ): name
            for name in selected
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001
                failures[name] = f"{type(error).__name__}: {error}"
                print(f"[{name}] FAILED: {failures[name]}", flush=True)
                traceback.print_exception(error)

    if args.mode in {"reproduce", "all"}:
        aggregate(output_root, PROJECT_ROOT / "fastsc.csv")
    failure_path = output_root / "failures.json"
    if failures:
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2, ensure_ascii=False)
        return 1
    if failure_path.exists():
        failure_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
