# FastSC reproduction

Reproduction of *Fast and Simple Spectral Clustering in Theory and Practice*
on the 18 datasets in `../data/D-Spec`.

## Protocol

The implementation follows Algorithm 2 in the paper:

1. Min-Max normalize every feature column.
2. Construct an unweighted, union-symmetrized nearest-neighbour graph.
3. Use `l = max(2, ceil(log2(k)))` Gaussian vectors and the normalized
   signless operator `M = (I + D^(-1/2) A D^(-1/2)) / 2`.
4. Run the power method and cluster `D^(-1/2)Y` with KMeans (`n_init=1`).

Parameter search uses seed `42` and the following default grid:

- graph neighbours: `3, 5, 10, 15, 20, 30`
- power-method constant: `1, 2, 5, 10, 15, 20, 30`

The best setting maximizes NMI, with ties broken by ARI, Hungarian-aligned
macro-F1, and then lower parameter complexity. The chosen setting is then run
with seeds `42, 3407, 4079, 2024, 0`. Reported standard deviations use
`ddof=0`.

Runtime is measured independently for every reproduction seed, starting before
loading the MATLAB file and ending immediately after predicted labels are
obtained. It therefore includes loading, Min-Max normalization, nearest-neighbour
graph construction, operator construction, power iterations, and KMeans. Metric
calculation, saving, and plotting are excluded.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run search and reproduction for all datasets with dataset-level parallelism:

```bash
python main.py all --jobs 6 --knn-jobs 8
```

The phases can also be run separately or on selected datasets:

```bash
python main.py search --datasets spiral 4C AC --jobs 3
python main.py reproduce --datasets spiral 4C AC --jobs 3
```

Use `--skip-existing` to resume without recomputing completed search or
reproduction files. Search grids can be overridden with `--neighbors-grid` and
`--t-const-grid`.

## Outputs

Each `results/<dataset>/` directory contains:

- `search.csv`: all seed-42 parameter-search results;
- `best_params.json`: selected parameters and search metrics;
- `best_labels_seed42.npy` and `.csv`: seed-42 labels at the best setting;
- `plots/clustering_result.jpg` and `plots/true_labels.jpg`;
- `runs.csv`: the five fixed-parameter reproduction runs;
- `summary.json`: mean and population standard deviation.

The final aggregate is written to `fastsc.csv` in the requested dataset order.
Its columns contain dataset size, dimensionality, class count, NMI/ARI/macro-F1
and runtime mean/std, and the selected parameters.
