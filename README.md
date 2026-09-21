# PIHT sparse classification

PIHT fits a class-weighted logistic model with at most K nonzero coefficients.
The intercept is not thresholded. Adaptive stochastic gradients, independent
acceptance batches, and hard thresholding implement the sparse optimiser.

## Current municipal experiment

The 27 configurations retain one-year inputs, horizons 1–3, and the existing
feature families and candidate K grids. Each configuration now runs **once**:

- Stratified random **80% training / 20% testing**, seed **42**, as in the main
  Bank of Italy `classification.ipynb` training function.
- No separate validation partition. Every candidate K is fitted on the whole
  training partition. Select K by **training average precision**; ties favour
  the smaller K because the candidate grid is sorted.
- Select the probability threshold by **training F1** on the Bank notebook's
  grid `0, 0.01, ..., 1`, with a strict `probability > threshold` rule and the
  first threshold winning ties. Test outcomes do not select K or the threshold.
- Fit StandardScaler on training rows only. This PIHT optimisation preprocessing
  is retained; the Bank notebook does not apply the same additional scaling.
- Balanced training weights, stratified minibatches with inverse-probability
  correction, initial batch 256, no L2 penalty, and maximum 10,000 iterations.
- L1/L2 comparison baselines are disabled in the full grid.

Training selection is in-sample; it is not cross-validation. The same municipality
can appear in both partitions because the split is over municipality-window rows.
One-year inputs produce different rows from the Bank study's 4–6-year inputs, so
matching the split rule/seed does not mean identical held-out observations.

## Getting the data

This repository does not contain any data. The raw CSVs come from the companion
repository [`municipal-financial-distress`](https://github.com/Berga53/municipal-financial-distress).
Clone it and point `BANKIT_ROOT` at the checkout before running anything:

```bash
git clone https://github.com/Berga53/municipal-financial-distress.git
export BANKIT_ROOT=$PWD/municipal-financial-distress
```

Every entry point also accepts `--bankit-root PATH`, which overrides `BANKIT_ROOT`.

**Anticipazioni data are not in that repository.** `data/Anticipazioni/` is
confidential (Banca d'Italia data-sharing agreement), so it is gitignored there and
you will not get it from the clone. Ask Matteo for the folder and place it at
`$BANKIT_ROOT/data/Anticipazioni/`. Without it, the `anticipazioni` feature sets cannot
be built, and the source hash check will refuse to run because the
hash covers every CSV under `data/`.

`data/processed/` and `results/` are generated locally and are gitignored. Results
derived from the anticipazioni data are covered by the same agreement; do not commit
or share them.

## National sample and source files

The data loader includes all regions by default. All predictor families retain
the same municipality roster from `$BANKIT_ROOT/data/comuni.csv`. Indicator files
are reindexed to that roster and missing indicator entries are filled with zero;
missing non-indicator values still raise an error rather than dropping rows.

The inspected source roster and the thesis reference panel contain **7,773
municipalities**. The saved configuration records this count and a source-data hash.
If the source roster is corrected later, regenerate the configuration from the
new data before rerunning. The grid script refuses changed source files rather than
silently combining results from different datasets.

ReadyBDAP is the Impegni/Accertamenti subset of the shared financial CSVs. The
loader itself does not establish that those CSVs are forecast-budget data.

## Run all 27 configurations

With no experiment options, the grid script runs every experiment in
`configs/experiments_10000.json` (10,000 iterations, seed 42, random 80/20 split):

```bash
cd piht_classification_git   # with BANKIT_ROOT set, see "Getting the data"
PYTHONPATH=src caffeinate -i .venv/bin/python -u scripts/run_bdap_piht_grid.py
```

Each completed result is written atomically to `results/`. Matching completed
configurations are skipped on resume. Incompatible results are refused, not
overwritten. Processed datasets are rebuilt from the current source CSVs and an older
filtered NPZ is never reused. The run also refuses to start if the source CSVs differ
from the ones recorded in the configuration (source hash) or if a data shape changes.

The summary table contains the training-selected model for each of the three horizons.
Diagnostic figures additionally contain one held-out point and support for every
candidate sparsity value at each horizon; candidate test results do not enter selection.
`notebooks/inspect_results.ipynb` reads the saved results; it is not refreshed
automatically.

## Individual runs

`--features` selects a single feature set and a custom grid; `--periods`, `--input-depths`,
`--target-depths` and `--k` only apply together with it. `--bankit-root` overrides `$BANKIT_ROOT`.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_bdap_piht_grid.py \
  --features bdap --periods 1 --input-depths 1 --target-depths 1 \
  --k 5 10 15 20 25 30 35 36 --iterations 10000 --repeats 1 --seed 42 \
  --results-dir results_single --data-dir data/processed_single
```

The installed `piht-classification prepare-bankit` and `run` commands also use
the new defaults. `prepare-bankit --exclude-autonomous-regions` explicitly opts
into the old regional subset; the national grid never uses that option.
The optional group split remains available as a separate design.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Tests check exact Bank-style split indices and threshold decisions, training-only
selection, common indicator cohorts, sparse optimisation, and protocol-aware
resume behaviour. No complete municipal training run is performed by the tests.
