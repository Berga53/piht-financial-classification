# PIHT sparse classification

This repository turns the PIHT method from the sparsity chapter into a binary
classifier and evaluates it with the experimental structure of the final
Bankitalia chapter.

The model solves

\[
\min_{w,b}\; \frac{1}{n}\sum_i \alpha_i
\left[\log(1+e^{x_i^T w+b})-y_i(x_i^T w+b)\right]
+ \frac{\lambda_2}{2}\lVert w\rVert_2^2
\quad\text{subject to}\quad \lVert w\rVert_0\le K.
\]

Here `K` is the exact feature budget. PIHT takes an adaptive stochastic-gradient
step and hard-thresholds the coefficient vector after every proposal. The
intercept is never thresholded. Balanced observation weights handle the rare
positive class.

## What is reused

From `/Users/matteobergamaschi/Desktop/dott/SCSO/Sparsity`:

- independent mini-batches for the gradient and acceptance test;
- the adaptive radius `delta` and accept/reject rule;
- the increasing batch-size schedule;
- hard thresholding after each proposed step.

From `/Users/matteobergamaschi/Desktop/bankit_git/classification.ipynb`:

- municipality/year panel construction;
- input depths 4, 5, and 6 and target depths 1, 2, and 3;
- balanced classification and F1 threshold selection;
- precision, recall, F1, ROC-AUC, and PR-AUC reporting;
- L1 and L2 logistic-regression baselines.

## Important experimental correction

The chapter notebook randomly splits rows after concatenating overlapping time
windows. The same municipality can consequently occur in both train and test
sets. It also chooses the classification threshold on fitted training scores.
Both choices can make results optimistic.

The default experiment here uses municipality-disjoint train/validation/test
splits and chooses the threshold on validation predictions. Pass
`--split-strategy random` only to reproduce the old row-level split as a
sensitivity check.

## Setup

```bash
cd /Users/matteobergamaschi/Desktop/piht_classification_git
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The original data stay in `bankit_git`; they are not copied into Git.

## Prepare one Bankitalia design matrix

This example reproduces the BDAP feature family for period 2, six input years,
and a three-year prediction horizon:

```bash
piht-classification prepare-bankit \
  --bankit-root /Users/matteobergamaschi/Desktop/bankit_git \
  --output data/processed/bdap_p2_i6_h3.npz \
  --period 2 \
  --input-depth 6 \
  --target-depth 3 \
  --features bdap
```

Available feature families are `bdap`, `bdap-anticipazioni`,
`bdap-indicatori-anticipazioni`, and `indicatori`. The loader follows the
chapter notebook, including the excluded autonomous regions and its period
window definitions.

## Run the sparse experiment

```bash
piht-classification run \
  --dataset data/processed/bdap_p2_i6_h3.npz \
  --output results/bdap_p2_i6_h3.json \
  --k 5 10 20 40 80 \
  --iterations 1000 \
  --repeats 10
```

`K` must be selected on validation PR-AUC, never on the test set. A sensible
full thesis grid is:

- dataset family: the same families used in the final chapter;
- period: 1 and 2;
- input depth: 4, 5, 6;
- target depth: 1, 2, 3;
- sparsity budget: logarithmic grid from very small supports to the full model;
- 10 random seeds with identical splits for PIHT, L1 logistic, and L2 logistic.

Report both predictive performance and sparsity: median test PR-AUC, ROC-AUC,
F1, precision, recall, support size, and feature-selection frequency. Keep the
test set untouched until `K`, `lambda2`, PIHT settings, and the probability
threshold have been chosen.

## Smoke test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/run_synthetic.py
```

## Recommended next steps

1. Run one small Bankitalia configuration and inspect convergence/acceptance.
2. Tune `K` and `lambda2` in the inner validation split.
3. Freeze the protocol before running all 18 depth/period combinations.
4. Add the final chapter's tree, random-forest, boosting, CNN, and GNN results
   from the same outer splits if a direct chapter-to-chapter comparison is
   required.
5. Compare feature stability across repetitions, not just one selected support.

