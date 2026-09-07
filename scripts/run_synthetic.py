"""Small end-to-end smoke experiment."""

from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import make_classification

from piht_classification.evaluation import run_experiment


X, y = make_classification(
    n_samples=800,
    n_features=30,
    n_informative=5,
    n_redundant=2,
    weights=[0.85, 0.15],
    class_sep=1.2,
    random_state=42,
)
groups = np.repeat(np.arange(200), 4)
results = run_experiment(
    X,
    y,
    groups,
    feature_names=[f"x{i}" for i in range(X.shape[1])],
    k_values=(3, 5, 10),
    iterations=300,
    repeats=1,
)
print(json.dumps(results["repetitions"][0], indent=2))
