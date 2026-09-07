"""Leakage-safe model selection and evaluation."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .estimator import SparsePIHTLogisticClassifier


def best_f1_threshold(y_true, probabilities, step: float = 0.01) -> tuple[float, float]:
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.arange(0.0, 1.0 + 1e-12, step):
        score = f1_score(y_true, probabilities > threshold, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold, best_f1


def binary_metrics(y_true, probabilities, threshold: float) -> dict:
    prediction = probabilities > threshold
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return {
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "pr_auc_trapezoid": float(trapezoid(precision[::-1], recall[::-1])),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }


def _group_split(X, y, groups, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        return next(splitter.split(X, y, groups))
    except ValueError:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        return next(splitter.split(X, y, groups))


def _three_way_split(X, y, groups, strategy: str, seed: int):
    indices = np.arange(len(y))
    if strategy == "group":
        train_val_local, test_local = _group_split(X, y, groups, seed)
        fit_local, val_within = _group_split(
            X[train_val_local], y[train_val_local], groups[train_val_local], seed + 1
        )
        fit = train_val_local[fit_local]
        validation = train_val_local[val_within]
        return fit, validation, test_local
    if strategy == "random":
        train_val, test = train_test_split(
            indices, test_size=0.2, stratify=y, random_state=seed
        )
        fit, validation = train_test_split(
            train_val, test_size=0.2, stratify=y[train_val], random_state=seed + 1
        )
        return fit, validation, test
    raise ValueError("split strategy must be 'group' or 'random'")


def _fit_baseline(penalty, X_fit, y_fit, X_validation, y_validation, X_test, y_test, seed):
    model = LogisticRegression(
        penalty=penalty,
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=10_000,
        random_state=seed,
    )
    # NumPy 2 can surface stale BLAS floating-point flags from sklearn's
    # safe_sparse_dot even when the returned arrays are finite. Silence only
    # that narrow warning and validate the outputs explicitly below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath"
        )
        model.fit(X_fit, y_fit)
        validation_probability = model.predict_proba(X_validation)[:, 1]
        test_probability = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(model.coef_).all():
        raise FloatingPointError(f"{penalty} logistic baseline produced non-finite coefficients")
    if not np.isfinite(validation_probability).all() or not np.isfinite(test_probability).all():
        raise FloatingPointError(f"{penalty} logistic baseline produced non-finite probabilities")
    threshold, _ = best_f1_threshold(y_validation, validation_probability)
    metrics = binary_metrics(y_test, test_probability, threshold)
    metrics["support_size"] = int(np.count_nonzero(model.coef_))
    return metrics


def run_experiment(
    X,
    y,
    groups,
    *,
    feature_names: list[str],
    k_values: Iterable[int],
    iterations: int = 1000,
    repeats: int = 10,
    split_strategy: str = "group",
    l2: float = 0.0,
    batch_size_initial: int = 64,
) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    k_values = sorted(set(map(int, k_values)))
    if not k_values:
        raise ValueError("provide at least one K value")
    if k_values[0] < 0 or k_values[-1] > X.shape[1]:
        raise ValueError(f"K values must be between 0 and {X.shape[1]}")

    repetitions = []
    for seed in range(repeats):
        fit_idx, validation_idx, test_idx = _three_way_split(
            X, y, groups, split_strategy, seed
        )
        scaler = StandardScaler().fit(X[fit_idx])
        X_fit = scaler.transform(X[fit_idx])
        X_validation = scaler.transform(X[validation_idx])
        X_test = scaler.transform(X[test_idx])

        candidates = []
        fitted_models = []
        for k in k_values:
            model = SparsePIHTLogisticClassifier(
                k=k,
                max_iter=iterations,
                batch_size_initial=batch_size_initial,
                batch_size_max=len(fit_idx),
                l2=l2,
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(X_fit, y[fit_idx])
            validation_probability = model.predict_proba(X_validation)[:, 1]
            validation_ap = average_precision_score(y[validation_idx], validation_probability)
            candidates.append({"k": k, "validation_average_precision": float(validation_ap)})
            fitted_models.append(model)

        best_position = int(np.argmax([item["validation_average_precision"] for item in candidates]))
        best_model = fitted_models[best_position]
        best_k = candidates[best_position]["k"]
        validation_probability = best_model.predict_proba(X_validation)[:, 1]
        threshold, validation_f1 = best_f1_threshold(y[validation_idx], validation_probability)
        test_probability = best_model.predict_proba(X_test)[:, 1]
        piht_metrics = binary_metrics(y[test_idx], test_probability, threshold)
        support = best_model.support_.tolist()
        piht_metrics.update(
            {
                "selected_k": best_k,
                "support_size": len(support),
                "selected_features": [feature_names[index] for index in support],
                "validation_f1_at_threshold": validation_f1,
                "iterations_run": int(best_model.n_iter_[0]),
                "acceptance_rate": float(
                    np.mean([row["accepted"] for row in best_model.history_])
                )
                if best_model.history_
                else 0.0,
            }
        )

        repetitions.append(
            {
                "seed": seed,
                "sizes": {
                    "fit": len(fit_idx),
                    "validation": len(validation_idx),
                    "test": len(test_idx),
                },
                "k_selection": candidates,
                "piht": piht_metrics,
                "l1_logistic": _fit_baseline(
                    "l1", X_fit, y[fit_idx], X_validation, y[validation_idx], X_test, y[test_idx], seed
                ),
                "l2_logistic": _fit_baseline(
                    "l2", X_fit, y[fit_idx], X_validation, y[validation_idx], X_test, y[test_idx], seed
                ),
            }
        )

    return {
        "split_strategy": split_strategy,
        "repeats": repeats,
        "k_values": k_values,
        "l2": l2,
        "iterations": iterations,
        "repetitions": repetitions,
    }
