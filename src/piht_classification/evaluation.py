"""Bankitalia-style 80/20 evaluation; selection uses training data only."""

from __future__ import annotations

import warnings
from time import perf_counter
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


PROTOCOL = "piht_80_20_training_selection_per_s_v2"


def best_f1_threshold(y_true, probabilities) -> tuple[float, float]:
    """Match Bankitalia: training F1, grid 0:0.01:1, strict >, first tied maximum."""
    thresholds = np.arange(0, 1 + 1e-9, 0.01)
    scores = [f1_score(y_true, probabilities > t, zero_division=0) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


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


def _train_test_split(X, y, groups, strategy: str, seed: int):
    if strategy == "group":
        return _group_split(X, y, groups, seed)
    if strategy == "random":
        return train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed
        )
    raise ValueError("split strategy must be 'group' or 'random'")


def _fit_baseline(penalty, X_fit, y_fit, X_test, y_test, seed):
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
        training_probability = model.predict_proba(X_fit)[:, 1]
        test_probability = model.predict_proba(X_test)[:, 1]
    if not np.isfinite(model.coef_).all():
        raise FloatingPointError(f"{penalty} logistic baseline produced non-finite coefficients")
    if not np.isfinite(training_probability).all() or not np.isfinite(test_probability).all():
        raise FloatingPointError(f"{penalty} logistic baseline produced non-finite probabilities")
    threshold, _ = best_f1_threshold(y_fit, training_probability)
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
    repeats: int = 1,
    seed: int = 42,
    split_strategy: str = "random",
    l2: float = 0.0,
    batch_size_initial: int = 256,
    batch_sampling: str = "stratified",
    min_positive_fraction: float = 0.1,
    include_baselines: bool = True,
    verbose: bool = False,
    progress_every: int = 1000,
) -> dict:
    experiment_started = perf_counter()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    k_values = sorted(set(map(int, k_values)))
    if not k_values:
        raise ValueError("provide at least one K value")
    if k_values[0] < 0 or k_values[-1] > X.shape[1]:
        raise ValueError(f"K values must be between 0 and {X.shape[1]}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1")

    if verbose:
        print(
            f"PIHT experiment: {repeats} repetitions, {len(k_values)} K values, "
            f"{iterations} maximum iterations per fit",
            flush=True,
        )

    repetitions = []
    first_seed = seed
    for repetition_index, seed in enumerate(range(first_seed, first_seed + repeats), start=1):
        repetition_started = perf_counter()
        repetition_prefix = f"[repetition {repetition_index}/{repeats}, seed={seed}]"
        if verbose:
            print(f"{repetition_prefix} creating train/test split", flush=True)
        fit_idx, test_idx = _train_test_split(
            X, y, groups, split_strategy, seed
        )
        scaler = StandardScaler().fit(X[fit_idx])
        X_fit = scaler.transform(X[fit_idx])
        X_test = scaler.transform(X[test_idx])
        if verbose:
            print(
                f"{repetition_prefix} sizes: fit={len(fit_idx)}, "
                f"test={len(test_idx)}",
                flush=True,
            )

        candidates = []
        for k_position, k in enumerate(k_values, start=1):
            fit_prefix = (
                f"{repetition_prefix} [K {k_position}/{len(k_values)}: {k}]"
            )
            model = SparsePIHTLogisticClassifier(
                k=k,
                max_iter=iterations,
                batch_size_initial=batch_size_initial,
                batch_size_max=len(fit_idx),
                batch_sampling=batch_sampling,
                min_positive_fraction=min_positive_fraction,
                l2=l2,
                class_weight="balanced",
                random_state=seed,
            )
            fit_started = perf_counter()
            if verbose:
                print(f"{fit_prefix} fitting", flush=True)

            def report_iteration(state, *, _started=fit_started, _prefix=fit_prefix):
                iteration = int(state["iteration"])
                if iteration != 1 and iteration % progress_every != 0:
                    return
                print(
                    f"{_prefix} iteration {iteration}/{iterations}; "
                    f"loss={float(state['loss']):.6f}; "
                    f"batch={int(state['batch_size'])}; "
                    f"positives={int(state['gradient_positive_count'])}; "
                    f"support={int(state['support_size'])}; "
                    f"elapsed={perf_counter() - _started:.1f}s",
                    flush=True,
                )

            model.fit(
                X_fit,
                y[fit_idx],
                progress_callback=report_iteration if verbose else None,
            )
            fit_seconds = perf_counter() - fit_started
            training_probability = model.predict_proba(X_fit)[:, 1]
            training_ap = average_precision_score(y[fit_idx], training_probability)
            threshold, training_f1 = best_f1_threshold(
                y[fit_idx], training_probability
            )
            test_probability = model.predict_proba(X_test)[:, 1]
            test_metrics = binary_metrics(y[test_idx], test_probability, threshold)
            support = model.support_.tolist()
            candidates.append(
                {
                    "k": k,
                    "training_average_precision": float(training_ap),
                    "training_f1_at_threshold": training_f1,
                    "test_metrics": test_metrics,
                    "support_size": len(support),
                    "selected_features": [feature_names[index] for index in support],
                    "iterations_run": int(model.n_iter_[0]),
                    "acceptance_rate": float(
                        np.mean([row["accepted"] for row in model.history_])
                    )
                    if model.history_
                    else 0.0,
                    "fit_seconds": fit_seconds,
                }
            )
            if verbose:
                print(
                    f"{fit_prefix} finished in {fit_seconds:.1f}s; "
                    f"iterations={int(model.n_iter_[0])}; training AP={training_ap:.6f}",
                    flush=True,
                )

        training_scores = [item["training_average_precision"] for item in candidates]
        best_position = int(np.argmax(training_scores))
        selected_candidate = candidates[best_position]
        best_k = selected_candidate["k"]
        piht_metrics = dict(selected_candidate["test_metrics"])
        piht_metrics.update(
            {
                "selected_k": best_k,
                "support_size": selected_candidate["support_size"],
                "selected_features": selected_candidate["selected_features"],
                "training_f1_at_threshold": selected_candidate[
                    "training_f1_at_threshold"
                ],
                "iterations_run": selected_candidate["iterations_run"],
                "acceptance_rate": selected_candidate["acceptance_rate"],
                "selected_model_fit_seconds": selected_candidate["fit_seconds"],
            }
        )

        repetition = {
            "seed": seed,
            "sizes": {
                "train": len(fit_idx),
                "test": len(test_idx),
            },
            "k_selection": candidates,
            "piht": piht_metrics,
        }
        if include_baselines:
            repetition.update(
                {
                    "l1_logistic": _fit_baseline(
                        "l1",
                        X_fit,
                        y[fit_idx],
                        X_test,
                        y[test_idx],
                        seed,
                    ),
                    "l2_logistic": _fit_baseline(
                        "l2",
                        X_fit,
                        y[fit_idx],
                        X_test,
                        y[test_idx],
                        seed,
                    ),
                }
            )
        repetition["runtime_seconds"] = perf_counter() - repetition_started
        repetitions.append(repetition)
        if verbose:
            print(
                f"{repetition_prefix} selected K={best_k}; "
                f"test AP={piht_metrics['average_precision']:.6f}; "
                f"ROC AUC={piht_metrics['roc_auc']:.6f}; "
                f"finished in {repetition['runtime_seconds']:.1f}s",
                flush=True,
            )

    result = {
        "protocol": PROTOCOL,
        "seed": first_seed,
        "selection_data": "training",
        "scaling": "standard_fit_on_training",
        "split_strategy": split_strategy,
        "repeats": repeats,
        "k_values": k_values,
        "l2": l2,
        "iterations": iterations,
        "batch_size_initial": batch_size_initial,
        "batch_sampling": batch_sampling,
        "min_positive_fraction": min_positive_fraction,
        "include_baselines": include_baselines,
        "repetitions": repetitions,
        "runtime_seconds": perf_counter() - experiment_started,
    }
    if verbose:
        print(f"PIHT experiment finished in {result['runtime_seconds']:.1f}s", flush=True)
    return result
