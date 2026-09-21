"""Core PIHT operations for cardinality-constrained logistic regression."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PIHTConfig:
    k: int
    max_iter: int = 1000
    batch_size_initial: int = 256
    batch_size_max: int | None = None
    batch_sampling: str = "stratified"
    min_positive_fraction: float = 0.1
    l2: float = 0.0
    eta1: float = 1e-4
    eta2: float = 1e-4
    delta0: float = 1.0
    delta_max: float = 10.0
    gamma: float = 2.0
    min_delta: float = 1e-12
    random_state: int | None = None


@dataclass
class PIHTResult:
    coef: FloatArray
    intercept: float
    history: list[dict[str, float | int | bool]]
    n_iter: int


def hard_threshold(values: FloatArray, k: int) -> FloatArray:
    """Return a copy retaining exactly the k largest magnitudes (unless k >= p)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("hard_threshold expects a one-dimensional array")
    if not 0 <= k <= values.size:
        raise ValueError(f"k must be between 0 and {values.size}, got {k}")
    if k == values.size:
        return values.copy()
    result = np.zeros_like(values)
    if k:
        selected = np.argpartition(np.abs(values), -k)[-k:]
        result[selected] = values[selected]
    return result


def _sigmoid(scores: FloatArray) -> FloatArray:
    magnitude = np.exp(-np.abs(scores))
    return np.where(scores >= 0, 1.0 / (1.0 + magnitude), magnitude / (1.0 + magnitude))


def logistic_loss(
    coef: FloatArray,
    intercept: float,
    X: FloatArray,
    y: FloatArray,
    sample_weight: FloatArray,
    l2: float,
) -> float:
    # Some BLAS builds leave harmless floating-point status flags behind after
    # matrix multiplication. Suppress the flags locally, then explicitly reject
    # any genuinely non-finite result.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = X @ coef + intercept
        losses = np.logaddexp(0.0, scores) - y * scores
        weighted_loss = float(np.dot(sample_weight, losses) / sample_weight.sum())
    if not np.isfinite(weighted_loss):
        raise FloatingPointError("logistic loss is not finite")
    return weighted_loss + 0.5 * l2 * float(np.dot(coef, coef))


def logistic_gradient(
    coef: FloatArray,
    intercept: float,
    X: FloatArray,
    y: FloatArray,
    sample_weight: FloatArray,
    l2: float,
) -> tuple[FloatArray, float]:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        residual = (_sigmoid(X @ coef + intercept) - y) * sample_weight
        denominator = sample_weight.sum()
        coef_gradient = X.T @ residual / denominator + l2 * coef
        intercept_gradient = float(residual.sum() / denominator)
    if not np.isfinite(coef_gradient).all() or not np.isfinite(intercept_gradient):
        raise FloatingPointError("logistic gradient is not finite")
    return np.asarray(coef_gradient, dtype=float), intercept_gradient


def adaptive_batch_size(
    delta: float,
    delta0: float,
    batch_size_initial: int,
    batch_size_max: int,
) -> int:
    """Increase batch size as the trust-region radius contracts.

    ``batch_size_initial`` is the actual batch size while ``delta >= delta0``.
    """
    if delta <= 0:
        return batch_size_max
    exponent = max(0.0, delta0 / delta - 1.0)
    if exponent > 12:
        return batch_size_max
    return min(batch_size_max, max(1, int(batch_size_initial * (2.0**exponent))))


def sample_minibatch(
    rng: np.random.Generator,
    y: FloatArray,
    batch_size: int,
    *,
    strategy: str,
    min_positive_fraction: float,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Sample a batch and return inverse-probability weight corrections.

    Stratified batches never reduce the observed positive share and, when
    possible, contain at least ``min_positive_fraction`` positives. The
    corrections preserve the full-data class contribution, avoiding accidental
    double weighting when the estimator also uses balanced class weights.
    """
    y = np.asarray(y)
    n_samples = len(y)
    if not 1 <= batch_size <= n_samples:
        raise ValueError("batch_size must be between 1 and the number of samples")
    if strategy not in {"uniform", "stratified"}:
        raise ValueError("batch_sampling must be 'uniform' or 'stratified'")
    if not 0.0 <= min_positive_fraction < 1.0:
        raise ValueError("min_positive_fraction must be at least 0 and less than 1")

    if batch_size == n_samples:
        return np.arange(n_samples, dtype=np.int64), np.ones(n_samples, dtype=float)
    if strategy == "uniform":
        indices = rng.choice(n_samples, size=batch_size, replace=False)
        return np.asarray(indices, dtype=np.int64), np.ones(batch_size, dtype=float)
    if batch_size < 2:
        raise ValueError("stratified batching requires a batch size of at least 2")

    positive_pool = np.flatnonzero(y == 1)
    negative_pool = np.flatnonzero(y == 0)
    if not len(positive_pool) or not len(negative_pool):
        raise ValueError("stratified batching requires both classes")

    observed_fraction = len(positive_pool) / n_samples
    target_fraction = max(observed_fraction, min_positive_fraction)
    positive_count = max(1, int(np.ceil(batch_size * target_fraction)))
    positive_count = min(positive_count, len(positive_pool), batch_size - 1)
    negative_count = batch_size - positive_count

    if negative_count > len(negative_pool):
        negative_count = len(negative_pool)
        positive_count = batch_size - negative_count
    if positive_count > len(positive_pool):
        positive_count = len(positive_pool)
        negative_count = batch_size - positive_count

    positive_indices = rng.choice(positive_pool, size=positive_count, replace=False)
    negative_indices = rng.choice(negative_pool, size=negative_count, replace=False)
    indices = np.concatenate((positive_indices, negative_indices))
    corrections = np.concatenate(
        (
            np.full(positive_count, len(positive_pool) / positive_count),
            np.full(negative_count, len(negative_pool) / negative_count),
        )
    )
    order = rng.permutation(batch_size)
    return indices[order].astype(np.int64, copy=False), corrections[order]


def fit_piht_logistic(
    X: FloatArray,
    y: FloatArray,
    sample_weight: FloatArray,
    config: PIHTConfig,
    progress_callback: Callable[[dict[str, float | int | bool]], None] | None = None,
) -> PIHTResult:
    """Fit sparse logistic regression with the chapter's PIHT accept/reject rule."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    n_samples, n_features = X.shape

    if not 0 <= config.k <= n_features:
        raise ValueError(f"k must be between 0 and {n_features}, got {config.k}")
    if config.gamma <= 1:
        raise ValueError("gamma must be greater than 1")
    if config.delta0 <= 0 or config.delta_max <= 0:
        raise ValueError("delta0 and delta_max must be positive")
    if config.batch_size_initial < 1:
        raise ValueError("batch_size_initial must be positive")
    if config.batch_sampling not in {"uniform", "stratified"}:
        raise ValueError("batch_sampling must be 'uniform' or 'stratified'")
    if not 0.0 <= config.min_positive_fraction < 1.0:
        raise ValueError("min_positive_fraction must be at least 0 and less than 1")
    if sample_weight.shape != (n_samples,) or np.any(sample_weight <= 0):
        raise ValueError("sample_weight must contain one positive value per observation")

    batch_size_max = min(config.batch_size_max or n_samples, n_samples)
    rng = np.random.default_rng(config.random_state)
    coef = np.zeros(n_features, dtype=float)
    prevalence = np.clip(np.average(y, weights=sample_weight), 1e-8, 1 - 1e-8)
    intercept = float(np.log(prevalence / (1.0 - prevalence)))
    delta = min(config.delta0, config.delta_max)
    history: list[dict[str, float | int | bool]] = []

    for iteration in range(config.max_iter):
        batch_size = adaptive_batch_size(
            delta, config.delta0, config.batch_size_initial, batch_size_max
        )
        gradient_indices, gradient_correction = sample_minibatch(
            rng,
            y,
            batch_size,
            strategy=config.batch_sampling,
            min_positive_fraction=config.min_positive_fraction,
        )
        grad_coef, grad_intercept = logistic_gradient(
            coef,
            intercept,
            X[gradient_indices],
            y[gradient_indices],
            sample_weight[gradient_indices] * gradient_correction,
            config.l2,
        )
        gradient_norm = float(
            np.sqrt(np.dot(grad_coef, grad_coef) + grad_intercept * grad_intercept)
        )
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("PIHT produced a non-finite gradient")
        if gradient_norm == 0:
            break

        step_scale = min(delta / gradient_norm, 1.0)
        candidate_coef = hard_threshold(coef - step_scale * grad_coef, config.k)
        candidate_intercept = intercept - step_scale * grad_intercept

        acceptance_indices, acceptance_correction = sample_minibatch(
            rng,
            y,
            batch_size,
            strategy=config.batch_sampling,
            min_positive_fraction=config.min_positive_fraction,
        )
        current_loss = logistic_loss(
            coef,
            intercept,
            X[acceptance_indices],
            y[acceptance_indices],
            sample_weight[acceptance_indices] * acceptance_correction,
            config.l2,
        )
        candidate_loss = logistic_loss(
            candidate_coef,
            candidate_intercept,
            X[acceptance_indices],
            y[acceptance_indices],
            sample_weight[acceptance_indices] * acceptance_correction,
            config.l2,
        )
        denominator = gradient_norm * delta
        ratio = (current_loss - candidate_loss) / denominator if denominator else -np.inf
        accepted = bool(ratio > config.eta1 and gradient_norm > config.eta2 * delta)

        if accepted:
            coef = candidate_coef
            intercept = candidate_intercept
            delta = min(delta * config.gamma, config.delta_max)
        else:
            delta /= config.gamma

        state = {
            "iteration": iteration + 1,
            "accepted": accepted,
            "loss": candidate_loss if accepted else current_loss,
            "candidate_loss": candidate_loss,
            "gradient_norm": gradient_norm,
            "ratio": float(ratio),
            "delta": delta,
            "batch_size": batch_size,
            "gradient_positive_count": int(y[gradient_indices].sum()),
            "acceptance_positive_count": int(y[acceptance_indices].sum()),
            "support_size": int(np.count_nonzero(coef)),
        }
        history.append(state)
        if progress_callback is not None:
            progress_callback(state)
        if delta < config.min_delta:
            break

    return PIHTResult(coef=coef, intercept=intercept, history=history, n_iter=len(history))


def sigmoid(scores: FloatArray) -> FloatArray:
    """Public stable sigmoid used by the estimator."""
    return _sigmoid(np.asarray(scores, dtype=float))
