"""Core PIHT operations for cardinality-constrained logistic regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PIHTConfig:
    k: int
    max_iter: int = 1000
    batch_size_initial: int = 64
    batch_size_max: int | None = None
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
    """Batch schedule used by the sparsity-chapter implementation."""
    if delta <= 0:
        return batch_size_max
    exponent = delta0 / delta
    if exponent > 12:
        return batch_size_max
    return min(batch_size_max, max(1, int(batch_size_initial * (2.0**exponent))))


def fit_piht_logistic(
    X: FloatArray,
    y: FloatArray,
    sample_weight: FloatArray,
    config: PIHTConfig,
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
        gradient_indices = rng.choice(n_samples, size=batch_size, replace=False)
        grad_coef, grad_intercept = logistic_gradient(
            coef,
            intercept,
            X[gradient_indices],
            y[gradient_indices],
            sample_weight[gradient_indices],
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

        acceptance_indices = rng.choice(n_samples, size=batch_size, replace=False)
        current_loss = logistic_loss(
            coef,
            intercept,
            X[acceptance_indices],
            y[acceptance_indices],
            sample_weight[acceptance_indices],
            config.l2,
        )
        candidate_loss = logistic_loss(
            candidate_coef,
            candidate_intercept,
            X[acceptance_indices],
            y[acceptance_indices],
            sample_weight[acceptance_indices],
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

        history.append(
            {
                "iteration": iteration + 1,
                "accepted": accepted,
                "loss": candidate_loss if accepted else current_loss,
                "candidate_loss": candidate_loss,
                "gradient_norm": gradient_norm,
                "ratio": float(ratio),
                "delta": delta,
                "batch_size": batch_size,
                "support_size": int(np.count_nonzero(coef)),
            }
        )
        if delta < config.min_delta:
            break

    return PIHTResult(coef=coef, intercept=intercept, history=history, n_iter=len(history))


def sigmoid(scores: FloatArray) -> FloatArray:
    """Public stable sigmoid used by the estimator."""
    return _sigmoid(np.asarray(scores, dtype=float))
