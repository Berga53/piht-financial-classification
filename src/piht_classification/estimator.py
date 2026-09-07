"""A scikit-learn compatible sparse PIHT binary classifier."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .optimizer import PIHTConfig, fit_piht_logistic, sigmoid


class SparsePIHTLogisticClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        k: int,
        *,
        max_iter: int = 1000,
        batch_size_initial: int = 64,
        batch_size_max: int | None = None,
        l2: float = 0.0,
        eta1: float = 1e-4,
        eta2: float = 1e-4,
        delta0: float = 1.0,
        delta_max: float = 10.0,
        gamma: float = 2.0,
        min_delta: float = 1e-12,
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int | None = None,
    ) -> None:
        self.k = k
        self.max_iter = max_iter
        self.batch_size_initial = batch_size_initial
        self.batch_size_max = batch_size_max
        self.l2 = l2
        self.eta1 = eta1
        self.eta2 = eta2
        self.delta0 = delta0
        self.delta_max = delta_max
        self.gamma = gamma
        self.min_delta = min_delta
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, dtype=float, ensure_all_finite=True)
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(f"SparsePIHTLogisticClassifier requires labels 0 and 1; got {classes}")

        weights = np.ones(X.shape[0], dtype=float)
        if self.class_weight is not None:
            weights *= compute_sample_weight(self.class_weight, y)
        if sample_weight is not None:
            user_weights = np.asarray(sample_weight, dtype=float)
            if user_weights.shape != weights.shape:
                raise ValueError("sample_weight has the wrong shape")
            weights *= user_weights
        weights /= weights.mean()

        config = PIHTConfig(
            k=self.k,
            max_iter=self.max_iter,
            batch_size_initial=self.batch_size_initial,
            batch_size_max=self.batch_size_max,
            l2=self.l2,
            eta1=self.eta1,
            eta2=self.eta2,
            delta0=self.delta0,
            delta_max=self.delta_max,
            gamma=self.gamma,
            min_delta=self.min_delta,
            random_state=self.random_state,
        )
        result = fit_piht_logistic(X, y.astype(float), weights, config)
        self.classes_ = classes
        self.coef_ = result.coef.reshape(1, -1)
        self.intercept_ = np.array([result.intercept])
        self.history_ = result.history
        self.n_iter_ = np.array([result.n_iter])
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        check_is_fitted(self, ("coef_", "intercept_"))
        X = check_array(X, dtype=float, ensure_all_finite=True)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = X @ self.coef_[0] + self.intercept_[0]
        if not np.isfinite(scores).all():
            raise FloatingPointError("classifier produced non-finite scores")
        return scores

    def predict_proba(self, X):
        positive = sigmoid(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def support_(self):
        check_is_fitted(self, "coef_")
        return np.flatnonzero(self.coef_[0])
