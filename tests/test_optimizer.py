from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from piht_classification import SparsePIHTLogisticClassifier, hard_threshold
from piht_classification.optimizer import logistic_gradient, logistic_loss


class OptimizerTests(unittest.TestCase):
    def test_hard_threshold_is_exact_and_non_mutating(self):
        values = np.array([1.0, -4.0, 2.0, 0.5])
        original = values.copy()
        result = hard_threshold(values, 2)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(result, np.array([0.0, -4.0, 2.0, 0.0]))

    def test_logistic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 4))
        y = rng.integers(0, 2, size=20).astype(float)
        weights = rng.uniform(0.5, 2.0, size=20)
        coef = rng.normal(size=4)
        intercept = 0.2
        l2 = 0.07
        gradient, intercept_gradient = logistic_gradient(coef, intercept, X, y, weights, l2)

        epsilon = 1e-6
        numerical = np.empty_like(coef)
        for index in range(len(coef)):
            shift = np.zeros_like(coef)
            shift[index] = epsilon
            numerical[index] = (
                logistic_loss(coef + shift, intercept, X, y, weights, l2)
                - logistic_loss(coef - shift, intercept, X, y, weights, l2)
            ) / (2 * epsilon)
        numerical_intercept = (
            logistic_loss(coef, intercept + epsilon, X, y, weights, l2)
            - logistic_loss(coef, intercept - epsilon, X, y, weights, l2)
        ) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, atol=1e-6)
        self.assertAlmostEqual(intercept_gradient, numerical_intercept, places=6)

    def test_classifier_respects_support_budget(self):
        X, y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=4,
            weights=[0.8, 0.2],
            class_sep=1.2,
            random_state=7,
        )
        X = StandardScaler().fit_transform(X)
        model = SparsePIHTLogisticClassifier(
            k=4,
            max_iter=300,
            batch_size_initial=32,
            random_state=7,
        ).fit(X, y)
        self.assertLessEqual(len(model.support_), 4)
        score = average_precision_score(y, model.predict_proba(X)[:, 1])
        self.assertGreater(score, 0.60)


if __name__ == "__main__":
    unittest.main()
