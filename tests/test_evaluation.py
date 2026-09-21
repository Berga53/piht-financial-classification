from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from piht_classification.evaluation import (
    PROTOCOL, _train_test_split, best_f1_threshold, binary_metrics, run_experiment,
)


class EvaluationTests(unittest.TestCase):
    def test_random_split_matches_bankit_exactly(self):
        groups = np.tile(np.arange(100), 5)
        y = (groups % 5 == 0).astype(int)
        X = np.arange(len(y)).reshape(-1, 1)
        expected_train, expected_test = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
        )
        train, test = _train_test_split(X, y, groups, "random", 42)
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)
        self.assertEqual((len(train), len(test)), (400, 100))
        self.assertFalse(set(train) & set(test))
        self.assertTrue(set(groups[train]) & set(groups[test]))

    def test_threshold_matches_bankit_grid_and_strict_comparison(self):
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.504, 0.505, 0.9])
        grid = np.arange(0, 1 + 1e-9, 0.01)
        scores = [f1_score(y, probabilities > t) for t in grid]
        threshold, score = best_f1_threshold(y, probabilities)
        self.assertEqual(threshold, grid[np.argmax(scores)])
        self.assertEqual(score, max(scores))
        # Equality to the cutoff must be classified negative, as in the notebook.
        m = binary_metrics([0, 1], np.array([0.5, 0.6]), 0.5)
        self.assertEqual(m['confusion_matrix'], [[1, 0], [0, 1]])

    def test_one_run_uses_all_training_rows_and_training_only_selection(self):
        X, y = make_classification(n_samples=150, n_features=8, n_informative=4,
                                   weights=[0.8, 0.2], random_state=4)
        train, test = train_test_split(np.arange(len(y)), test_size=.2,
                                      stratify=y, random_state=42)
        output = io.StringIO()
        with patch('piht_classification.evaluation.StandardScaler', wraps=StandardScaler) as scaler:
            with patch('piht_classification.evaluation.average_precision_score',
                       wraps=__import__('sklearn.metrics', fromlist=['average_precision_score']).average_precision_score) as ap:
                with redirect_stdout(output):
                    result = run_experiment(
                        X, y, np.arange(len(y)), feature_names=[f'x{i}' for i in range(8)],
                        k_values=[2, 4], iterations=5, include_baselines=False, verbose=True,
                    )
                # Each candidate is selected with training AP and evaluated on the test split.
                np.testing.assert_array_equal(ap.call_args_list[0].args[0], y[train])
                np.testing.assert_array_equal(ap.call_args_list[1].args[0], y[test])
                np.testing.assert_array_equal(ap.call_args_list[2].args[0], y[train])
                np.testing.assert_array_equal(ap.call_args_list[3].args[0], y[test])
            self.assertEqual(scaler.call_count, 1)
        self.assertEqual(result['protocol'], PROTOCOL)
        self.assertEqual(result['seed'], 42)
        self.assertEqual(result['repeats'], 1)
        rep = result['repetitions'][0]
        self.assertEqual(rep['seed'], 42)
        self.assertEqual(rep['sizes'], {'train': 120, 'test': 30})
        self.assertNotIn('l1_logistic', rep)
        self.assertEqual([c['k'] for c in rep['k_selection']], [2, 4])
        self.assertTrue(all('training_average_precision' in c for c in rep['k_selection']))
        self.assertTrue(all('test_metrics' in c for c in rep['k_selection']))
        self.assertTrue(all('selected_features' in c for c in rep['k_selection']))
        selected = max(rep['k_selection'], key=lambda c: c['training_average_precision'])
        self.assertEqual(rep['piht']['selected_k'], selected['k'])
        self.assertEqual(rep['piht']['average_precision'], selected['test_metrics']['average_precision'])
        self.assertIn('training_f1_at_threshold', rep['piht'])
        self.assertIn('[repetition 1/1, seed=42]', output.getvalue())
        self.assertIn('training AP=', output.getvalue())


if __name__ == '__main__':
    unittest.main()
