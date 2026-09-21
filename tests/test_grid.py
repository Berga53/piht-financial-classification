from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from piht_classification.cli import _parser


spec = importlib.util.spec_from_file_location(
    "run_bdap_piht_grid", Path(__file__).resolve().parents[1] / "scripts/run_bdap_piht_grid.py"
)
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)


class GridTests(unittest.TestCase):
    def test_runners_default_to_random_and_allow_group(self):
        for options, expected in [([], "random"), (["--split-strategy", "group"], "group")]:
            with self.subTest(strategy=expected):
                args = _parser().parse_args(
                    ["run", "--dataset", "data.npz", "--output", "result.json", "--k", "5"]
                    + options
                )
                self.assertEqual(args.split_strategy, expected)
                self.assertEqual(args.seed, 42)
                self.assertEqual(args.repeats, 1)
                with patch("sys.argv", ["grid", "--bankit-root", "."] + options):
                    self.assertEqual(grid.parse_args().split_strategy, expected)

    def test_grid_does_not_reuse_results_from_another_split(self):
        with patch("sys.argv", ["grid", "--bankit-root", ".", "--features", "bdap"]):
            args = grid.parse_args()
        result = {
            key: getattr(args, key)
            for key in (
                "iterations", "repeats", "seed", "batch_size_initial", "batch_sampling",
                "min_positive_fraction",
            )
        }
        metadata = {"municipality_count": 100}
        k_values = sorted(set(args.k))
        shape = [100, 5]
        result.update(
            protocol=grid.PROTOCOL, dataset=metadata, shape=shape,
            k_values=k_values, include_baselines=False,
            repetitions=[{}] * args.repeats,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for strategy, expected in [(None, False), ("group", False), ("random", True)]:
                with self.subTest(strategy=strategy):
                    if strategy is not None:
                        result["split_strategy"] = strategy
                    path.write_text(json.dumps(result), encoding="utf-8")
                    self.assertEqual(grid._result_is_complete(path, args, metadata, shape, k_values), expected)

    def test_result_reuse_ignores_checkout_location_but_not_source_hash(self):
        with patch("sys.argv", ["grid", "--bankit-root", ".", "--features", "bdap"]):
            args = grid.parse_args()
        metadata = {"municipality_count": 100, "source_sha256": "abc", "bankit_root": "/here"}
        result = {key: getattr(args, key) for key in (
            "iterations", "repeats", "seed", "batch_size_initial", "batch_sampling",
            "min_positive_fraction", "split_strategy")}
        result.update(protocol=grid.PROTOCOL, shape=[100, 5], k_values=[5], include_baselines=False,
                      repetitions=[{}] * args.repeats)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for stored, expected in [
                ({**metadata, "bankit_root": "/elsewhere"}, True),
                ({**metadata, "source_sha256": "changed"}, False),
            ]:
                path.write_text(json.dumps({**result, "dataset": stored}), encoding="utf-8")
                self.assertEqual(grid._result_is_complete(path, args, metadata, [100, 5], [5]), expected)

    def test_default_runs_the_manifest_and_custom_options_need_features(self):
        manifest = json.loads(grid.MANIFEST.read_text())
        text = grid.MANIFEST.read_text()
        self.assertNotIn("/Users/", text)
        self.assertEqual(len(manifest["experiments"]), 27)
        self.assertEqual(manifest["settings"]["iterations"], 10000)
        with patch("sys.argv", ["grid", "--bankit-root", "."]):
            args = grid.parse_args()
        self.assertIsNone(args.features)
        self.assertEqual((args.iterations, args.seed, args.repeats, args.split_strategy),
                         (10000, 42, 1, "random"))
        with patch("sys.argv", ["grid", "--bankit-root", ".", "--k", "5"]), \
                self.assertRaises(SystemExit):
            grid.parse_args()

    def test_custom_grid_uses_features_and_sorted_k(self):
        with patch("sys.argv", ["grid", "--bankit-root", ".", "--features", "bdap",
                                "--periods", "1", "--target-depths", "1", "2", "--k", "10", "5"]):
            runs = grid._runs(grid.parse_args())
        self.assertEqual([(r["period"], r["target_depth"], r["k_values"]) for r in runs],
                         [(1, 1, [5, 10]), (1, 2, [5, 10])])


if __name__ == "__main__":
    unittest.main()
