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
        with patch("sys.argv", ["grid", "--bankit-root", "."]):
            args = grid.parse_args()
        result = {
            key: getattr(args, key)
            for key in (
                "iterations", "repeats", "seed", "batch_size_initial", "batch_sampling",
                "min_positive_fraction",
            )
        }
        metadata = {"municipality_count": 100}
        shape = [100, 5]
        result.update(
            protocol=grid.PROTOCOL, dataset=metadata, shape=shape,
            k_values=sorted(set(args.k)), include_baselines=False,
            repetitions=[{}] * args.repeats,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            for strategy, expected in [(None, False), ("group", False), ("random", True)]:
                with self.subTest(strategy=strategy):
                    if strategy is not None:
                        result["split_strategy"] = strategy
                    path.write_text(json.dumps(result), encoding="utf-8")
                    self.assertEqual(grid._result_is_complete(path, args, metadata, shape), expected)


if __name__ == "__main__":
    unittest.main()
