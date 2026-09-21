from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("workflow", scripts / "run_10000_workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def test_resume_requires_new_budget_and_all_seeds(self):
        expected = {
            "iterations": 10000, "repeats": 1, "seed": 42, "split_strategy": "random",
            "shape": [100, 5], "k_values": [2, 4],
        }
        result = {**expected, "repetitions": [
            {"seed": seed, "k_selection": [
                {"k": k, "training_average_precision": .5,
                 "training_f1_at_threshold": .4, "test_metrics": {},
                 "support_size": k, "selected_features": []}
                for k in [2, 4]
            ],
             "piht": {"selected_k": 2}, "sizes": {"train": 80, "test": 20}}
            for seed in [42]
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            self.assertTrue(workflow.is_complete(path, expected))
            result["iterations"] = 5000
            path.write_text(json.dumps(result))
            self.assertFalse(workflow.is_complete(path, expected))
            result["iterations"] = 10000
            result["repetitions"].pop()
            path.write_text(json.dumps(result))
            self.assertFalse(workflow.is_complete(path, expected))

    def test_manifest_has_no_machine_specific_paths(self):
        text = workflow.MANIFEST.read_text()
        self.assertNotIn("/Users/", text)
        self.assertNotIn("bankit_root", text)

    def test_completion_ignores_checkout_location(self):
        dataset = {"period": 1, "source_sha256": "abc"}
        expected = {
            "iterations": 10000, "repeats": 1, "seed": 42, "split_strategy": "random",
            "shape": [100, 5], "k_values": [2], "dataset": dataset,
        }
        result = {**expected, "dataset": {**dataset, "bankit_root": "/somewhere/else"},
                  "repetitions": [{"seed": 42, "k_selection": [
                      {"k": 2, "training_average_precision": .5, "training_f1_at_threshold": .4,
                       "test_metrics": {}, "support_size": 2, "selected_features": []}],
                      "piht": {"selected_k": 2}, "sizes": {"train": 80, "test": 20}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            self.assertTrue(workflow.is_complete(path, expected))
            result["dataset"]["source_sha256"] = "changed"
            path.write_text(json.dumps(result))
            self.assertFalse(workflow.is_complete(path, expected))

    def test_manifest_commands_preserve_each_configuration(self):
        manifest = json.loads(workflow.MANIFEST.read_text())
        self.assertEqual(len(manifest["experiments"]), 27)
        for experiment in manifest["experiments"]:
            command = workflow.command_for(experiment, manifest["settings"], Path("/data/bankit"))
            with self.subTest(filename=experiment["filename"]):
                self.assertEqual(command[command.index("--iterations") + 1], "10000")
                self.assertEqual(command[command.index("--seed") + 1], "42")
                self.assertEqual(command[command.index("--repeats") + 1], "1")
                self.assertNotIn("--overwrite", command)
                self.assertEqual(command[command.index("--split-strategy") + 1], "random")
                self.assertEqual(
                    list(map(int, command[command.index("--k") + 1:command.index("--iterations")])),
                    experiment["k_values"],
                )
                self.assertEqual(command[command.index("--features") + 1], experiment["dataset"]["features"])


if __name__ == "__main__":
    unittest.main()
