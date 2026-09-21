"""Rerun the fixed experiment manifest, refresh reporting, and write a completion manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from refresh_results_notebook import refresh


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/experiments_10000.json"


def is_complete(path: Path, expected: dict) -> bool:
    try:
        result = json.loads(path.read_text())
        # The source checkout location is machine-specific; the source hash identifies the data.
        if isinstance(result.get("dataset"), dict):
            result["dataset"] = {k: v for k, v in result["dataset"].items() if k != "bankit_root"}
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        repetitions = result["repetitions"]
        return (
            [rep["seed"] for rep in repetitions] == list(range(expected["seed"], expected["seed"] + expected["repeats"]))
            and all(
                [candidate["k"] for candidate in rep["k_selection"]] == expected["k_values"]
                and all(
                    set(candidate) >= {
                        "training_average_precision", "training_f1_at_threshold",
                        "test_metrics", "support_size", "selected_features",
                    }
                    for candidate in rep["k_selection"]
                )
                and rep["piht"]["selected_k"] in expected["k_values"]
                and set(rep["sizes"]) == {"train", "test"}
                and rep["sizes"]["test"] == math.ceil(expected["shape"][0] * 0.2)
                and sum(rep["sizes"].values()) == expected["shape"][0]
                for rep in repetitions
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def command_for(experiment: dict, settings: dict, bankit_root: Path) -> list[str]:
    dataset = experiment["dataset"]
    return [
        sys.executable, str(ROOT / "scripts/run_bdap_piht_grid.py"),
        "--bankit-root", str(bankit_root),
        "--features", dataset["features"],
        "--periods", str(dataset["period"]),
        "--input-depths", str(dataset["input_depth"]),
        "--target-depths", str(dataset["target_depth"]),
        "--k", *map(str, experiment["k_values"]),
        "--iterations", str(settings["iterations"]),
        "--repeats", str(settings["repeats"]),
        "--seed", str(settings["seed"]),
        "--batch-size-initial", str(settings["batch_size_initial"]),
        "--batch-sampling", settings["batch_sampling"],
        "--min-positive-fraction", str(settings["min_positive_fraction"]),
        "--split-strategy", settings["split_strategy"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--bankit-root", type=Path, default=os.environ.get("BANKIT_ROOT"),
        help="checkout of municipal-financial-distress with its data/ folder (default: $BANKIT_ROOT)",
    )
    args = parser.parse_args()
    if args.bankit_root is None:
        parser.error("set --bankit-root or the BANKIT_ROOT environment variable")
    bankit_root = args.bankit_root.expanduser().resolve()
    os.chdir(ROOT)
    manifest = json.loads(MANIFEST.read_text())
    settings = manifest["settings"]
    experiments = manifest["experiments"]
    # Refuse a run if source files changed after configuring it.
    sys.path.insert(0, str(ROOT / "src"))
    from piht_classification.data import source_fingerprint
    current_hash = source_fingerprint(bankit_root)
    if any(e["dataset"]["source_sha256"] != current_hash for e in experiments):
        raise RuntimeError(
            "Source data differ from the data this configuration was generated with "
            "(changed or missing files, e.g. the confidential data/Anticipazioni/ folder); "
            "regenerate the experiment configuration first"
        )
    # The grid runner uses these fixed defaults; fail if the manifest changes them.
    assert settings["l2"] == 0.0 and settings["include_baselines"] is False
    assert all(not e["dataset"]["excluded_autonomous_regions"] for e in experiments)
    assert settings["repeats"] == 1 and settings["seed"] == 42
    assert all(e["dataset"]["input_depth"] == 1 for e in experiments)
    for module in ["IPython", "matplotlib", "matplotlib_inline", "PIL", "pandas", "sklearn"]:
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Missing reporting/training dependency: {module}")
    ready = ROOT / "results/workflow_10000_complete.json"
    if not args.dry_run:
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.unlink(missing_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result_paths = []
    for index, experiment in enumerate(experiments, start=1):
        path = ROOT / "results" / experiment["filename"]
        expected = {**settings, **{k: experiment[k] for k in ("dataset", "shape", "k_values")}}
        result_paths.append(path)
        if is_complete(path, expected):
            print(f"[{index}/{len(experiments)}] Already complete: {path.name}", flush=True)
            continue
        if path.exists():
            raise FileExistsError(f"{path} is incomplete or incompatible; remove that file before retrying")
        command = command_for(experiment, settings, bankit_root)
        print(f"[{index}/{len(experiments)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, check=True)
            if not is_complete(path, expected):
                raise RuntimeError(f"Result validation failed: {path}")
    if args.dry_run:
        print("Then refresh notebook tables and plots, and write the thesis-ready completion manifest.")
        return

    report_paths = refresh(ROOT)
    files = [MANIFEST, *result_paths, *report_paths,
             *sorted((ROOT / "src/piht_classification").glob("*.py")),
             ROOT / "scripts/run_bdap_piht_grid.py", Path(__file__).resolve()]
    completion = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": settings["protocol"],
        "seed": settings["seed"],
        "iterations": settings["iterations"],
        "split_strategy": settings["split_strategy"],
        "experiment_count": len(experiments),
        "repetitions_per_experiment": settings["repeats"],
        "sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    temporary = ready.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(completion, indent=2) + "\n")
    temporary.replace(ready)
    print(f"Workflow complete. Completion manifest: {ready}")


if __name__ == "__main__":
    main()
