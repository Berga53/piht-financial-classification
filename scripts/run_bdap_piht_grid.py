"""Run PIHT experiments. By default, runs every experiment in configs/experiments_10000.json.

Pass --features to run a single feature set over a custom period/depth/horizon grid instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from piht_classification.data import (
    flatten_panel,
    prepare_bankit_dataset,
    save_dataset,
    source_fingerprint,
)
from piht_classification.evaluation import PROTOCOL, run_experiment

MANIFEST = Path(__file__).resolve().parents[1] / "configs/experiments_10000.json"
# Defaults for a custom grid; they only apply together with --features.
CUSTOM_GRID_DEFAULTS = {
    "periods": [1, 2],
    "input_depths": [1],
    "target_depths": [1, 2, 3],
    "k": list(range(10, 101, 10)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bankit-root", type=Path, default=os.environ.get("BANKIT_ROOT"),
        required="BANKIT_ROOT" not in os.environ,
        help="checkout of municipal-financial-distress (default: $BANKIT_ROOT)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--features",
        help="run this feature set over a custom grid; omit to run all experiments "
        "in configs/experiments_10000.json",
    )
    parser.add_argument("--periods", type=int, nargs="+", choices=[1, 2])
    parser.add_argument("--input-depths", type=int, nargs="+")
    parser.add_argument("--target-depths", type=int, nargs="+")
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy", choices=("group", "random"), default="random",
        help="random row split as in the Bankitalia notebook (default), or municipality groups",
    )
    parser.add_argument("--batch-size-initial", type=int, default=256)
    parser.add_argument(
        "--batch-sampling", choices=("stratified", "uniform"), default="stratified"
    )
    parser.add_argument("--min-positive-fraction", type=float, default=0.1)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        metavar="ITERATIONS",
        help="print a progress update this often during each PIHT fit (default: 1000)",
    )
    parser.add_argument("--quiet", action="store_true", help="disable live fit progress")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name, default in CUSTOM_GRID_DEFAULTS.items():
        if args.features is None and getattr(args, name) is not None:
            parser.error(f"--{name.replace('_', '-')} requires --features")
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


def _runs(args: argparse.Namespace) -> list[dict]:
    """The experiments to run: the fixed manifest, or the custom grid for --features."""
    if args.features is not None:
        return [
            {"features": args.features, "period": period, "input_depth": input_depth,
             "target_depth": target_depth, "k_values": sorted(set(args.k)), "shape": None}
            for period in args.periods
            for target_depth in args.target_depths
            for input_depth in args.input_depths
        ]
    experiments = json.loads(MANIFEST.read_text())["experiments"]
    fingerprint = source_fingerprint(args.bankit_root.expanduser().resolve())
    if any(e["dataset"]["source_sha256"] != fingerprint for e in experiments):
        raise RuntimeError(
            "Source data differ from the data configs/experiments_10000.json was generated with "
            "(changed or missing files, e.g. the confidential data/Anticipazioni/ folder)"
        )
    return [
        {**{key: e["dataset"][key] for key in ("features", "period", "input_depth", "target_depth")},
         "k_values": e["k_values"], "shape": e["shape"]}
        for e in experiments
    ]


def _without_root(metadata: dict | None) -> dict | None:
    """Drop the machine-specific checkout path; the source hash identifies the data."""
    return None if metadata is None else {k: v for k, v in metadata.items() if k != "bankit_root"}


def _result_is_complete(
    path: Path, args: argparse.Namespace, metadata: dict, shape: list, k_values: list[int]
) -> bool:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("protocol") == PROTOCOL
        and result.get("seed") == args.seed
        and _without_root(result.get("dataset")) == _without_root(metadata)
        and result.get("shape") == shape
        and result.get("iterations") == args.iterations
        and result.get("repeats") == args.repeats
        and result.get("split_strategy") == args.split_strategy
        and result.get("k_values") == k_values
        and result.get("include_baselines") is False
        and result.get("batch_size_initial") == args.batch_size_initial
        and result.get("batch_sampling") == args.batch_sampling
        and result.get("min_positive_fraction") == args.min_positive_fraction
        and len(result.get("repetitions", [])) == args.repeats
        and all(
            all(
                set(candidate) >= {
                    "training_average_precision", "training_f1_at_threshold",
                    "test_metrics", "support_size", "selected_features",
                }
                for candidate in repetition.get("k_selection", [])
            )
            for repetition in result.get("repetitions", [])
        )
    )


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    runs = _runs(args)
    grid_started = perf_counter()

    for position, run in enumerate(runs, start=1):
        features, k_values = run["features"], run["k_values"]
        period, input_depth, target_depth = run["period"], run["input_depth"], run["target_depth"]
        stem = f"{features.replace('-', '_')}_p{period}_i{input_depth}_h{target_depth}"
        dataset_path = args.data_dir / f"{stem}.npz"
        output_path = args.results_dir / f"{stem}_piht.json"
        prefix = f"[{position}/{len(runs)}] {stem}"
        # Rebuild from source: an old NPZ must never silently restore a filtered cohort.
        configuration_started = perf_counter()
        print(f"{prefix}: preparing current national dataset", flush=True)
        dataset = prepare_bankit_dataset(
            args.bankit_root, period=period, input_depth=input_depth,
            target_depth=target_depth, features=features,
            exclude_autonomous_regions=False,
        )
        X, feature_names = flatten_panel(dataset)
        if run["shape"] is not None and list(X.shape) != run["shape"]:
            raise RuntimeError(f"{stem}: data shape {list(X.shape)} differs from configured {run['shape']}")
        if output_path.exists() and not args.overwrite:
            if _result_is_complete(output_path, args, dataset.metadata, list(X.shape), k_values):
                print(f"{prefix}: already complete; skipping", flush=True)
                continue
            raise FileExistsError(
                f"{output_path} belongs to a different protocol; use a fresh results directory"
            )
        save_dataset(dataset, dataset_path)
        print(f"{prefix}: fitting shape={X.shape}", flush=True)
        result = run_experiment(
            X,
            dataset.y,
            dataset.groups,
            feature_names=feature_names,
            k_values=k_values,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
            split_strategy=args.split_strategy,
            batch_size_initial=args.batch_size_initial,
            batch_sampling=args.batch_sampling,
            min_positive_fraction=args.min_positive_fraction,
            include_baselines=False,
            verbose=not args.quiet,
            progress_every=args.progress_every,
        )
        result["dataset"] = dataset.metadata
        result["shape"] = list(X.shape)
        result["grid_configuration_seconds"] = perf_counter() - configuration_started
        temporary_path = output_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary_path.replace(output_path)
        print(
            f"{prefix}: saved {output_path} "
            f"in {result['grid_configuration_seconds']:.1f}s",
            flush=True,
        )

    print(f"Grid finished in {perf_counter() - grid_started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
