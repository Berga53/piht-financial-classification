"""Command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import flatten_panel, load_dataset, prepare_bankit_dataset, save_dataset
from .evaluation import run_experiment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PIHT sparse classification experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-bankit", help="build a Bankitalia panel dataset")
    prepare.add_argument("--bankit-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--period", type=int, choices=(1, 2), required=True)
    prepare.add_argument("--input-depth", type=int, choices=(4, 5, 6), required=True)
    prepare.add_argument("--target-depth", type=int, choices=(1, 2, 3), required=True)
    prepare.add_argument(
        "--features",
        choices=(
            "bdap",
            "bdap-anticipazioni",
            "bdap-indicatori-anticipazioni",
            "indicatori",
        ),
        default="bdap",
    )
    prepare.add_argument("--include-autonomous-regions", action="store_true")

    run = subparsers.add_parser("run", help="select K and evaluate PIHT")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--k", type=int, nargs="+", required=True)
    run.add_argument("--iterations", type=int, default=1000)
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--split-strategy", choices=("group", "random"), default="group")
    run.add_argument("--l2", type=float, default=0.0)
    run.add_argument("--batch-size-initial", type=int, default=64)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-bankit":
        dataset = prepare_bankit_dataset(
            args.bankit_root,
            period=args.period,
            input_depth=args.input_depth,
            target_depth=args.target_depth,
            features=args.features,
            exclude_autonomous_regions=not args.include_autonomous_regions,
        )
        save_dataset(dataset, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "X_shape": dataset.X.shape,
                    "positives": int(dataset.y.sum()),
                    "observations": len(dataset.y),
                },
                indent=2,
            )
        )
        return

    dataset = load_dataset(args.dataset)
    X, feature_names = flatten_panel(dataset)
    results = run_experiment(
        X,
        dataset.y,
        dataset.groups,
        feature_names=feature_names,
        k_values=args.k,
        iterations=args.iterations,
        repeats=args.repeats,
        split_strategy=args.split_strategy,
        l2=args.l2,
        batch_size_initial=args.batch_size_initial,
    )
    results["dataset"] = dataset.metadata
    results["shape"] = list(X.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "shape": X.shape}, indent=2))


if __name__ == "__main__":
    main()

