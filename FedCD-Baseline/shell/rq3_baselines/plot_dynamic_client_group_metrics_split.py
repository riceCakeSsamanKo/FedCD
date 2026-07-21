#!/usr/bin/env python3
"""Create separate FL and PFL dynamic-client metric plots."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import plot_dynamic_client_group_metrics as base


BASELINE_GROUPS = {
    "fl": {
        "label": "FL",
        "methods": {"FedAvg", "FedProx", "FedCross", "FedDST"},
    },
    "pfl": {
        "label": "PFL",
        "methods": {
            "FedBN",
            "FedALA",
            "FedAS",
            "pFedMe",
            "cwFedAvg",
            "FedCP",
            "DualFed",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("batch_runs/rq3_dynamic_parallel"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_runs/rq3_dynamic_parallel/plots_seed1/by_family"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = base.successful_runs(args.batch_root, args.seed)
    if not runs:
        raise SystemExit(
            f"No successful seed={args.seed} runs found under {args.batch_root}"
        )

    all_rows = defaultdict(dict)
    for run in runs:
        all_rows[run["dataset"]][run["method"]] = base.parse_wrapper_log(
            Path(run["wrapper_log"])
        )

    classified = set().union(
        *(group["methods"] for group in BASELINE_GROUPS.values())
    )
    discovered = {
        method for method_rows in all_rows.values() for method in method_rows
    }
    unknown = discovered - classified
    if unknown:
        raise ValueError(f"Unclassified methods: {', '.join(sorted(unknown))}")

    for family, group in BASELINE_GROUPS.items():
        family_dir = args.output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for dataset, method_rows in sorted(all_rows.items()):
            selected_rows = {
                method: rows
                for method, rows in method_rows.items()
                if method in group["methods"]
            }
            if not selected_rows:
                continue
            for metric, original_title in base.METRICS.items():
                base.METRICS[metric] = (
                    f"{group['label']} baselines — {original_title}"
                )
                base.plot_metric(
                    dataset,
                    selected_rows,
                    metric,
                    family_dir,
                    args.formats,
                    args.dpi,
                )
                base.METRICS[metric] = original_title

    print(
        "FL methods: "
        + ", ".join(sorted(BASELINE_GROUPS["fl"]["methods"]))
    )
    print(
        "PFL methods: "
        + ", ".join(sorted(BASELINE_GROUPS["pfl"]["methods"]))
    )
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
