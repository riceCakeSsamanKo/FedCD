#!/usr/bin/env python3
"""Count completed splitgp rho baseline acc.csv runs by method.

Usage:
  cd /ds1home/aislab/Min/FedCD/FedCD-Baseline
  python3 tools/count_complete_acc_runs.py
  python3 tools/count_complete_acc_runs.py --by-rho
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = ("cifar10", "FashionMNIST")
DEFAULT_METHODS = (
    "cwFedAvg",
    "FedALA",
    "FedAS",
    "FedAvg",
    "FedBN",
    "FedCross",
    "pFedMe",
    "FedProx",
)


DATASET_ALIASES = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "Cifar10": "cifar10",
    "CIFAR10": "cifar10",
    "CIFAR-10": "cifar10",
    "fashionmnist": "FashionMNIST",
    "FashionMNIST": "FashionMNIST",
    "fashion-mnist": "FashionMNIST",
    "Fashion-MNIST": "FashionMNIST",
}


@dataclass(frozen=True)
class CountSummary:
    completed: int
    scanned: int


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def normalize_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def acc_csv_is_complete(path: Path, required_round: int) -> bool:
    count = 0
    max_round: int | None = None
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "round" not in reader.fieldnames:
                return False
            for row in reader:
                text = (row.get("round") or "").strip()
                if not text.isdigit():
                    continue
                round_idx = int(text)
                count += 1
                if max_round is None or round_idx > max_round:
                    max_round = round_idx
    except OSError:
        return False

    return count >= required_round and max_round is not None and max_round >= required_round


def extract_rho(path: Path) -> str:
    for part in path.parts:
        if part.startswith("splitgp_rho"):
            return part.removeprefix("splitgp_rho")
    return ""


def find_acc_csvs(
    logs_root: Path,
    dataset: str,
    method: str,
    model: str,
    clients: int,
    split_glob: str,
) -> list[Path]:
    root = logs_root / dataset / method / f"GM_{model}"
    pattern = f"{split_glob}/NC_{clients}/date_*/time_*/acc.csv"
    return sorted(root.glob(pattern))


def count_runs(paths: list[Path], required_round: int) -> CountSummary:
    completed = sum(1 for path in paths if acc_csv_is_complete(path, required_round))
    return CountSummary(completed=completed, scanned=len(paths))


def print_markdown(
    counts: dict[tuple[str, str], CountSummary],
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    show_scanned: bool,
) -> None:
    if show_scanned:
        headers = ["method", *(f"{dataset} complete/scanned" for dataset in datasets), "total complete/scanned"]
    else:
        headers = ["method", *datasets, "total"]

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for method in methods:
        row = [method]
        total_completed = 0
        total_scanned = 0
        for dataset in datasets:
            summary = counts[(dataset, method)]
            total_completed += summary.completed
            total_scanned += summary.scanned
            if show_scanned:
                row.append(f"{summary.completed}/{summary.scanned}")
            else:
                row.append(str(summary.completed))
        if show_scanned:
            row.append(f"{total_completed}/{total_scanned}")
        else:
            row.append(str(total_completed))
        print("| " + " | ".join(row) + " |")


def print_csv(
    counts: dict[tuple[str, str], CountSummary],
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    show_scanned: bool,
) -> None:
    writer = csv.writer(sys.stdout)
    if show_scanned:
        writer.writerow(
            [
                "method",
                *[f"{dataset}_completed" for dataset in datasets],
                *[f"{dataset}_scanned" for dataset in datasets],
                "total_completed",
                "total_scanned",
            ]
        )
    else:
        writer.writerow(["method", *datasets, "total"])

    for method in methods:
        completed_values = [counts[(dataset, method)].completed for dataset in datasets]
        scanned_values = [counts[(dataset, method)].scanned for dataset in datasets]
        if show_scanned:
            writer.writerow([method, *completed_values, *scanned_values, sum(completed_values), sum(scanned_values)])
        else:
            writer.writerow([method, *completed_values, sum(completed_values)])


def print_by_rho(
    rho_counts: dict[tuple[str, str, str], CountSummary],
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    output_format: str,
    show_scanned: bool,
) -> None:
    rhos = sorted({key[2] for key in rho_counts}, key=lambda value: float(value) if value else -1.0)
    rows: list[list[str]] = []
    for dataset in datasets:
        for method in methods:
            for rho in rhos:
                summary = rho_counts.get((dataset, method, rho), CountSummary(0, 0))
                if show_scanned:
                    rows.append([dataset, method, rho, str(summary.completed), str(summary.scanned)])
                else:
                    rows.append([dataset, method, rho, str(summary.completed)])

    if output_format == "csv":
        writer = csv.writer(sys.stdout)
        writer.writerow(["dataset", "method", "rho", "completed", *([] if not show_scanned else ["scanned"])])
        writer.writerows(rows)
        return

    headers = ["dataset", "method", "rho", "completed", *([] if not show_scanned else ["scanned"])]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count acc.csv files that completed through the required round.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="FedCD-Baseline logs root.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names. Aliases include cifar-10 and fashionmnist.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method names.",
    )
    parser.add_argument("--model", default="VGG8", help="Model name used in GM_<model>.")
    parser.add_argument("--clients", type=int, default=50, help="Client count used in NC_<clients>.")
    parser.add_argument(
        "--split-glob",
        default="splitgp_rho*",
        help="Partition directory glob under GM_<model>.",
    )
    parser.add_argument(
        "--required-round",
        type=int,
        default=101,
        help="A run is complete when it has at least this many numeric round rows and reaches this round.",
    )
    parser.add_argument(
        "--show-scanned",
        action="store_true",
        help="Also show how many acc.csv files were scanned.",
    )
    parser.add_argument(
        "--by-rho",
        action="store_true",
        help="Print counts split by rho instead of only method totals.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logs_root = args.logs_root
    datasets = tuple(normalize_dataset(name) for name in parse_csv_list(args.datasets))
    methods = parse_csv_list(args.methods)

    counts: dict[tuple[str, str], CountSummary] = {}
    rho_buckets: dict[tuple[str, str, str], list[Path]] = defaultdict(list)

    for dataset in datasets:
        for method in methods:
            paths = find_acc_csvs(
                logs_root=logs_root,
                dataset=dataset,
                method=method,
                model=args.model,
                clients=args.clients,
                split_glob=args.split_glob,
            )
            counts[(dataset, method)] = count_runs(paths, args.required_round)
            for path in paths:
                rho_buckets[(dataset, method, extract_rho(path))].append(path)

    if args.by_rho:
        rho_counts = {
            key: count_runs(paths, args.required_round)
            for key, paths in rho_buckets.items()
        }
        print_by_rho(
            rho_counts=rho_counts,
            datasets=datasets,
            methods=methods,
            output_format=args.format,
            show_scanned=args.show_scanned,
        )
    elif args.format == "csv":
        print_csv(counts=counts, datasets=datasets, methods=methods, show_scanned=args.show_scanned)
    else:
        print_markdown(counts=counts, datasets=datasets, methods=methods, show_scanned=args.show_scanned)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
