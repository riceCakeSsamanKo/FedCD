#!/usr/bin/env python3
"""Summarize 3 completed splitgp rho baseline runs as best_acc mean ± std.

Usage:
  cd /ds1home/aislab/Min/FedCD/FedCD-Baseline
  python3 tools/summarize_splitgp_3run_best_acc.py
  python3 tools/summarize_splitgp_3run_best_acc.py --format csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev, stdev


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

BASELINE_TABLE_METHODS = (
    ("FedAvg", "FedAvg"),
    ("FedAS", "FedAS"),
    ("FedProx", "FedProx"),
    ("FedBN", "FedBN"),
    ("FedALA", "FedALA"),
    ("FedCross", "FedCross"),
    ("pFedMe", "pFedME"),
    ("cwFedAvg", "cwFedAVG"),
)
BASELINE_TABLE_DATASETS = (
    ("cifar10", "CIFAR10"),
    ("FashionMNIST", "FMNIST"),
)
BASELINE_TABLE_RHOS = ("0", "0.2", "0.4", "0.6", "0.8")
DEFAULT_BASELINE_OUTPUT = Path(__file__).resolve().parents[2] / "baseline_result.csv"

METHOD_ALIASES = {
    "pFedME": "pFedMe",
    "pfedme": "pFedMe",
    "cwFedAVG": "cwFedAvg",
    "cwfedavg": "cwFedAvg",
}

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
class RunBest:
    path: Path
    metric: str
    best_acc: float
    best_round: int
    row_count: int
    max_round: int


@dataclass(frozen=True)
class SettingSummary:
    dataset: str
    method: str
    rho: str
    metric: str
    completed_runs: int
    used_runs: int
    mean_value: float
    std_value: float
    formatted: str
    sources: tuple[Path, ...]


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def normalize_dataset(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def normalize_method(name: str) -> str:
    return METHOD_ALIASES.get(name, name)


def rho_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def extract_rho(path: Path) -> str:
    for part in path.parts:
        if part.startswith("splitgp_rho"):
            return part.removeprefix("splitgp_rho")
    return ""


def display_rho(value: str) -> str:
    number = to_float(value)
    if number is None:
        return value
    if number == 0:
        return "0"
    return f"{number:g}"


def metric_for_method(method: str) -> str:
    if method.lower() == "pfedme":
        return "personalized_local_test_acc"
    return "local_test_acc"


def read_completed_best(path: Path, metric: str, required_round: int) -> RunBest | None:
    rows: list[dict[str, str]] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None
            fields = {field.strip() for field in reader.fieldnames}
            if "round" not in fields or metric not in fields:
                return None
            rows = [
                {
                    key.strip(): "" if value is None else value.strip()
                    for key, value in row.items()
                    if key is not None
                }
                for row in reader
            ]
    except OSError:
        return None

    numbered_rows: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        round_value = to_float(row.get("round"))
        if round_value is None:
            continue
        numbered_rows.append((int(round_value), row))

    if not numbered_rows:
        return None

    row_count = len(numbered_rows)
    max_round = max(round_number for round_number, _ in numbered_rows)
    if row_count < required_round or max_round < required_round:
        return None

    best_acc: float | None = None
    best_round: int | None = None
    for round_number, row in numbered_rows:
        if round_number > required_round:
            continue
        value = to_float(row.get(metric))
        if value is None:
            continue
        if best_acc is None or value > best_acc:
            best_acc = value
            best_round = round_number

    if best_acc is None or best_round is None:
        return None

    return RunBest(
        path=path,
        metric=metric,
        best_acc=best_acc,
        best_round=best_round,
        row_count=row_count,
        max_round=max_round,
    )


def stddev(values: list[float], mode: str) -> float:
    if len(values) <= 1:
        return 0.0
    if mode == "population":
        return pstdev(values)
    return stdev(values)


def scale_value(value: float, scale: str) -> float:
    if scale == "percent":
        return value * 100.0
    return value


def format_mean_pm(values: list[float], scale: str, std_mode: str, decimals: int) -> tuple[float, float, str]:
    scaled_values = [scale_value(value, scale) for value in values]
    mean_value = mean(scaled_values)
    std_value = stddev(scaled_values, std_mode)
    return mean_value, std_value, f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"


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


def select_runs(runs: list[RunBest], target_runs: int, selection: str) -> list[RunBest]:
    if selection == "latest":
        return sorted(runs, key=lambda run: str(run.path))[-target_runs:]
    if selection == "earliest":
        return sorted(runs, key=lambda run: str(run.path))[:target_runs]
    if selection == "best":
        return sorted(runs, key=lambda run: run.best_acc, reverse=True)[:target_runs]
    if selection == "all":
        return sorted(runs, key=lambda run: str(run.path))
    raise ValueError(f"Unsupported run selection: {selection}")


def build_summaries(args: argparse.Namespace) -> list[SettingSummary]:
    logs_root = args.logs_root
    datasets = tuple(normalize_dataset(name) for name in parse_csv_list(args.datasets))
    methods = tuple(normalize_method(name) for name in parse_csv_list(args.methods))

    buckets: dict[tuple[str, str, str], list[RunBest]] = defaultdict(list)
    for dataset in datasets:
        for method in methods:
            metric = metric_for_method(method)
            for path in find_acc_csvs(
                logs_root=logs_root,
                dataset=dataset,
                method=method,
                model=args.model,
                clients=args.clients,
                split_glob=args.split_glob,
            ):
                run_best = read_completed_best(path, metric, args.required_round)
                if run_best is None:
                    continue
                buckets[(dataset, method, extract_rho(path))].append(run_best)

    summaries: list[SettingSummary] = []
    for (dataset, method, rho), runs in buckets.items():
        if len(runs) < args.target_runs:
            continue
        selected = select_runs(runs, args.target_runs, args.run_selection)
        values = [run.best_acc for run in selected]
        mean_value, std_value, formatted = format_mean_pm(
            values,
            scale=args.scale,
            std_mode=args.std,
            decimals=args.decimals,
        )
        summaries.append(
            SettingSummary(
                dataset=dataset,
                method=method,
                rho=rho,
                metric=selected[0].metric,
                completed_runs=len(runs),
                used_runs=len(selected),
                mean_value=mean_value,
                std_value=std_value,
                formatted=formatted,
                sources=tuple(run.path for run in selected),
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            datasets.index(item.dataset) if item.dataset in datasets else len(datasets),
            methods.index(item.method) if item.method in methods else len(methods),
            rho_sort_key(item.rho),
        ),
    )


def print_markdown(summaries: list[SettingSummary]) -> None:
    headers = ["dataset", "method", "rho", "metric", "runs", "best_acc_mean_std"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for summary in summaries:
        row = [
            summary.dataset,
            summary.method,
            summary.rho,
            summary.metric,
            f"{summary.used_runs}/{summary.completed_runs}",
            summary.formatted,
        ]
        print("| " + " | ".join(row) + " |")


def print_csv_output(summaries: list[SettingSummary]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "dataset",
            "method",
            "rho",
            "metric",
            "completed_runs",
            "used_runs",
            "mean",
            "std",
            "formatted",
            "sources",
        ]
    )
    for summary in summaries:
        writer.writerow(
            [
                summary.dataset,
                summary.method,
                summary.rho,
                summary.metric,
                summary.completed_runs,
                summary.used_runs,
                f"{summary.mean_value:.10f}",
                f"{summary.std_value:.10f}",
                summary.formatted,
                ";".join(str(path) for path in summary.sources),
            ]
        )


def write_baseline_table_csv(path: Path, summaries: list[SettingSummary], clients: int) -> None:
    lookup = {
        (summary.dataset, summary.method, display_rho(summary.rho)): summary.formatted
        for summary in summaries
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        for section_index, (dataset, dataset_label) in enumerate(BASELINE_TABLE_DATASETS):
            if section_index:
                writer.writerow([])
            writer.writerow([f"Client {clients} / {dataset_label}"])
            writer.writerow(["메서드 \\ 정확도(%)", *BASELINE_TABLE_RHOS])
            for method, display_method in BASELINE_TABLE_METHODS:
                writer.writerow(
                    [
                        display_method,
                        *[
                            lookup.get((dataset, method, rho), "")
                            for rho in BASELINE_TABLE_RHOS
                        ],
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For splitgp rho baselines, report mean ± std of per-run best accuracy "
            "for settings with enough completed runs."
        )
    )
    parser.add_argument("--logs-root", type=Path, default=Path("logs"), help="FedCD-Baseline logs root.")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names. Aliases include cifar-10 and fashionmnist.",
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated method names.")
    parser.add_argument("--model", default="VGG8", help="Model name used in GM_<model>.")
    parser.add_argument("--clients", type=int, default=50, help="Client count used in NC_<clients>.")
    parser.add_argument("--split-glob", default="splitgp_rho*", help="Partition directory glob under GM_<model>.")
    parser.add_argument(
        "--required-round",
        type=int,
        default=101,
        help="Only include acc.csv runs with at least this many round rows and max round >= this value.",
    )
    parser.add_argument(
        "--target-runs",
        type=int,
        default=3,
        help="Only summarize settings with at least this many completed runs.",
    )
    parser.add_argument(
        "--run-selection",
        choices=("latest", "earliest", "best", "all"),
        default="latest",
        help="Which completed runs to use when a setting has more than --target-runs runs.",
    )
    parser.add_argument(
        "--scale",
        choices=("raw", "percent"),
        default="raw",
        help="Use raw 0..1 accuracy values or multiply accuracies by 100.",
    )
    parser.add_argument(
        "--std",
        choices=("sample", "population"),
        default="sample",
        help="How to compute the value after ±.",
    )
    parser.add_argument("--decimals", type=int, default=4, help="Decimal places for output cells.")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown", help="Output format.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Write a paper-style pivot CSV. For the requested baseline table, use "
            f"{DEFAULT_BASELINE_OUTPUT}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = build_summaries(args)
    if args.output_csv is not None:
        write_baseline_table_csv(args.output_csv, summaries, args.clients)
    if args.format == "csv":
        print_csv_output(summaries)
    else:
        print_markdown(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
