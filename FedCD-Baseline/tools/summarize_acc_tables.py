#!/usr/bin/env python3
"""Summarize completed FedCD acc.csv runs into paper-style tables.

The script scans logs/<dataset>/<method>/<model>/<partition>/NC_<n>/.../acc.csv,
keeps only runs that reached the required round, takes the best accuracy in
each completed run, then reports mean \\pm std across runs.

실행 명령어

cd /home/mulsoap0504/FedCD/FedCD-Baseline
python3 tools/summarize_acc_tables.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Iterable


DEFAULT_DATASETS = ("cifar10", "Cifar100", "FashionMNIST")
DEFAULT_METHODS = (
    "FedAS",
    "FedAvg",
    "FedProx",
    "Ditto",
    "FedBN",
    "FedALA",
    "FedCross",
    "cwFedAvg",
)
DEFAULT_CLIENTS = (20, 50)
PARTITION_ORDER = ("PAT", "DIR(0.1)", "DIR(0.5)", "DIR(1.0)")


@dataclass(frozen=True)
class RunKey:
    dataset: str
    method: str
    client: int
    partition: str


@dataclass
class RunSummary:
    key: RunKey
    path: Path
    local_best: float | None
    global_best: float | None
    communication_mb: float | None


def parse_csv_list(value: str, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def stddev(values: list[float], mode: str) -> float:
    if len(values) <= 1:
        return 0.0
    if mode == "population":
        return pstdev(values)
    return stdev(values)


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


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items() if key is not None}


def parse_run_key(path: Path, logs_root: Path) -> RunKey | None:
    try:
        parts = path.relative_to(logs_root).parts
    except ValueError:
        return None

    if len(parts) < 7:
        return None

    dataset, method = parts[0], parts[1]
    split = parts[3].lower()
    if split == "pat":
        partition = "PAT"
        nc_index = 4
    elif split == "dir":
        if len(parts) < 8:
            return None
        partition = f"DIR({parts[4]})"
        nc_index = 5
    else:
        return None

    nc_text = parts[nc_index]
    if not nc_text.startswith("NC_"):
        return None

    try:
        client = int(nc_text.removeprefix("NC_"))
    except ValueError:
        return None

    return RunKey(dataset=dataset, method=method, client=client, partition=partition)


def choose_metric_columns(
    fieldnames: Iterable[str],
    personalized_columns: str,
) -> tuple[str | None, str | None]:
    fields = {field.strip() for field in fieldnames}
    if {"local_test_acc", "global_test_acc"}.issubset(fields):
        return "local_test_acc", "global_test_acc"

    personalized = ("personalized_local_test_acc", "personalized_global_test_acc")
    global_model = ("global_local_test_acc", "global_global_test_acc")

    if personalized_columns == "personalized" and set(personalized).issubset(fields):
        return personalized
    if personalized_columns == "global" and set(global_model).issubset(fields):
        return global_model
    if set(personalized).issubset(fields):
        return personalized
    if set(global_model).issubset(fields):
        return global_model
    return None, None


def completed_rows(
    rows: list[dict[str, str]],
    required_round: int,
    include_incomplete: bool,
    strict_round_count: bool,
) -> tuple[list[dict[str, str]], str | None]:
    numbered_rows: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        round_number = to_float(row.get("round"))
        if round_number is None:
            continue
        numbered_rows.append((int(round_number), row))

    if not numbered_rows:
        return [], "no_rounds"

    rounds = [round_number for round_number, _ in numbered_rows]
    unique_rounds = set(rounds)
    reached_required = required_round in unique_rounds or max(rounds) >= required_round
    if not include_incomplete and not reached_required:
        return [], "incomplete"

    if strict_round_count and not include_incomplete:
        expected = set(range(1, required_round + 1))
        if unique_rounds != expected:
            return [], "not_exactly_required_rounds"

    usable = [row for round_number, row in numbered_rows if round_number <= required_round]
    return usable, None


def best_value(rows: list[dict[str, str]], column: str | None) -> float | None:
    if column is None:
        return None
    values = [to_float(row.get(column)) for row in rows]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


def communication_value(rows: list[dict[str, str]], mode: str) -> float | None:
    round_value_pairs = []
    for row in rows:
        round_number = to_float(row.get("round"))
        total_mb = to_float(row.get("total_mb"))
        if round_number is None or total_mb is None:
            continue
        round_value_pairs.append((int(round_number), total_mb))

    numeric_values = [value for _, value in round_value_pairs]
    if not numeric_values:
        return None
    if mode == "sum":
        return sum(numeric_values)
    if mode == "mean-round":
        return mean(numeric_values)
    if mode == "mean-active-round":
        active_values = [value for round_number, value in round_value_pairs if round_number >= 2]
        if not active_values:
            return None
        return mean(active_values)
    if mode == "last-round":
        return numeric_values[-1]
    raise ValueError(f"Unsupported communication mode: {mode}")


def read_run_summary(
    path: Path,
    logs_root: Path,
    args: argparse.Namespace,
) -> tuple[RunSummary | None, str | None]:
    key = parse_run_key(path, logs_root)
    if key is None:
        return None, "unrecognized_path"

    if key.dataset not in args.datasets:
        return None, "dataset_filtered"
    if key.method not in args.methods:
        return None, "method_filtered"
    if key.client not in args.clients:
        return None, "client_filtered"
    if key.partition not in PARTITION_ORDER:
        return None, "partition_filtered"

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return None, "empty_csv"
            fieldnames = [field.strip() for field in reader.fieldnames]
            rows = [normalize_row(row) for row in reader]
    except OSError:
        return None, "read_error"

    rows, skip_reason = completed_rows(
        rows,
        required_round=args.required_round,
        include_incomplete=args.include_incomplete,
        strict_round_count=args.strict_round_count,
    )
    if skip_reason is not None:
        return None, skip_reason

    local_column, global_column = choose_metric_columns(
        fieldnames,
        personalized_columns=args.personalized_columns,
    )
    local_best = best_value(rows, local_column)
    global_best = best_value(rows, global_column)
    communication_mb = communication_value(rows, args.communication_mode)

    if local_best is None and global_best is None:
        return None, "missing_accuracy_columns"

    return RunSummary(
        key=key,
        path=path,
        local_best=local_best,
        global_best=global_best,
        communication_mb=communication_mb,
    ), None


def scaled(value: float, scale: str) -> float:
    if scale == "percent":
        return value * 100.0
    return value


def formatted_mean_pm(
    values: list[float],
    scale: str,
    std_mode: str,
    decimals: int,
) -> str:
    if not values:
        return "-"
    scaled_values = [scaled(value, scale) for value in values]
    avg = mean(scaled_values)
    spread = stddev(scaled_values, std_mode)
    return f"{avg:.{decimals}f} \\pm {spread:.{decimals}f}"


def write_summary_table(
    output_path: Path,
    grouped: dict[RunKey, list[RunSummary]],
    args: argparse.Namespace,
    delimiter: str,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(
            [
                "dataset",
                "client",
                "metric",
                "메서드 \\ 정확도",
                "PAT",
                "DIR(0.1)",
                "DIR(0.5)",
                "DIR(1.0)",
                "통신 비용 (MB)",
            ]
        )

        for dataset in args.datasets:
            for client in args.clients:
                for metric_name, attr_name in (
                    ("Local Test ACC", "local_best"),
                    ("Global Test ACC", "global_best"),
                ):
                    for method in args.methods:
                        cells: list[str] = []
                        for partition in PARTITION_ORDER:
                            key = RunKey(dataset, method, client, partition)
                            values = [
                                value
                                for run in grouped.get(key, [])
                                if (value := getattr(run, attr_name)) is not None
                            ]
                            cells.append(
                                formatted_mean_pm(
                                    values,
                                    args.scale,
                                    args.std,
                                    args.decimals,
                                )
                            )

                        cost_values = [
                            value
                            for partition in PARTITION_ORDER
                            for run in grouped.get(
                                RunKey(dataset, method, client, partition),
                                [],
                            )
                            if (value := run.communication_mb) is not None
                        ]
                        cost_cell = formatted_mean_pm(
                            cost_values,
                            "raw",
                            args.std,
                            args.cost_decimals,
                        )
                        writer.writerow(
                            [
                                dataset,
                                client,
                                metric_name,
                                method,
                                *cells,
                                cost_cell,
                            ]
                        )


def write_long_csv(
    output_path: Path,
    grouped: dict[RunKey, list[RunSummary]],
    args: argparse.Namespace,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "client",
                "metric",
                "method",
                "partition",
                "n",
                "mean",
                "std",
                "formatted",
            ]
        )
        for dataset in args.datasets:
            for client in args.clients:
                for method in args.methods:
                    for partition in PARTITION_ORDER:
                        for metric_name, attr_name in (
                            ("local_test_acc", "local_best"),
                            ("global_test_acc", "global_best"),
                            ("communication_mb", "communication_mb"),
                        ):
                            key = RunKey(dataset, method, client, partition)
                            values = [
                                value
                                for run in grouped.get(key, [])
                                if (value := getattr(run, attr_name)) is not None
                            ]
                            if metric_name == "communication_mb":
                                output_values = values
                                decimals = args.cost_decimals
                                scale = "raw"
                            else:
                                output_values = [scaled(value, args.scale) for value in values]
                                decimals = args.decimals
                                scale = args.scale

                            if output_values:
                                avg = mean(output_values)
                                spread = stddev(output_values, args.std)
                                formatted = f"{avg:.{decimals}f} \\pm {spread:.{decimals}f}"
                            else:
                                avg = ""
                                spread = ""
                                formatted = "-"
                            writer.writerow(
                                [
                                    dataset,
                                    client,
                                    metric_name,
                                    method,
                                    partition,
                                    len(output_values),
                                    avg,
                                    spread,
                                    formatted,
                                ]
                            )


def write_run_details(output_path: Path, runs: list[RunSummary], args: argparse.Namespace) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "method",
                "client",
                "partition",
                "local_best",
                "global_best",
                "communication_mb",
                "acc_csv",
            ]
        )
        for run in sorted(
            runs,
            key=lambda item: (
                item.key.dataset,
                item.key.client,
                item.key.method,
                item.key.partition,
                str(item.path),
            ),
        ):
            writer.writerow(
                [
                    run.key.dataset,
                    run.key.method,
                    run.key.client,
                    run.key.partition,
                    "" if run.local_best is None else scaled(run.local_best, args.scale),
                    "" if run.global_best is None else scaled(run.global_best, args.scale),
                    "" if run.communication_mb is None else run.communication_mb,
                    run.path,
                ]
            )


def write_skip_report(output_path: Path, skipped: Counter[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reason", "count"])
        for reason, count in skipped.most_common():
            writer.writerow([reason, count])


def write_skip_details(output_path: Path, skipped_details: list[tuple[str, Path]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["reason", "acc_csv"])
        for reason, path in sorted(skipped_details, key=lambda item: (item[0], str(item[1]))):
            writer.writerow([reason, path])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-style mean +- std accuracy tables from completed acc.csv logs."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root directory containing dataset/method log folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/acc_summary_complete101"),
        help="Directory where summary table/csv files will be written.",
    )
    parser.add_argument(
        "--datasets",
        type=lambda value: parse_csv_list(value),
        default=DEFAULT_DATASETS,
        help="Comma-separated dataset directories to include.",
    )
    parser.add_argument(
        "--methods",
        type=lambda value: parse_csv_list(value),
        default=DEFAULT_METHODS,
        help="Comma-separated method directories to include, in table order.",
    )
    parser.add_argument(
        "--clients",
        type=lambda value: parse_csv_list(value, int),
        default=DEFAULT_CLIENTS,
        help="Comma-separated client counts to include, in table order.",
    )
    parser.add_argument(
        "--required-round",
        type=int,
        default=101,
        help="Only include acc.csv runs that reached this round.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include acc.csv files even if they did not reach --required-round.",
    )
    parser.add_argument(
        "--strict-round-count",
        action="store_true",
        help="Require exactly rounds 1..required-round with no extra round IDs.",
    )
    parser.add_argument(
        "--personalized-columns",
        choices=("personalized", "global"),
        default="personalized",
        help=(
            "For Ditto/pFedMe-style acc.csv files, choose personalized_* or "
            "global_* accuracy columns."
        ),
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
        help="How to compute the +- value across completed runs.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for accuracy cells.",
    )
    parser.add_argument(
        "--cost-decimals",
        type=int,
        default=2,
        help="Decimal places for communication cost cells.",
    )
    parser.add_argument(
        "--communication-mode",
        choices=("sum", "mean-round", "mean-active-round", "last-round"),
        default="mean-active-round",
        help=(
            "How to summarize total_mb inside each completed acc.csv. "
            "mean-active-round averages communication over rounds 2..required_round."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logs_root = args.logs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunSummary] = []
    grouped: dict[RunKey, list[RunSummary]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    skipped_details: list[tuple[str, Path]] = []

    for path in sorted(logs_root.rglob("acc.csv")):
        summary, skip_reason = read_run_summary(path, logs_root, args)
        if summary is None:
            reason = skip_reason or "unknown"
            skipped[reason] += 1
            skipped_details.append((reason, path))
            continue
        runs.append(summary)
        grouped[summary.key].append(summary)

    write_summary_table(output_dir / "summary_tables.tsv", grouped, args, delimiter="\t")
    write_summary_table(output_dir / "summary_tables.csv", grouped, args, delimiter=",")
    write_long_csv(output_dir / "summary_long.csv", grouped, args)
    write_run_details(output_dir / "run_details.csv", runs, args)
    write_skip_report(output_dir / "skip_report.csv", skipped)
    write_skip_details(output_dir / "skip_details.csv", skipped_details)

    print(f"Included completed runs: {len(runs)}")
    print(f"Skipped files: {sum(skipped.values())}")
    for reason, count in skipped.most_common():
        print(f"  {reason}: {count}")
    print(f"Wrote: {output_dir / 'summary_tables.tsv'}")
    print(f"Wrote: {output_dir / 'summary_tables.csv'}")
    print(f"Wrote: {output_dir / 'summary_long.csv'}")
    print(f"Wrote: {output_dir / 'run_details.csv'}")
    print(f"Wrote: {output_dir / 'skip_report.csv'}")
    print(f"Wrote: {output_dir / 'skip_details.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
