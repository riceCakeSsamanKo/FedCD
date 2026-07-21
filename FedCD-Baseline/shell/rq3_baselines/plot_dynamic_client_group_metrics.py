#!/usr/bin/env python3
"""Plot existing/newcomer client ID and client-level OOD accuracy curves.

The script discovers successful experiments from the RQ3 parallel-launcher
status files and parses the per-round dynamic-client summaries from their
wrapper logs.  If the same method/dataset/seed was run more than once, the
latest successful run is used.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = {
    "existing_id": "Initial clients: ID accuracy",
    "existing_ood": "Initial clients: client-level OOD accuracy",
    "newcomer_id": "New clients: ID accuracy",
    "newcomer_ood": "New clients: client-level OOD accuracy",
}

VALUE_PATTERN = r"(?:N/A|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

SUMMARY_RE = re.compile(
    r"\[Dynamic Clients\]\[dynamic_test\]\[round=(?P<round>\d+)\]"
    r"\[(?P<phase>[^\]]+)\]\s+"
    rf"existing ID/OOD=(?P<existing_id>{VALUE_PATTERN})/"
    rf"(?P<existing_ood>{VALUE_PATTERN})\s*\|\s*"
    rf"newcomer ID/OOD=(?P<newcomer_id>{VALUE_PATTERN})/"
    rf"(?P<newcomer_ood>{VALUE_PATTERN})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("batch_runs/rq3_dynamic_parallel"),
        help="Directory containing launcher status.tsv files.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_runs/rq3_dynamic_parallel/plots_seed1"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "svg"],
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def successful_runs(batch_root: Path, seed: int) -> list[dict[str, str]]:
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    for status_path in sorted(batch_root.glob("date_*/time_*/status.tsv")):
        with status_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("status") != "ok" or int(row["seed"]) != seed:
                    continue
                key = (row["method"], row["dataset"], row["seed"])
                previous = latest.get(key)
                if previous is None or parse_time(row["end_utc"]) > parse_time(
                    previous["end_utc"]
                ):
                    latest[key] = row
    return sorted(latest.values(), key=lambda row: (row["dataset"], row["method"]))


def as_float(value: str) -> float | None:
    if value in {"N/A", "", "nan", "NaN"}:
        return None
    return float(value)


def parse_wrapper_log(path: Path) -> list[dict[str, float | int | str | None]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows_by_round: dict[int, dict[str, float | int | str | None]] = {}
    for match in SUMMARY_RE.finditer(text):
        round_index = int(match.group("round"))
        rows_by_round[round_index] = {
            "round": round_index,
            "phase": match.group("phase"),
            "existing_id": as_float(match.group("existing_id")),
            "existing_ood": as_float(match.group("existing_ood")),
            "newcomer_id": as_float(match.group("newcomer_id")),
            "newcomer_ood": as_float(match.group("newcomer_ood")),
        }
    if not rows_by_round:
        raise ValueError(f"No dynamic-client metrics found in {path}")
    return [rows_by_round[index] for index in sorted(rows_by_round)]


def dataset_display_name(dataset: str) -> str:
    return {"cifar10": "CIFAR-10", "fashionmnist": "FashionMNIST"}.get(
        dataset.lower(), dataset
    )


def write_merged_csv(
    output_path: Path,
    all_rows: dict[str, dict[str, list[dict[str, float | int | str | None]]]],
) -> None:
    fields = [
        "dataset",
        "method",
        "round",
        "phase",
        "existing_id",
        "existing_ood",
        "newcomer_id",
        "newcomer_ood",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in sorted(all_rows):
            for method in sorted(all_rows[dataset]):
                for row in all_rows[dataset][method]:
                    writer.writerow({"dataset": dataset, "method": method, **row})


def plot_metric(
    dataset: str,
    method_rows: dict[str, list[dict[str, float | int | str | None]]],
    metric: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7))
    colors = plt.get_cmap("tab20").colors
    line_styles = ["-", "--", "-.", ":"]

    join_rounds: list[int] = []
    for method_index, (method, rows) in enumerate(sorted(method_rows.items())):
        points = [
            (int(row["round"]), float(row[metric]) * 100.0)
            for row in rows
            if row[metric] is not None
        ]
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            label=method,
            color=colors[method_index % len(colors)],
            linestyle=line_styles[(method_index // len(colors)) % len(line_styles)],
            linewidth=2.0,
            alpha=0.95,
        )
        newcomer_rounds = [
            int(row["round"]) for row in rows if row["newcomer_id"] is not None
        ]
        if newcomer_rounds:
            join_rounds.append(min(newcomer_rounds))

    if not join_rounds:
        raise ValueError(f"Could not infer the newcomer join round for {dataset}")
    join_round = min(join_rounds)
    ax.axvline(
        join_round,
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label=f"New clients join (round {join_round})",
    )

    ax.set_title(f"{dataset_display_name(dataset)} — {METRICS[metric]}", fontsize=15)
    ax.set_xlabel("Communication round", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, max(int(row["round"]) for rows in method_rows.values() for row in rows))
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout()

    stem = f"{dataset.lower()}_{metric}_acc"
    for output_format in formats:
        fig.savefig(
            output_dir / f"{stem}.{output_format}",
            dpi=dpi if output_format == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = successful_runs(args.batch_root, args.seed)
    if not runs:
        raise SystemExit(
            f"No successful seed={args.seed} runs found under {args.batch_root}"
        )

    all_rows: dict[
        str, dict[str, list[dict[str, float | int | str | None]]]
    ] = defaultdict(dict)
    for run in runs:
        log_path = Path(run["wrapper_log"])
        rows = parse_wrapper_log(log_path)
        rounds = [int(row["round"]) for row in rows]
        if rounds != list(range(rounds[0], rounds[-1] + 1)):
            raise ValueError(f"Missing rounds in {log_path}")
        all_rows[run["dataset"]][run["method"]] = rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_merged_csv(args.output_dir / "dynamic_client_group_metrics.csv", all_rows)
    for dataset, method_rows in sorted(all_rows.items()):
        for metric in METRICS:
            plot_metric(
                dataset,
                method_rows,
                metric,
                args.output_dir,
                args.formats,
                args.dpi,
            )

    run_count = sum(len(method_rows) for method_rows in all_rows.values())
    print(f"Plotted {run_count} successful runs from {len(all_rows)} datasets")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
