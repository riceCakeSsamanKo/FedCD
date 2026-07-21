#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


DEFAULT_DATASETS = ('Cifar10', 'FashionMNIST')
DEFAULT_METHODS = (
    'FedAvg',
    'FedProx',
    'FedCross',
    'FedBN',
    'FedALA',
    'FedAS',
    'pFedMe',
    'cwFedAvg',
    'FedDST',
    'PMOE_FedPer',
    'FedCP',
    'DualFed',
)
VALID_SCENARIOS = ('id', 'ood', 'mix')


@dataclass(frozen=True)
class RunBest:
    path: Path
    value: float
    best_round: int
    max_round: int


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(',') if item.strip())


def log_dataset_name(dataset: str) -> str:
    key = dataset.strip().lower().replace('-', '')
    if key == 'cifar10':
        return 'cifar10'
    if key == 'fashionmnist':
        return 'FashionMNIST'
    return dataset


def display_dataset_name(dataset: str) -> str:
    return 'CIFAR10' if log_dataset_name(dataset) == 'cifar10' else 'FMNIST'


def metric_for_method(method: str) -> str:
    if method.lower() == 'pfedme':
        return 'personalized_local_test_acc'
    return 'local_test_acc'


def to_float(value: object) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def read_completed_best(path: Path, metric: str, required_round: int) -> RunBest | None:
    try:
        with path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    if not rows or metric not in rows[0] or 'round' not in rows[0]:
        return None

    values: list[tuple[int, float]] = []
    for row in rows:
        round_value = to_float(row.get('round'))
        metric_value = to_float(row.get(metric))
        if round_value is None or metric_value is None:
            continue
        values.append((int(round_value), metric_value))
    if not values:
        return None
    max_round = max(item[0] for item in values)
    if max_round < required_round:
        return None
    best_round, best_value = max(values, key=lambda item: (item[1], -item[0]))
    return RunBest(path=path, value=best_value, best_round=best_round, max_round=max_round)


def collect_runs(
    logs_root: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    model: str,
    clients: int,
    scenarios: tuple[str, ...],
    required_round: int,
) -> dict[tuple[str, str, str], list[RunBest]]:
    runs: dict[tuple[str, str, str], list[RunBest]] = defaultdict(list)
    for dataset in datasets:
        log_dataset = log_dataset_name(dataset)
        for method in methods:
            metric = metric_for_method(method)
            base = (
                logs_root
                / log_dataset
                / method
                / f'GM_{model}'
                / 'fedprism_idoodmix'
                / f'NC_{clients}'
            )
            for scenario in scenarios:
                pattern = f'date_*/time_*/eval_{scenario}/acc.csv'
                for path in sorted(base.glob(pattern)):
                    run = read_completed_best(path, metric, required_round)
                    if run is not None:
                        runs[(dataset, method, scenario)].append(run)
    return runs


def format_values(values: list[float], scale: str, decimals: int) -> str:
    factor = 100.0 if scale == 'percent' else 1.0
    scaled = [value * factor for value in values]
    center = mean(scaled)
    spread = stdev(scaled) if len(scaled) > 1 else 0.0
    return f'{center:.{decimals}f} ± {spread:.{decimals}f}'


def write_summary(
    output: Path,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    scenarios: tuple[str, ...],
    runs: dict[tuple[str, str, str], list[RunBest]],
    target_runs: int,
    scale: str,
    decimals: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Dataset', 'Method', *[item.upper() for item in scenarios], 'Runs'])
        for dataset in datasets:
            for method in methods:
                cells = []
                run_counts = []
                for scenario in scenarios:
                    candidates = runs.get((dataset, method, scenario), [])
                    selected = candidates[-target_runs:] if len(candidates) >= target_runs else []
                    cells.append(
                        format_values([item.value for item in selected], scale, decimals)
                        if selected
                        else ''
                    )
                    run_counts.append(str(len(selected)))
                writer.writerow(
                    [display_dataset_name(dataset), method, *cells, '/'.join(run_counts)]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Summarize completed FedPRISM ID/OOD/Mix baseline runs.'
    )
    parser.add_argument('--logs-root', type=Path, default=Path('logs'))
    parser.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    parser.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    parser.add_argument('--model', default='VGG8')
    parser.add_argument('--clients', type=int, default=50)
    parser.add_argument('--scenarios', default=','.join(VALID_SCENARIOS))
    parser.add_argument('--required-round', type=int, default=101)
    parser.add_argument('--target-runs', type=int, default=1)
    parser.add_argument('--scale', choices=('raw', 'percent'), default='percent')
    parser.add_argument('--decimals', type=int, default=2)
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'fedprism_idoodmix_result.csv',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_csv_list(args.datasets)
    methods = parse_csv_list(args.methods)
    scenarios = tuple(item.lower() for item in parse_csv_list(args.scenarios))
    unknown = sorted(set(scenarios) - set(VALID_SCENARIOS))
    if unknown:
        raise SystemExit(f'Unknown scenario(s): {unknown}')
    if args.target_runs < 1:
        raise SystemExit('--target-runs must be positive')

    runs = collect_runs(
        args.logs_root,
        datasets,
        methods,
        args.model,
        args.clients,
        scenarios,
        args.required_round,
    )
    write_summary(
        args.output_csv,
        datasets,
        methods,
        scenarios,
        runs,
        args.target_runs,
        args.scale,
        args.decimals,
    )
    completed = sum(len(items) for items in runs.values())
    print(f'Wrote {args.output_csv} from {completed} completed scenario logs.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
