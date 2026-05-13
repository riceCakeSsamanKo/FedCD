#!/usr/bin/env python
"""Regenerate the canonical FL data setting used in the current runs.

Current setting:
  - CIFAR-10 and CIFAR-100: FedCD baseline style.
    The original torchvision train/test split is kept, then each split is
    partitioned across clients.
  - FashionMNIST: FedCCM style.
    The original torchvision train and test sets are merged, partitioned
    across clients, then each client is split 75/25 by dataset_utils.split_data.

The default target is the sibling ../fl_data directory.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "FedCD-Baseline" / "dataset"

TORCH = None
DATASET_SPECS = {}
separate_data = None
save_file = None
split_data = None

SCENARIOS = {
    "pat": ("pat", 0.1),
    "dir0.1": ("dir", 0.1),
    "dir0.5": ("dir", 0.5),
    "dir1.0": ("dir", 1.0),
}

DEFAULT_DATASETS = ("Cifar10", "Cifar100", "FashionMNIST")
DEFAULT_SCENARIOS = tuple(SCENARIOS)
DEFAULT_NUM_CLIENTS = (20, 50)


def load_generation_dependencies():
    global TORCH, DATASET_SPECS, separate_data, save_file, split_data

    if DATASET_SPECS:
        return

    try:
        import sys

        import torch
        import torchvision
        import torchvision.transforms as transforms
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing data-generation dependency. Run this script in the same "
            "PyTorch/torchvision environment used for FedCD/FedCCM experiments. "
            f"Original error: {exc}"
        ) from exc

    if str(DATASET_DIR) not in sys.path:
        sys.path.insert(0, str(DATASET_DIR))

    from utils.dataset_utils import (  # noqa: WPS433
        save_file as imported_save_file,
        separate_data as imported_separate_data,
        split_data as imported_split_data,
    )

    TORCH = torch
    separate_data = imported_separate_data
    save_file = imported_save_file
    split_data = imported_split_data
    DATASET_SPECS = {
        "Cifar10": {
            "torchvision": torchvision.datasets.CIFAR10,
            "raw_subdir": "Cifar10",
            "num_classes": 10,
            "class_per_client": 2,
            "style": "fedcd_original_split",
            "transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        },
        "Cifar100": {
            "torchvision": torchvision.datasets.CIFAR100,
            "raw_subdir": "Cifar100",
            "num_classes": 100,
            "class_per_client": 10,
            "style": "fedcd_original_split",
            "transform": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        },
        "FashionMNIST": {
            "torchvision": torchvision.datasets.FashionMNIST,
            "raw_subdir": "FashionMNIST",
            "num_classes": 10,
            "class_per_client": 2,
            "style": "fedccm_merged_75_25",
            "transform": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        },
    }


def load_torchvision_split(spec, raw_root: Path, train: bool, download: bool):
    dataset = spec["torchvision"](
        root=str(raw_root / spec["raw_subdir"]),
        train=train,
        download=download,
        transform=spec["transform"],
    )
    loader = TORCH.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False
    )
    images, labels = next(iter(loader))
    return images.cpu().numpy(), labels.cpu().numpy()


def output_dir(fl_data_root: Path, dataset_name: str, scenario_name: str, num_clients: int):
    return fl_data_root / f"{dataset_name}_{scenario_name}_nc{num_clients}"


def safe_delete_existing(
    fl_data_root: Path,
    datasets: tuple[str, ...],
    scenarios: tuple[str, ...],
    num_clients_values: tuple[int, ...],
    dry_run: bool = False,
):
    for dataset_name in datasets:
        for num_clients in num_clients_values:
            for scenario_name in scenarios:
                path = output_dir(fl_data_root, dataset_name, scenario_name, num_clients)
                if not path.is_dir():
                    continue
                print(f"[delete] {path}")
                if not dry_run:
                    shutil.rmtree(path)


def clear_npz_files(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    for npz_path in path.glob("*.npz"):
        npz_path.unlink()


def write_config_tags(config_path: Path, **tags):
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(tags)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f)


def write_original_split_dataset(
    fl_data_root: Path,
    raw_root: Path,
    dataset_name: str,
    scenario_name: str,
    partition: str,
    alpha: float,
    num_clients: int,
    download_raw: bool,
):
    spec = DATASET_SPECS[dataset_name]
    out_dir = output_dir(fl_data_root, dataset_name, scenario_name, num_clients)
    train_path = out_dir / "train"
    test_path = out_dir / "test"
    config_path = out_dir / "config.json"

    train_images, train_labels = load_torchvision_split(
        spec, raw_root, train=True, download=download_raw
    )
    test_images, test_labels = load_torchvision_split(
        spec, raw_root, train=False, download=download_raw
    )

    x_train, y_train, statistic = separate_data(
        (train_images, train_labels),
        num_clients,
        spec["num_classes"],
        niid=True,
        balance=True,
        partition=partition,
        class_per_client=spec["class_per_client"],
        alpha=alpha,
    )
    x_test, y_test, _ = separate_data(
        (test_images, test_labels),
        num_clients,
        spec["num_classes"],
        niid=True,
        balance=True,
        partition=partition,
        class_per_client=spec["class_per_client"],
        alpha=alpha,
    )

    train_data = [{"x": x_train[i], "y": y_train[i]} for i in range(num_clients)]
    test_data = [{"x": x_test[i], "y": y_test[i]} for i in range(num_clients)]

    clear_npz_files(train_path)
    clear_npz_files(test_path)
    save_file(
        str(config_path),
        str(train_path) + "/",
        str(test_path) + "/",
        train_data,
        test_data,
        num_clients,
        spec["num_classes"],
        statistic,
        niid=True,
        balance=True,
        partition=partition,
        alpha=alpha,
    )

    write_config_tags(
        config_path,
        generation_style=spec["style"],
        use_original_test_split=True,
        original_train_samples=int(len(train_labels)),
        original_test_samples=int(len(test_labels)),
    )

    print(
        f"[done] {out_dir.name}: style={spec['style']} "
        f"train={len(train_labels)} test={len(test_labels)} "
        f"partition={partition} alpha={alpha}"
    )


def write_merged_75_25_dataset(
    fl_data_root: Path,
    raw_root: Path,
    dataset_name: str,
    scenario_name: str,
    partition: str,
    alpha: float,
    num_clients: int,
    download_raw: bool,
):
    spec = DATASET_SPECS[dataset_name]
    out_dir = output_dir(fl_data_root, dataset_name, scenario_name, num_clients)
    train_path = out_dir / "train"
    test_path = out_dir / "test"
    config_path = out_dir / "config.json"

    train_images, train_labels = load_torchvision_split(
        spec, raw_root, train=True, download=download_raw
    )
    test_images, test_labels = load_torchvision_split(
        spec, raw_root, train=False, download=download_raw
    )
    dataset_images = np.concatenate([train_images, test_images], axis=0)
    dataset_labels = np.concatenate([train_labels, test_labels], axis=0)

    x_client, y_client, statistic = separate_data(
        (dataset_images, dataset_labels),
        num_clients,
        spec["num_classes"],
        niid=True,
        balance=True,
        partition=partition,
        class_per_client=spec["class_per_client"],
        alpha=alpha,
    )
    train_data, test_data = split_data(x_client, y_client)

    clear_npz_files(train_path)
    clear_npz_files(test_path)
    save_file(
        str(config_path),
        str(train_path) + "/",
        str(test_path) + "/",
        train_data,
        test_data,
        num_clients,
        spec["num_classes"],
        statistic,
        niid=True,
        balance=True,
        partition=partition,
        alpha=alpha,
    )

    write_config_tags(
        config_path,
        generation_style=spec["style"],
        use_original_test_split=False,
        merged_original_samples=int(len(dataset_labels)),
        train_ratio=0.75,
    )

    print(
        f"[done] {out_dir.name}: style={spec['style']} "
        f"total={len(dataset_labels)} partition={partition} alpha={alpha}"
    )


def generate_one(
    fl_data_root: Path,
    raw_root: Path,
    dataset_name: str,
    scenario_name: str,
    num_clients: int,
    download_raw: bool,
):
    partition, alpha = SCENARIOS[scenario_name]
    spec = DATASET_SPECS[dataset_name]
    if spec["style"] == "fedcd_original_split":
        write_original_split_dataset(
            fl_data_root,
            raw_root,
            dataset_name,
            scenario_name,
            partition,
            alpha,
            num_clients,
            download_raw,
        )
        return
    if spec["style"] == "fedccm_merged_75_25":
        write_merged_75_25_dataset(
            fl_data_root,
            raw_root,
            dataset_name,
            scenario_name,
            partition,
            alpha,
            num_clients,
            download_raw,
        )
        return
    raise ValueError(f"Unsupported generation style: {spec['style']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate CIFAR-10/CIFAR-100/FashionMNIST FL splits for the "
            "current reproducible FedCD/FedCCM data setting."
        )
    )
    parser.add_argument(
        "--fl-data-root",
        type=Path,
        default=REPO_ROOT.parent / "fl_data",
        help="Output root. Default: sibling ../fl_data",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Torchvision raw-data root. Default: <fl-data-root>/_raw",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
        help="Datasets to generate. Default: all canonical datasets.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
        help="Partition scenarios to generate. Default: pat dir0.1 dir0.5 dir1.0",
    )
    parser.add_argument(
        "--num-clients",
        nargs="+",
        type=int,
        choices=DEFAULT_NUM_CLIENTS,
        default=list(DEFAULT_NUM_CLIENTS),
        help="Client counts to generate. Default: 20 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base numpy/random seed. Default: 1",
    )
    parser.add_argument(
        "--reset-seed-before-fashion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reset the RNG to --seed before FashionMNIST. This matches the "
            "current workflow where FashionMNIST was regenerated separately."
        ),
    )
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--dry-run-delete", action="store_true")
    parser.add_argument(
        "--download-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let torchvision download missing raw datasets. Default: true.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fl_data_root = args.fl_data_root.expanduser().resolve()
    raw_root = (args.raw_root or (fl_data_root / "_raw")).expanduser().resolve()
    datasets = tuple(args.datasets)
    scenarios = tuple(args.scenarios)
    num_clients_values = tuple(args.num_clients)

    fl_data_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    if args.delete_existing or args.dry_run_delete:
        safe_delete_existing(
            fl_data_root,
            datasets,
            scenarios,
            num_clients_values,
            dry_run=args.dry_run_delete,
        )
        if args.dry_run_delete:
            return

    load_generation_dependencies()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for dataset_name in datasets:
        if dataset_name == "FashionMNIST" and args.reset_seed_before_fashion:
            random.seed(args.seed)
            np.random.seed(args.seed)
        for num_clients in num_clients_values:
            for scenario_name in scenarios:
                generate_one(
                    fl_data_root,
                    raw_root,
                    dataset_name,
                    scenario_name,
                    num_clients,
                    download_raw=args.download_raw,
                )


if __name__ == "__main__":
    main()
