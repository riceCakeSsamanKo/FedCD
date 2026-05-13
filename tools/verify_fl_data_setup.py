#!/usr/bin/env python
"""Verify the canonical FL data and OOD proxy directory layout."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Cifar10", "Cifar100", "FashionMNIST")
SCENARIOS = {
    "pat": ("pat", 0.1),
    "dir0.1": ("dir", 0.1),
    "dir0.5": ("dir", 0.5),
    "dir1.0": ("dir", 1.0),
}
NUM_CLIENTS = (20, 50)
EXPECTED_PROXY_DIRS = (
    "cifar10_proxy_mnist_rgb",
    "cifar10_proxy_coco_nonoverlap",
    "cifar10_proxy_tinyimagenet_nonoverlap",
    "cifar10_proxy_tinyimagenet_coco_nonoverlap_v2",
    "cifar100_proxy_coco_nonoverlap",
    "cifar100_proxy_tinyimagenet_nonoverlap",
    "cifar100_proxy_tinyimagenet_coco_nonoverlap_v1",
    "fashionmnist_proxy_mnist",
    "fashionmnist_proxy_coco_nonoverlap",
    "fashionmnist_proxy_tinyimagenet_nonoverlap",
    "fashionmnist_proxy_tinyimagenet_coco_nonoverlap_v1",
)


def count_npz_samples(path: Path) -> int:
    total = 0
    for npz_path in sorted(path.glob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            total += len(data["data"].item()["y"])
    return total


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expected_counts(dataset_name: str):
    if dataset_name in ("Cifar10", "Cifar100"):
        return 50000, 10000
    return None, None


def verify_fl_dataset(root: Path, dataset_name: str, scenario_name: str, num_clients: int):
    path = root / f"{dataset_name}_{scenario_name}_nc{num_clients}"
    if not path.is_dir():
        return False, f"missing {path.name}"

    config_path = path / "config.json"
    if not config_path.is_file():
        return False, f"missing config {path.name}"

    config = read_config(config_path)
    expected_partition, expected_alpha = SCENARIOS[scenario_name]
    problems = []
    if config.get("num_clients") != num_clients:
        problems.append(f"num_clients={config.get('num_clients')}")
    if config.get("partition") != expected_partition:
        problems.append(f"partition={config.get('partition')}")
    if float(config.get("alpha", -1.0)) != float(expected_alpha):
        problems.append(f"alpha={config.get('alpha')}")
    if config.get("non_iid") is not True:
        problems.append(f"non_iid={config.get('non_iid')}")
    if config.get("balance") is not True:
        problems.append(f"balance={config.get('balance')}")

    train_count = count_npz_samples(path / "train")
    test_count = count_npz_samples(path / "test")

    if dataset_name in ("Cifar10", "Cifar100"):
        exp_train, exp_test = expected_counts(dataset_name)
        if config.get("use_original_test_split") is not True:
            problems.append("use_original_test_split is not true")
        if train_count != exp_train or test_count != exp_test:
            problems.append(f"samples={train_count}/{test_count}")
    else:
        total = train_count + test_count
        if config.get("use_original_test_split") is True:
            problems.append("FashionMNIST should not use original test split")
        if total != 70000:
            problems.append(f"total={total}")
        if not (52480 <= train_count <= 52500 and 17500 <= test_count <= 17520):
            problems.append(f"samples={train_count}/{test_count}")

    if problems:
        return False, f"{path.name}: " + ", ".join(problems)

    return True, f"{path.name}: train={train_count} test={test_count}"


def parse_train_val_from_readme(readme_path: Path):
    if not readme_path.is_file():
        return None
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    train_match = re.search(r"Train images:\s*(\d+)", text)
    val_match = re.search(r"Val images:\s*(\d+)", text)
    if not train_match and not val_match:
        return "metadata present"
    train = train_match.group(1) if train_match else "?"
    val = val_match.group(1) if val_match else "?"
    return f"train={train} val={val}"


def verify_proxy_dirs(root: Path):
    proxy_root = root / "ood_proxy"
    results = []
    for dirname in EXPECTED_PROXY_DIRS:
        path = proxy_root / dirname
        if not path.is_dir():
            results.append((False, f"missing ood_proxy/{dirname}"))
            continue
        details = parse_train_val_from_readme(path / "README.txt")
        suffix = f": {details}" if details else ""
        results.append((True, f"ood_proxy/{dirname}{suffix}"))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify canonical FedCD/FedCCM data generation outputs."
    )
    parser.add_argument(
        "--fl-data-root",
        type=Path,
        default=REPO_ROOT.parent / "fl_data",
        help="Data root. Default: sibling ../fl_data",
    )
    parser.add_argument("--check-ood-proxy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.fl_data_root.expanduser().resolve()

    failures = []
    for dataset_name in DATASETS:
        for num_clients in NUM_CLIENTS:
            for scenario_name in SCENARIOS:
                ok, message = verify_fl_dataset(root, dataset_name, scenario_name, num_clients)
                prefix = "OK" if ok else "FAIL"
                print(f"[{prefix}] {message}")
                if not ok:
                    failures.append(message)

    if args.check_ood_proxy:
        for ok, message in verify_proxy_dirs(root):
            prefix = "OK" if ok else "FAIL"
            print(f"[{prefix}] {message}")
            if not ok:
                failures.append(message)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
