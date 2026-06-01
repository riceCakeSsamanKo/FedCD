#!/usr/bin/env python3
"""Generate SplitGP-style rho test splits for FedCCM and FedCD.

The generated dataset keeps the repository's existing FL data format:

    <output-root>/<dataset-name>/train/<client-id>.npz
    <output-root>/<dataset-name>/test/<client-id>.npz

Each ``.npz`` stores ``data={"x": ndarray, "y": ndarray}``, so both FedCCM
and the FedCD baseline can read it through their existing data loaders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


DEFAULT_RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)
DETERMINISTIC_SPLIT_VERSION = 2


DATASET_SPECS = {
    "cifar10": {
        "display_name": "Cifar10",
        "torchvision_cls": torchvision.datasets.CIFAR10,
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "fashionmnist": {
        "display_name": "FashionMNIST",
        "torchvision_cls": torchvision.datasets.FashionMNIST,
        "train_kwargs": {"train": True},
        "test_kwargs": {"train": False},
        "transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_root() -> Path:
    return repo_root().parent / "fl_data"


def rho_tag(rho: float) -> str:
    return f"{rho:.1f}"


def dataset_output_name(display_name: str, rho: float, num_clients: int) -> str:
    return f"{display_name}_splitgp_pat_rho{rho_tag(rho)}_nc{num_clients}"


def resolve_dataset_raw_root(raw_root: Path, display_name: str) -> Path:
    dataset_root = raw_root / display_name
    if dataset_root.exists():
        return dataset_root
    return raw_root


def deterministic_key(seed: int, context: str, value: int) -> bytes:
    raw = f"{seed}:{context}:{int(value)}".encode("utf-8")
    return hashlib.sha256(raw).digest()


def deterministic_order(values, seed: int, context: str) -> np.ndarray:
    ordered = sorted((int(value) for value in values), key=lambda v: deterministic_key(seed, context, v))
    return np.array(ordered, dtype=np.int64)


def deterministic_permutation(size: int, seed: int, context: str) -> np.ndarray:
    return deterministic_order(range(size), seed, context)


def load_torchvision_split(spec: dict, raw_root: Path, train: bool, download: bool):
    dataset_cls = spec["torchvision_cls"]
    kwargs = spec["train_kwargs"] if train else spec["test_kwargs"]
    dataset = dataset_cls(
        root=str(raw_root),
        download=download,
        transform=spec["transform"],
        **kwargs,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
    )
    x_tensor, y_tensor = next(iter(loader))
    x = x_tensor.cpu().numpy().astype(np.float32, copy=False)
    y = y_tensor.cpu().numpy().astype(np.int64, copy=False)
    return x, y


def make_splitgp_train_partition(
    train_y: np.ndarray,
    *,
    dataset_key: str,
    num_clients: int,
    num_shards: int,
    shards_per_client: int,
    seed: int,
) -> tuple[list[np.ndarray], list[list[int]]]:
    if num_shards != num_clients * shards_per_client:
        raise ValueError(
            "SplitGP shard setup requires num_shards == "
            "num_clients * shards_per_client; got "
            f"{num_shards} != {num_clients} * {shards_per_client}"
        )
    if len(train_y) % num_shards != 0:
        raise ValueError(
            f"Cannot divide {len(train_y)} training samples into "
            f"{num_shards} equal shards"
        )

    sorted_indices = np.argsort(train_y, kind="stable")
    shards = np.array_split(sorted_indices, num_shards)
    shard_order = deterministic_permutation(num_shards, seed, f"{dataset_key}:train:shards")
    client_indices = []
    client_shard_labels = []

    for client_id in range(num_clients):
        selected = shard_order[
            client_id * shards_per_client : (client_id + 1) * shards_per_client
        ]
        idx = np.concatenate([shards[shard_id] for shard_id in selected])
        idx = idx[
            deterministic_permutation(
                len(idx), seed, f"{dataset_key}:train:client:{client_id}"
            )
        ]
        client_indices.append(idx)
        client_shard_labels.append(
            [int(np.bincount(train_y[shards[shard_id]]).argmax()) for shard_id in selected]
        )

    return client_indices, client_shard_labels


def label_statistics(labels: np.ndarray) -> list[list[int]]:
    counts = Counter(int(label) for label in labels.tolist())
    return [[label, counts[label]] for label in sorted(counts)]


def make_client_test_indices(
    test_y: np.ndarray,
    main_classes: list[int],
    rho: float,
    *,
    dataset_key: str,
    client_id: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    main_classes_arr = np.array(main_classes, dtype=np.int64)
    main_mask = np.isin(test_y, main_classes_arr)
    main_idx = np.where(main_mask)[0]
    ood_pool = np.where(~main_mask)[0]
    ood_count = int(round(float(rho) * len(main_idx)))

    if ood_count <= 0:
        selected_ood = np.empty((0,), dtype=np.int64)
    else:
        if len(ood_pool) == 0:
            raise ValueError(f"Client {client_id} has no OOD test pool")
        ordered_ood = deterministic_order(
            ood_pool, seed, f"{dataset_key}:rho:{rho_tag(rho)}:client:{client_id}:ood"
        )
        if ood_count <= len(ordered_ood):
            selected_ood = ordered_ood[:ood_count]
        else:
            repeats = int(np.ceil(ood_count / len(ordered_ood)))
            selected_ood = np.tile(ordered_ood, repeats)[:ood_count]

    test_idx = np.concatenate([main_idx, selected_ood])
    test_idx = test_idx[
        deterministic_permutation(
            len(test_idx), seed, f"{dataset_key}:rho:{rho_tag(rho)}:client:{client_id}:test"
        )
    ]
    return test_idx, int(len(main_idx)), int(ood_count)


def update_array_hash(hasher, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(array)
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(arr.tobytes())


def content_fingerprint(train_data: list[dict], test_data: list[dict]) -> str:
    hasher = hashlib.sha256()
    for split_name, items in (("train", train_data), ("test", test_data)):
        hasher.update(split_name.encode("utf-8"))
        for client_id, item in enumerate(items):
            hasher.update(str(client_id).encode("utf-8"))
            update_array_hash(hasher, item["x"])
            update_array_hash(hasher, item["y"])
    return hasher.hexdigest()


def write_client_npz(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"x": x, "y": y.astype(np.int64, copy=False)}
    with path.open("wb") as f:
        np.savez_compressed(f, data=payload)


def config_matches(config_path: Path, expected: dict) -> bool:
    if not config_path.exists():
        return False
    try:
        with config_path.open("r", encoding="utf-8") as f:
            found = json.load(f)
    except Exception:
        return False

    keys = [
        "dataset_source",
        "num_clients",
        "num_classes",
        "partition",
        "partition_detail",
        "splitgp_rho",
        "splitgp_num_shards",
        "splitgp_shards_per_client",
        "seed",
        "deterministic_split_version",
        "content_sha256",
    ]
    return all(found.get(key) == expected.get(key) for key in keys)


def validate_dataset(root: Path, num_clients: int) -> None:
    for split in ("train", "test"):
        for client_id in range(num_clients):
            path = root / split / f"{client_id}.npz"
            with np.load(path, allow_pickle=True) as archive:
                data = archive["data"].tolist()
            if not {"x", "y"}.issubset(data):
                raise RuntimeError(f"Invalid payload keys in {path}")
            if len(data["x"]) != len(data["y"]):
                raise RuntimeError(f"Mismatched x/y lengths in {path}")


def generate_one_dataset(
    *,
    dataset_key: str,
    rho: float,
    output_root: Path,
    raw_root: Path,
    num_clients: int,
    num_shards: int,
    shards_per_client: int,
    seed: int,
    download: bool,
    force: bool,
) -> Path:
    spec = DATASET_SPECS[dataset_key]
    display_name = spec["display_name"]
    output_name = dataset_output_name(display_name, rho, num_clients)
    output_dir = output_root / output_name
    config_path = output_dir / "config.json"
    dataset_raw_root = resolve_dataset_raw_root(raw_root, display_name)

    train_x, train_y = load_torchvision_split(
        spec, dataset_raw_root, train=True, download=download
    )
    test_x, test_y = load_torchvision_split(
        spec, dataset_raw_root, train=False, download=download
    )
    num_classes = int(len(np.unique(train_y)))

    client_train_indices, client_shard_labels = make_splitgp_train_partition(
        train_y,
        dataset_key=dataset_key,
        num_clients=num_clients,
        num_shards=num_shards,
        shards_per_client=shards_per_client,
        seed=seed,
    )

    train_stats = []
    client_train_classes = []
    train_data = []
    test_data = []
    test_counts = []

    for client_id, train_idx in enumerate(client_train_indices):
        client_train_x = train_x[train_idx]
        client_train_y = train_y[train_idx]
        main_classes = sorted(int(label) for label in np.unique(client_train_y))
        test_idx, main_count, ood_count = make_client_test_indices(
            test_y,
            main_classes,
            rho,
            dataset_key=dataset_key,
            client_id=client_id,
            seed=seed,
        )

        train_stats.append(label_statistics(client_train_y))
        client_train_classes.append(main_classes)
        train_data.append({"x": client_train_x, "y": client_train_y})
        test_data.append({"x": test_x[test_idx], "y": test_y[test_idx]})
        test_counts.append(
            {
                "client_id": client_id,
                "main_samples": main_count,
                "ood_samples": ood_count,
                "total_samples": int(main_count + ood_count),
            }
        )

    expected_config = {
        "dataset_source": display_name,
        "num_clients": int(num_clients),
        "num_classes": int(num_classes),
        "non_iid": True,
        "balance": True,
        "partition": "pat",
        "partition_detail": "splitgp_shard",
        "Size of samples for labels in clients": train_stats,
        "alpha": 0.0,
        "batch_size": 10,
        "seed": int(seed),
        "deterministic_split_version": DETERMINISTIC_SPLIT_VERSION,
        "source_train_samples": int(len(train_y)),
        "source_test_samples": int(len(test_y)),
        "splitgp_rho": float(rho),
        "splitgp_num_shards": int(num_shards),
        "splitgp_shards_per_client": int(shards_per_client),
        "splitgp_test_definition": (
            "Client test set is all original test samples whose labels are in "
            "the client's train classes plus rho * #main samples drawn from "
            "the remaining original test classes."
        ),
        "client_train_classes": client_train_classes,
        "client_shard_labels": client_shard_labels,
        "client_test_counts": test_counts,
        "content_sha256": content_fingerprint(train_data, test_data),
    }

    if output_dir.exists() and not force and config_matches(config_path, expected_config):
        print(f"[skip] {output_name} already exists and matches requested settings")
        return output_dir
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    elif output_dir.exists() and not force:
        raise FileExistsError(
            f"{output_dir} already exists but config does not match. "
            "Use --force to overwrite."
        )

    for client_id in range(num_clients):
        write_client_npz(
            output_dir / "train" / f"{client_id}.npz",
            train_data[client_id]["x"],
            train_data[client_id]["y"],
        )
        write_client_npz(
            output_dir / "test" / f"{client_id}.npz",
            test_data[client_id]["x"],
            test_data[client_id]["y"],
        )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(expected_config, f, indent=2)

    validate_dataset(output_dir, num_clients)
    print(f"[ok] generated {output_name} at {output_dir}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SplitGP-style CIFAR-10/FashionMNIST rho datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "fashionmnist"],
        choices=sorted(DATASET_SPECS),
    )
    parser.add_argument(
        "--rhos",
        nargs="+",
        type=float,
        default=list(DEFAULT_RHOS),
        help="rho values: #OOD test samples / #main test samples",
    )
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=100)
    parser.add_argument("--shards-per-client", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-root",
        "--data-root",
        "--save-root",
        dest="output_root",
        type=Path,
        default=default_output_root(),
        help=(
            "Root directory where generated FL datasets are saved. "
            "Aliases: --data-root, --save-root. Defaults to ../fl_data."
        ),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Torchvision raw-data root. Defaults to <output-root>/_raw.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not let torchvision download missing raw data.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output dataset directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    raw_root = (args.raw_root or (output_root / "_raw")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] output root: {output_root}")
    print(f"[info] raw data root: {raw_root}")

    generated = []
    for dataset_key in args.datasets:
        for rho in args.rhos:
            if rho < 0:
                raise ValueError(f"rho must be non-negative, got {rho}")
            generated.append(
                generate_one_dataset(
                    dataset_key=dataset_key,
                    rho=float(rho),
                    output_root=output_root,
                    raw_root=raw_root,
                    num_clients=args.num_clients,
                    num_shards=args.num_shards,
                    shards_per_client=args.shards_per_client,
                    seed=args.seed,
                    download=not args.no_download,
                    force=args.force,
                )
            )

    manifest_path = output_root / "splitgp_rho_manifest.json"
    manifest = {
        "datasets": [str(path.name) for path in generated],
        "output_root": str(output_root),
        "raw_root": str(raw_root),
        "num_clients": int(args.num_clients),
        "num_shards": int(args.num_shards),
        "shards_per_client": int(args.shards_per_client),
        "rhos": [float(rho) for rho in args.rhos],
        "seed": int(args.seed),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
