#!/usr/bin/env python
"""Build the canonical OOD proxy datasets without changing FL splits.

This is a thin wrapper around FedCCM/tools/prepare_fedccmv19_data.py. The
FedCCM script also knows how to generate FashionMNIST FL splits, but this
wrapper imports only the proxy-building functions so it does not overwrite
the canonical FL data directories.
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PROXY_DIRS = {
    "mnist": (
        "fashionmnist_proxy_mnist",
        "cifar10_proxy_mnist_rgb",
    ),
    "tinyimagenet": (
        "cifar10_proxy_tinyimagenet_nonoverlap",
        "cifar100_proxy_tinyimagenet_nonoverlap",
        "fashionmnist_proxy_tinyimagenet_nonoverlap",
    ),
    "coco": (
        "cifar10_proxy_coco_nonoverlap",
        "cifar100_proxy_coco_nonoverlap",
        "fashionmnist_proxy_coco_nonoverlap",
    ),
    "combined": (
        "cifar10_proxy_tinyimagenet_coco_nonoverlap_v2",
        "cifar100_proxy_tinyimagenet_coco_nonoverlap_v1",
        "fashionmnist_proxy_tinyimagenet_coco_nonoverlap_v1",
    ),
}


def load_fedccm_prepare_module(fedccm_root: Path):
    module_path = fedccm_root / "tools" / "prepare_fedccmv19_data.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"FedCCM proxy script not found: {module_path}. "
            "Clone FedCCM next to FedCD or pass --fedccm-root."
        )

    spec = importlib.util.spec_from_file_location(
        "fedccm_prepare_fedccmv19_data", module_path
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_proxy_dirs(proxy_source: str) -> tuple[str, ...]:
    if proxy_source == "current":
        keys = ("mnist", "tinyimagenet", "coco", "combined")
    elif proxy_source == "tinyimagenet_coco":
        keys = ("tinyimagenet", "coco", "combined")
    else:
        keys = (proxy_source,)

    dirs = []
    for key in keys:
        dirs.extend(PROXY_DIRS[key])
    return tuple(dirs)


def delete_existing_proxy_dirs(proxy_root: Path, proxy_source: str, dry_run: bool):
    for dirname in selected_proxy_dirs(proxy_source):
        path = proxy_root / dirname
        if not path.exists():
            continue
        print(f"[delete] {path}")
        if not dry_run:
            shutil.rmtree(path)


def fill_url_defaults(args, module):
    if args.coco_annotations_url is None:
        args.coco_annotations_url = module.COCO_ANNOTATIONS_URL
    if args.coco_image_base_url is None:
        args.coco_image_base_url = module.COCO_IMAGE_BASE_URL
    if args.coco_train_images_url is None:
        args.coco_train_images_url = module.COCO_TRAIN_IMAGES_URL
    if args.coco_val_images_url is None:
        args.coco_val_images_url = module.COCO_VAL_IMAGES_URL
    if args.tinyimagenet_url is None:
        args.tinyimagenet_url = module.TINYIMAGENET_URL


def prepare_mnist_proxy(module, proxy_root: Path, train_max: int, val_max: int):
    fashion_proxy_root = module.prepare_fashionmnist_proxy(proxy_root)
    module.prepare_cifar10_proxy(proxy_root, fashion_proxy_root, train_max, val_max)


def prepare_tinyimagenet_proxy(module, raw_root: Path, proxy_root: Path, args):
    module.prepare_tinyimagenet_proxies(
        raw_root,
        proxy_root,
        args.tinyimagenet_url,
        args.tinyimagenet_train_max,
        args.tinyimagenet_val_max,
    )


def prepare_coco_proxy(module, raw_root: Path, proxy_root: Path, args):
    module.prepare_coco_proxies(
        raw_root,
        proxy_root,
        args.coco_annotations_url,
        args.coco_image_base_url,
        args.coco_image_source,
        args.coco_train_images_url,
        args.coco_val_images_url,
        args.coco_train_max,
        args.coco_val_max,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare canonical FedCCMV19 OOD proxy datasets."
    )
    parser.add_argument(
        "--fl-data-root",
        type=Path,
        default=REPO_ROOT.parent / "fl_data",
        help="Federated data root. Default: sibling ../fl_data",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Raw download/cache root. Default: <fl-data-root>/_raw",
    )
    parser.add_argument(
        "--proxy-root",
        type=Path,
        default=None,
        help="Proxy output root. Default: <fl-data-root>/ood_proxy",
    )
    parser.add_argument(
        "--fedccm-root",
        type=Path,
        default=REPO_ROOT.parent / "FedCCM",
        help="FedCCM repository root. Default: sibling ../FedCCM",
    )
    parser.add_argument(
        "--proxy-source",
        choices=("current", "mnist", "tinyimagenet", "coco", "tinyimagenet_coco"),
        default="current",
        help=(
            "Proxy family to build. 'current' builds MNIST plus "
            "TinyImageNet+COCO proxy families. Default: current."
        ),
    )
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--dry-run-delete", action="store_true")
    parser.add_argument("--cifar10-proxy-train-max", type=int, default=2048)
    parser.add_argument("--cifar10-proxy-val-max", type=int, default=1024)
    parser.add_argument("--coco-train-max", type=int, default=1024)
    parser.add_argument("--coco-val-max", type=int, default=512)
    parser.add_argument("--coco-annotations-url", default=None)
    parser.add_argument("--coco-image-base-url", default=None)
    parser.add_argument(
        "--coco-image-source",
        choices=("zip", "per-image"),
        default="zip",
        help="Default: zip",
    )
    parser.add_argument("--coco-train-images-url", default=None)
    parser.add_argument("--coco-val-images-url", default=None)
    parser.add_argument("--tinyimagenet-url", default=None)
    parser.add_argument("--tinyimagenet-train-max", type=int, default=48976)
    parser.add_argument("--tinyimagenet-val-max", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    fl_data_root = args.fl_data_root.expanduser().resolve()
    raw_root = (args.raw_root or (fl_data_root / "_raw")).expanduser().resolve()
    proxy_root = (args.proxy_root or (fl_data_root / "ood_proxy")).expanduser().resolve()
    fedccm_root = args.fedccm_root.expanduser().resolve()

    module = load_fedccm_prepare_module(fedccm_root)
    fill_url_defaults(args, module)

    raw_root.mkdir(parents=True, exist_ok=True)
    proxy_root.mkdir(parents=True, exist_ok=True)

    if args.delete_existing or args.dry_run_delete:
        delete_existing_proxy_dirs(proxy_root, args.proxy_source, args.dry_run_delete)
        if args.dry_run_delete:
            return

    if args.proxy_source in ("current", "mnist"):
        prepare_mnist_proxy(
            module,
            proxy_root,
            args.cifar10_proxy_train_max,
            args.cifar10_proxy_val_max,
        )

    if args.proxy_source in ("current", "tinyimagenet", "tinyimagenet_coco"):
        prepare_tinyimagenet_proxy(module, raw_root, proxy_root, args)

    if args.proxy_source in ("current", "coco", "tinyimagenet_coco"):
        prepare_coco_proxy(module, raw_root, proxy_root, args)

    if args.proxy_source in ("current", "tinyimagenet_coco"):
        module.prepare_combined_tinyimagenet_coco_proxy(proxy_root)

    print("[ood-proxy] Done.")


if __name__ == "__main__":
    main()
