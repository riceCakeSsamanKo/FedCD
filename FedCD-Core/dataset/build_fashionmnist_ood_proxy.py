#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


FASHIONMNIST_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# TinyImageNet classes that directly or near-directly overlap FashionMNIST
# clothing, shoe, and bag/accessory semantics.
TINY_EXCLUDED_WNIDS = {
    "n04507155": "umbrella/accessory-like",
    "n02963159": "cardigan/top-like",
    "n02769748": "backpack/bag-like",
    "n04254777": "sock/footwear-like",
    "n03404251": "fur coat/coat-like",
    "n02669723": "academic gown/dress-like",
    "n03770439": "miniskirt/dress-like",
    "n04023962": "punching bag/bag-like",
    "n04133789": "sandal/footwear-like",
    "n04532106": "vestment/clothing-like",
    "n03026506": "stocking/footwear-like",
    "n02730930": "apron/clothing-like",
    "n04371430": "swimming trunks/trouser-like",
    "n02837789": "bikini/clothing-like",
    "n02883205": "bow tie/accessory-like",
    "n03617480": "kimono/dress-like",
    "n03980874": "poncho/coat-like",
}

# COCO does not label most clothes explicitly, so person images are removed as
# near-overlap. Accessory categories are removed for bag/tie/suitcase overlap.
COCO_EXCLUDED_SUPERCATEGORIES = {"person", "accessory"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a FashionMNIST non-overlap OOD proxy dataset from TinyImageNet and COCO."
    )
    parser.add_argument(
        "--tiny-root",
        type=Path,
        default=Path("/home/mulsoap0504/TinyImagenet/rawdata/tiny-imagenet-200"),
        help="TinyImageNet root directory.",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("/home/mulsoap0504/coco"),
        help="COCO root directory containing train2017/val2017 and annotations/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/mulsoap0504/FedCD/ood_data/fashionmnist_proxy_tinyimagenet_coco_nonoverlap_v1"),
        help="Output dataset root.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def load_tiny_words(tiny_root: Path):
    words = {}
    for line in (tiny_root / "words.txt").read_text().splitlines():
        if not line.strip():
            continue
        wnid, desc = line.split("\t", 1)
        words[wnid.strip()] = desc.strip()
    return words


def build_tinyimagenet_split(tiny_root: Path, output_root: Path, split: str, words: dict):
    target_dir = output_root / split / "tinyimagenet_ood"
    ensure_dir(target_dir)

    kept = 0
    excluded = 0
    kept_classes = set()

    if split == "train":
        for class_dir in sorted((tiny_root / "train").iterdir()):
            if not class_dir.is_dir():
                continue
            wnid = class_dir.name
            image_paths = sorted((class_dir / "images").glob("*"))
            if wnid in TINY_EXCLUDED_WNIDS:
                excluded += len(image_paths)
                continue
            kept_classes.add(wnid)
            for image_path in image_paths:
                symlink_file(image_path, target_dir / image_path.name)
                kept += 1
    elif split == "val":
        image_to_wnid = {}
        for line in (tiny_root / "val" / "val_annotations.txt").read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                image_to_wnid[parts[0]] = parts[1]

        for image_path in sorted((tiny_root / "val" / "images").glob("*")):
            wnid = image_to_wnid.get(image_path.name)
            if not wnid:
                continue
            if wnid in TINY_EXCLUDED_WNIDS:
                excluded += 1
                continue
            kept_classes.add(wnid)
            symlink_file(image_path, target_dir / image_path.name)
            kept += 1
    else:
        raise ValueError(f"Unsupported TinyImageNet split: {split}")

    return {
        "kept_images": kept,
        "excluded_images": excluded,
        "kept_classes": sorted(kept_classes),
        "excluded_classes": {
            wnid: {
                "description": words.get(wnid, ""),
                "reason": reason,
            }
            for wnid, reason in sorted(TINY_EXCLUDED_WNIDS.items())
        },
    }


def load_coco_instances(annotation_path: Path):
    with annotation_path.open("r") as f:
        data = json.load(f)
    categories = {cat["id"]: cat for cat in data["categories"]}
    image_info = {img["id"]: img["file_name"] for img in data["images"]}
    image_to_categories = defaultdict(set)
    for ann in data["annotations"]:
        image_to_categories[ann["image_id"]].add(ann["category_id"])
    return categories, image_info, image_to_categories


def build_coco_split(coco_root: Path, output_root: Path, split: str):
    split_name = f"{split}2017"
    annotation_path = coco_root / "annotations" / f"instances_{split_name}.json"
    image_root = coco_root / split_name
    target_dir = output_root / split / "coco_ood"
    ensure_dir(target_dir)

    categories, image_info, image_to_categories = load_coco_instances(annotation_path)
    kept_image_count = 0
    excluded_image_count = 0
    kept_category_names = set()
    excluded_category_names = set()

    for image_id, file_name in image_info.items():
        category_ids = image_to_categories.get(image_id, set())
        if not category_ids:
            continue

        category_names = {categories[cid]["name"] for cid in category_ids}
        supercategories = {categories[cid]["supercategory"] for cid in category_ids}
        if supercategories & COCO_EXCLUDED_SUPERCATEGORIES:
            excluded_image_count += 1
            excluded_category_names.update(category_names)
            continue

        kept_category_names.update(category_names)
        symlink_file(image_root / file_name, target_dir / file_name)
        kept_image_count += 1

    return {
        "kept_images": kept_image_count,
        "excluded_images": excluded_image_count,
        "kept_category_names": sorted(kept_category_names),
        "excluded_supercategories": sorted(COCO_EXCLUDED_SUPERCATEGORIES),
        "excluded_category_names": sorted(excluded_category_names),
    }


def write_summary(output_root: Path, summary: dict):
    ensure_dir(output_root)
    with (output_root / "build_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    (output_root / "README.txt").write_text(
        "FashionMNIST non-overlap proxy OOD dataset built from TinyImageNet and COCO.\n"
        "Task dataset: FashionMNIST\n"
        "Proxy sources: TinyImageNet + COCO\n"
        "Proxy channels: 1\n"
        "Structure:\n"
        "  train/tinyimagenet_ood\n"
        "  train/coco_ood\n"
        "  val/tinyimagenet_ood\n"
        "  val/coco_ood\n"
        "Images are symlinked from the original datasets and should be loaded as grayscale for FashionMNIST models.\n"
        "TinyImageNet excludes clothing, footwear, and bag/accessory-like classes.\n"
        "COCO excludes any image containing person or accessory supercategories.\n"
        "See build_summary.json for filtering rules and counts.\n"
    )


def main():
    args = parse_args()
    words = load_tiny_words(args.tiny_root)
    summary = {
        "dataset_name": args.output_root.name,
        "sources": {
            "tinyimagenet_root": str(args.tiny_root),
            "coco_root": str(args.coco_root),
        },
        "fashionmnist_labels": FASHIONMNIST_LABELS,
        "rationale": {
            "tinyimagenet": "Exclude classes directly or near-directly overlapping FashionMNIST clothing, footwear, and bag/accessory semantics.",
            "coco": "Exclude images containing person or accessory annotations to avoid FashionMNIST clothing and bag/accessory overlap.",
            "channels": "Load proxy images as grayscale because FashionMNIST FedCCM models use one input channel.",
        },
        "splits": {},
    }

    for split in ("train", "val"):
        summary["splits"][split] = {
            "tinyimagenet": build_tinyimagenet_split(args.tiny_root, args.output_root, split, words),
            "coco": build_coco_split(args.coco_root, args.output_root, split),
        }

    write_summary(args.output_root, summary)

    print(f"Built OOD proxy dataset at: {args.output_root}")
    for split in ("train", "val"):
        tiny = summary["splits"][split]["tinyimagenet"]
        coco = summary["splits"][split]["coco"]
        print(
            f"[{split}] TinyImageNet kept={tiny['kept_images']} excluded={tiny['excluded_images']} | "
            f"COCO kept={coco['kept_images']} excluded={coco['excluded_images']}"
        )


if __name__ == "__main__":
    main()
