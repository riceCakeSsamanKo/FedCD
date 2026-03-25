#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


TINY_EXCLUDED_WNIDS_V1 = {
    # Direct or near-semantic overlap with CIFAR-10 classes:
    # airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    "n02124075": "cat-like",
    "n01641577": "frog-like",
    "n02106662": "dog-like",
    "n04146614": "vehicle-like",
    "n02058221": "bird-like",
    "n02099712": "dog-like",
    "n02056570": "bird-like",
    "n03796401": "vehicle-like",
    "n02123045": "cat-like",
    "n01855672": "bird-like",
    "n02917067": "vehicle-like",
    "n02423022": "deer-like",
    "n04465501": "vehicle-like",
    "n02099601": "dog-like",
    "n03977966": "vehicle-like",
    "n02125311": "cat-like",
    "n03670208": "vehicle-like",
    "n02002724": "bird-like",
    "n04285008": "vehicle-like",
    "n02403003": "deer-like",
    "n02094433": "dog-like",
    "n02814533": "vehicle-like",
    "n04487081": "vehicle-like",
    "n03100240": "vehicle-like",
    "n02415577": "deer-like",
    "n01644900": "frog-like",
    "n02129165": "cat-like",
    "n03447447": "ship-like",
    "n02113799": "dog-like",
    "n03662601": "ship-like",
}

# v2 is intentionally much more conservative: remove all TinyImageNet animal-like
# classes and obvious transport/vehicle classes so the proxy OOD pool does not
# contain CIFAR-10-near semantics.
TINY_EXCLUDED_WNIDS_V2 = {
    # Animal-like classes.
    "n01443537": "animal-like",
    "n01629819": "animal-like",
    "n01641577": "animal-like",
    "n01644900": "animal-like",
    "n01698640": "animal-like",
    "n01742172": "animal-like",
    "n01768244": "animal-like",
    "n01770393": "animal-like",
    "n01774384": "animal-like",
    "n01774750": "animal-like",
    "n01784675": "animal-like",
    "n01855672": "animal-like",
    "n01882714": "animal-like",
    "n01910747": "animal-like",
    "n01917289": "animal-like",
    "n01944390": "animal-like",
    "n01945685": "animal-like",
    "n01950731": "animal-like",
    "n01983481": "animal-like",
    "n01984695": "animal-like",
    "n02002724": "animal-like",
    "n02056570": "animal-like",
    "n02058221": "animal-like",
    "n02074367": "animal-like",
    "n02085620": "animal-like",
    "n02094433": "animal-like",
    "n02099601": "animal-like",
    "n02099712": "animal-like",
    "n02106662": "animal-like",
    "n02113799": "animal-like",
    "n02123045": "animal-like",
    "n02123394": "animal-like",
    "n02124075": "animal-like",
    "n02125311": "animal-like",
    "n02129165": "animal-like",
    "n02132136": "animal-like",
    "n02165456": "animal-like",
    "n02190166": "animal-like",
    "n02206856": "animal-like",
    "n02226429": "animal-like",
    "n02231487": "animal-like",
    "n02233338": "animal-like",
    "n02236044": "animal-like",
    "n02268443": "animal-like",
    "n02279972": "animal-like",
    "n02281406": "animal-like",
    "n02321529": "animal-like",
    "n02364673": "animal-like",
    "n02395406": "animal-like",
    "n02403003": "animal-like",
    "n02410509": "animal-like",
    "n02415577": "animal-like",
    "n02423022": "animal-like",
    "n02437312": "animal-like",
    "n02480495": "animal-like",
    "n02481823": "animal-like",
    "n02486410": "animal-like",
    "n02504458": "animal-like",
    "n02509815": "animal-like",
    # Transport / vehicle-like classes.
    "n02814533": "vehicle-like",
    "n02917067": "vehicle-like",
    "n03100240": "vehicle-like",
    "n03393912": "vehicle-like",
    "n03444034": "vehicle-like",
    "n03447447": "ship-like",
    "n03599486": "vehicle-like",
    "n03662601": "ship-like",
    "n03670208": "vehicle-like",
    "n03796401": "vehicle-like",
    "n03977966": "vehicle-like",
    "n04146614": "vehicle-like",
    "n04285008": "vehicle-like",
    "n04399382": "animal-like-object",
    "n04465501": "vehicle-like",
    "n04487081": "vehicle-like",
}

COCO_EXCLUDED_SUPERCATEGORIES = {"animal", "vehicle"}
COCO_EXCLUDED_CATEGORY_NAMES = set()


def parse_args():
    parser = argparse.ArgumentParser(description="Build a CIFAR-10 non-overlap OOD proxy dataset from TinyImageNet and COCO.")
    parser.add_argument(
        "--profile",
        choices=("v1", "v2"),
        default="v2",
        help="Filtering profile. v2 is the stricter non-overlap variant.",
    )
    parser.add_argument(
        "--tiny-root",
        type=Path,
        default=Path("/home/mulsoap0504/TinyImagenet/rawdata/tiny-imagenet-200"),
        help="TinyImageNet root directory",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("/home/mulsoap0504/coco"),
        help="COCO root directory containing train2017/val2017 and annotations/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. If omitted, a profile-specific default is used.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src, dst)


def load_tiny_words(tiny_root: Path):
    words_path = tiny_root / "words.txt"
    words = {}
    for line in words_path.read_text().splitlines():
        if not line.strip():
            continue
        wnid, desc = line.split("\t", 1)
        words[wnid.strip()] = desc.strip()
    return words


def build_tinyimagenet_split(tiny_root: Path, output_root: Path, split: str, words: dict, excluded_wnids: dict):
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
            if wnid in excluded_wnids:
                excluded += len(list((class_dir / "images").glob("*")))
                continue
            kept_classes.add(wnid)
            for image_path in sorted((class_dir / "images").glob("*")):
                symlink_file(image_path, target_dir / image_path.name)
                kept += 1
    elif split == "val":
        ann_path = tiny_root / "val" / "val_annotations.txt"
        image_to_wnid = {}
        for line in ann_path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                image_to_wnid[parts[0]] = parts[1]
        images_dir = tiny_root / "val" / "images"
        for image_path in sorted(images_dir.glob("*")):
            wnid = image_to_wnid.get(image_path.name)
            if not wnid:
                continue
            if wnid in excluded_wnids:
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
        "excluded_classes": sorted(excluded_wnids.keys()),
        "excluded_class_descriptions": {
            wnid: {
                "description": words.get(wnid, ""),
                "reason": excluded_wnids[wnid],
            }
            for wnid in sorted(excluded_wnids)
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
    excluded_image_count = 0
    kept_image_count = 0

    kept_category_names = set()
    excluded_category_names = set()

    for image_id, file_name in image_info.items():
        category_ids = image_to_categories.get(image_id, set())
        if not category_ids:
            continue
        category_names = {categories[cid]["name"] for cid in category_ids}
        supercategories = {categories[cid]["supercategory"] for cid in category_ids}

        should_exclude = bool(supercategories & COCO_EXCLUDED_SUPERCATEGORIES) or bool(
            category_names & COCO_EXCLUDED_CATEGORY_NAMES
        )
        if should_exclude:
            excluded_image_count += 1
            excluded_category_names.update(category_names)
            continue

        kept_category_names.update(category_names)
        src = image_root / file_name
        dst = target_dir / file_name
        symlink_file(src, dst)
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
    summary_path = output_root / "build_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    readme_path = output_root / "README.txt"
    readme_path.write_text(
        "CIFAR-10 non-overlap proxy OOD dataset built from TinyImageNet and COCO.\n"
        "Structure:\n"
        "  train/tinyimagenet_ood\n"
        "  train/coco_ood\n"
        "  val/tinyimagenet_ood\n"
        "  val/coco_ood\n"
        "Images are symlinked from the original datasets.\n"
        "See build_summary.json for filtering rules and counts.\n"
    )


def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = Path(
            f"/home/mulsoap0504/FedCD/ood_data/cifar10_proxy_tinyimagenet_coco_nonoverlap_{args.profile}"
        )
    words = load_tiny_words(args.tiny_root)
    tiny_excluded_wnids = TINY_EXCLUDED_WNIDS_V2 if args.profile == "v2" else TINY_EXCLUDED_WNIDS_V1

    summary = {
        "dataset_name": f"cifar10_proxy_tinyimagenet_coco_nonoverlap_{args.profile}",
        "profile": args.profile,
        "sources": {
            "tinyimagenet_root": str(args.tiny_root),
            "coco_root": str(args.coco_root),
        },
        "rationale": {
            "tinyimagenet": (
                "Exclude CIFAR-10-overlapping and near-overlapping TinyImageNet classes. "
                "v2 uses a more conservative animal/transport denylist."
            ),
            "coco": "Exclude any image containing vehicle or animal annotations to avoid supervision conflict with CIFAR-10 semantics.",
        },
        "splits": {},
    }

    for split in ("train", "val"):
        summary["splits"][split] = {
            "tinyimagenet": build_tinyimagenet_split(args.tiny_root, args.output_root, split, words, tiny_excluded_wnids),
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
