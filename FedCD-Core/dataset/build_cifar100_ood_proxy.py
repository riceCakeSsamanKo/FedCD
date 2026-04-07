#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


# CIFAR-100 covers a much broader semantic space than CIFAR-10, so we use a
# noticeably stricter denylist for TinyImageNet and a whitelist for COCO.
CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

# A conservative keyword list: enough to avoid near-ID semantics for CIFAR-100,
# but still leaves a usable OOD pool.
TINY_EXCLUSION_KEYWORDS = sorted(
    set(
        """
        apple aquarium fish goldfish flatfish ray shark trout
        baby boy girl man woman person human
        bear beaver bed bee beetle bicycle bottle bowl bridge bus butterfly camel can castle
        caterpillar cattle chair chimpanzee clock cloud cockroach couch crab crocodile cup dinosaur
        dolphin elephant forest fox hamster house kangaroo keyboard lamp lawn mower leopard lion lizard
        lobster maple oak palm pine willow mountain mouse mushroom orange orchid otter pear pickup truck
        pine plain plate poppy porcupine possum rabbit raccoon road rocket rose sea seal shark shrew skunk
        snail snake spider squirrel streetcar sunflower sweet pepper table tank telephone television tv tiger
        tractor train turtle wardrobe whale wolf worm deer dog cat horse frog bird goose penguin albatross
        retriever terrier poodle chihuahua sheep ox bison gazelle orangutan baboon koala panda alligator boa
        salamander turtle lizard snake scorpion tarantula widow dragonfly grasshopper fly mantis centipede
        flower tree plant fruit vegetable vehicle wagon truck motorcycle ship boat
        skyscraper
        """
        .split()
    )
)

# Keep only COCO categories that are clearly outside CIFAR-100 semantics.
COCO_ALLOWED_CATEGORY_NAMES = {
    "person",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "fork", "knife", "spoon",
    "banana", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "toilet",
    "laptop", "mouse", "remote", "cell phone",
    "microwave", "oven", "sink", "refrigerator",
    "book", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a CIFAR-100-specific non-overlap OOD proxy from TinyImageNet and COCO."
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
        default=Path("/home/mulsoap0504/FedCD/ood_data/cifar100_proxy_tinyimagenet_coco_nonoverlap_v1"),
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
    words_path = tiny_root / "words.txt"
    words = {}
    for line in words_path.read_text().splitlines():
        if not line.strip():
            continue
        wnid, desc = line.split("\t", 1)
        words[wnid.strip()] = desc.strip()
    return words


def compile_keyword_pattern(keywords):
    return re.compile(
        "|".join(re.escape(keyword.lower()) for keyword in sorted(keywords, key=len, reverse=True))
    )


def build_tinyimagenet_split(tiny_root: Path, output_root: Path, split: str, words: dict, keyword_pattern):
    target_dir = output_root / split / "tinyimagenet_ood"
    ensure_dir(target_dir)

    kept = 0
    excluded = 0
    kept_classes = set()
    excluded_classes = {}

    def should_exclude(wnid: str):
        desc = words.get(wnid, "").lower()
        match = keyword_pattern.search(desc)
        return match is not None, match.group(0) if match else ""

    if split == "train":
        for class_dir in sorted((tiny_root / "train").iterdir()):
            if not class_dir.is_dir():
                continue
            wnid = class_dir.name
            exclude, reason = should_exclude(wnid)
            image_paths = sorted((class_dir / "images").glob("*"))
            if exclude:
                excluded += len(image_paths)
                excluded_classes[wnid] = {"description": words.get(wnid, ""), "matched_keyword": reason}
                continue
            kept_classes.add(wnid)
            for image_path in image_paths:
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
            exclude, reason = should_exclude(wnid)
            if exclude:
                excluded += 1
                excluded_classes.setdefault(
                    wnid,
                    {"description": words.get(wnid, ""), "matched_keyword": reason},
                )
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
        "excluded_classes": dict(sorted(excluded_classes.items())),
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
        # Exclude images if any category falls outside the whitelist.
        if not category_names.issubset(COCO_ALLOWED_CATEGORY_NAMES):
            excluded_image_count += 1
            excluded_category_names.update(category_names - COCO_ALLOWED_CATEGORY_NAMES)
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
        "allowed_category_names": sorted(COCO_ALLOWED_CATEGORY_NAMES),
        "excluded_category_names": sorted(excluded_category_names),
    }


def write_summary(output_root: Path, summary: dict):
    ensure_dir(output_root)
    with (output_root / "build_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    (output_root / "README.txt").write_text(
        "CIFAR-100-specific non-overlap proxy OOD dataset built from TinyImageNet and COCO.\n"
        "Structure:\n"
        "  train/tinyimagenet_ood\n"
        "  train/coco_ood\n"
        "  val/tinyimagenet_ood\n"
        "  val/coco_ood\n"
        "Images are symlinked from the original datasets.\n"
        "TinyImageNet uses a broad CIFAR-100 overlap denylist.\n"
        "COCO keeps only a whitelist of categories judged safely outside CIFAR-100 semantics.\n"
        "See build_summary.json for filtering rules and counts.\n"
    )


def main():
    args = parse_args()
    words = load_tiny_words(args.tiny_root)
    tiny_pattern = compile_keyword_pattern(TINY_EXCLUSION_KEYWORDS)

    summary = {
        "dataset_name": args.output_root.name,
        "sources": {
            "tinyimagenet_root": str(args.tiny_root),
            "coco_root": str(args.coco_root),
        },
        "cifar100_fine_labels": CIFAR100_FINE_LABELS,
        "rationale": {
            "tinyimagenet": (
                "Exclude TinyImageNet classes whose descriptions overlap with CIFAR-100 fine classes or close semantic families."
            ),
            "coco": (
                "Keep only COCO categories that are clearly outside CIFAR-100 semantics; exclude mixed images containing any non-whitelisted category."
            ),
        },
        "splits": {},
    }

    for split in ("train", "val"):
        summary["splits"][split] = {
            "tinyimagenet": build_tinyimagenet_split(
                args.tiny_root, args.output_root, split, words, tiny_pattern
            ),
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
