"""
Convert tennis ball annotation JSON to YOLO format.

Sources:
  1. ml/data/phone_ball/ball_annotations.json  (809 entries, park_* + phone_ball_01)
  2. phone ball data/ball_annotations.json      (49 entries, phone_ball_01)

Merges both sources (deduplicating by image filename, preferring source 1).
Images come from:
  - ml/data/phone_ball/frames_720p/  (park_* and phone_ball_01 frames at 720p)
  - ml/data/phone_ball/frames/       (phone_ball_01 originals, fallback)

Output: ml/data/yolo_ball/
  images/train/, images/val/
  labels/train/, labels/val/
  data.yaml

Split: 80/20 by video ID (group frames from same video together).
"""

import json
import os
import shutil
import random
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT = Path(r"C:\Users\kim\OneDrive\잡\바탕 화면\TENNIS SHOT")

ANNO_FILES = [
    PROJECT / "ml" / "data" / "phone_ball" / "ball_annotations.json",
    PROJECT / "phone ball data" / "ball_annotations.json",
]

IMAGE_DIRS = [
    PROJECT / "ml" / "data" / "phone_ball" / "frames_720p",
    PROJECT / "ml" / "data" / "phone_ball" / "frames",
]

OUTPUT_DIR = PROJECT / "ml" / "data" / "yolo_ball"

# Ball bbox size (normalized). ~20px ball in 1280x720 => w=20/1280, h=20/720
BALL_W = 0.015
BALL_H = 0.028

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def extract_video_id(filename: str) -> str:
    """Extract video ID from filename like 'park_20260321_101449_frame_0002.jpg'."""
    parts = filename.rsplit("_frame_", 1)
    if len(parts) == 2:
        return parts[0]
    # Fallback: use the whole filename minus extension
    return Path(filename).stem


def find_image(filename: str) -> Path | None:
    """Find image file in the known image directories."""
    for img_dir in IMAGE_DIRS:
        p = img_dir / filename
        if p.exists():
            return p
    return None


def load_annotations() -> dict[str, dict]:
    """Load and merge annotations from all sources. Keyed by image filename."""
    merged = {}
    for anno_file in ANNO_FILES:
        if not anno_file.exists():
            print(f"  [SKIP] {anno_file} not found")
            continue
        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        count_new = 0
        for entry in data:
            img = entry["image"]
            if img not in merged:
                merged[img] = entry
                count_new += 1
        print(f"  Loaded {anno_file.name}: {len(data)} entries, {count_new} new")
    return merged


def make_yolo_label(entry: dict) -> str:
    """Convert annotation entry to YOLO label line. Returns empty string for invisible."""
    vis = entry.get("visibility", 0)
    if vis <= 0:
        return ""  # empty file = negative sample

    x = entry["x"]
    y = entry["y"]

    # Sanity check: skip invalid coordinates
    if x < 0 or x > 1 or y < 0 or y > 1:
        return ""

    # Clamp bbox to [0, 1]
    x_center = max(0.0, min(1.0, x))
    y_center = max(0.0, min(1.0, y))
    w = BALL_W
    h = BALL_H

    # Ensure bbox doesn't exceed image bounds
    if x_center - w / 2 < 0:
        w = x_center * 2
    if x_center + w / 2 > 1:
        w = (1 - x_center) * 2
    if y_center - h / 2 < 0:
        h = y_center * 2
    if y_center + h / 2 > 1:
        h = (1 - y_center) * 2

    return f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def main():
    random.seed(RANDOM_SEED)

    print("Loading annotations...")
    annotations = load_annotations()
    print(f"Total merged annotations: {len(annotations)}")

    # Group by video ID
    video_groups: dict[str, list[str]] = {}
    for img_name in annotations:
        vid = extract_video_id(img_name)
        video_groups.setdefault(vid, []).append(img_name)

    print(f"\nVideo groups ({len(video_groups)}):")
    for vid, imgs in sorted(video_groups.items()):
        vis_count = sum(1 for i in imgs if annotations[i].get("visibility", 0) > 0)
        print(f"  {vid}: {len(imgs)} frames ({vis_count} visible)")

    # Split by video ID
    video_ids = sorted(video_groups.keys())
    random.shuffle(video_ids)
    split_idx = max(1, int(len(video_ids) * TRAIN_RATIO))
    train_vids = set(video_ids[:split_idx])
    val_vids = set(video_ids[split_idx:])

    print(f"\nSplit: train={len(train_vids)} videos, val={len(val_vids)} videos")
    print(f"  Train: {train_vids}")
    print(f"  Val:   {val_vids}")

    # Create output directories
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process
    stats = {"train": {"total": 0, "visible": 0, "missing": 0},
             "val": {"total": 0, "visible": 0, "missing": 0}}

    for img_name, entry in sorted(annotations.items()):
        vid = extract_video_id(img_name)
        split = "train" if vid in train_vids else "val"

        # Find source image
        src_img = find_image(img_name)
        if src_img is None:
            stats[split]["missing"] += 1
            continue

        # Copy image
        dst_img = OUTPUT_DIR / "images" / split / img_name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write label
        label_name = Path(img_name).stem + ".txt"
        dst_label = OUTPUT_DIR / "labels" / split / label_name
        label_content = make_yolo_label(entry)
        with open(dst_label, "w") as f:
            if label_content:
                f.write(label_content + "\n")
            # else: empty file = negative sample

        stats[split]["total"] += 1
        if label_content:
            stats[split]["visible"] += 1

    # Write data.yaml
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['ball']
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    # Summary
    print("\n=== Results ===")
    for split in ["train", "val"]:
        s = stats[split]
        neg = s["total"] - s["visible"]
        print(f"  {split}: {s['total']} images "
              f"({s['visible']} positive, {neg} negative, {s['missing']} missing)")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"data.yaml written to: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
