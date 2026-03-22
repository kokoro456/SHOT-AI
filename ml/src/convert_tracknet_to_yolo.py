"""
Convert TrackNet ball_combined.json to YOLO format labels.

Input:  ball_combined.json with entries like:
  {"image": "game1_Clip1_frame_0001.jpg", "x": 0.45, "y": 0.32, "visibility": 1}

Output: One .txt file per image in YOLO format:
  class_id x_center y_center width height
  - class_id = 0 (ball)
  - x, y already normalized [0,1]
  - width = 0.015 (~20px / 1280px)
  - height = 0.028 (~20px / 720px)
  - visibility == 0 → empty .txt file (no ball visible)
"""

import json
import os
import shutil
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\Users\kim\OneDrive\잡\바탕 화면\TENNIS SHOT\ml\data")
JSON_PATH = DATA_ROOT / "ball_tracknet" / "ball_combined.json"
FRAMES_DIR = DATA_ROOT / "ball_tracknet" / "frames"
OUT_DIR = DATA_ROOT / "yolo_tracknet"

# YOLO bbox size for a tennis ball (~20px in 1280x720)
BALL_W = 0.015
BALL_H = 0.028

# ── train/val split ────────────────────────────────────────────────────
VAL_RATIO = 0.15  # ~15% validation


def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {JSON_PATH}")

    # Create output dirs
    for split in ("train", "val"):
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect unique game-clip combos for split by clip (not by frame)
    clips = {}
    for entry in data:
        # e.g. "game1_Clip1_frame_0001.jpg" → "game1_Clip1"
        parts = entry["image"].rsplit("_frame_", 1)
        clip_key = parts[0] if len(parts) == 2 else entry["image"]
        clips.setdefault(clip_key, []).append(entry)

    clip_keys = sorted(clips.keys())
    n_val = max(1, int(len(clip_keys) * VAL_RATIO))
    val_clips = set(clip_keys[-n_val:])  # last N clips go to val
    print(f"Total clips: {len(clip_keys)}, val clips: {n_val}")

    stats = {"train": {"total": 0, "visible": 0}, "val": {"total": 0, "visible": 0}}
    missing_frames = 0

    for clip_key, entries in clips.items():
        split = "val" if clip_key in val_clips else "train"

        for entry in entries:
            img_name = entry["image"]
            stem = Path(img_name).stem
            vis = entry["visibility"]

            # Check if source frame exists
            src_frame = FRAMES_DIR / img_name
            if not src_frame.exists():
                missing_frames += 1
                continue

            # Symlink or copy image
            dst_img = OUT_DIR / split / "images" / img_name
            if not dst_img.exists():
                # Use relative symlink if possible, else copy
                try:
                    os.symlink(src_frame, dst_img)
                except OSError:
                    shutil.copy2(src_frame, dst_img)

            # Write label
            label_path = OUT_DIR / split / "labels" / f"{stem}.txt"
            if vis == 0:
                # No ball visible → empty label file
                label_path.write_text("")
            else:
                x = entry["x"]
                y = entry["y"]
                # Clamp to [0, 1]
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                label_path.write_text(f"0 {x:.6f} {y:.6f} {BALL_W} {BALL_H}\n")
                stats[split]["visible"] += 1

            stats[split]["total"] += 1

    # Write data.yaml
    yaml_path = OUT_DIR / "data.yaml"
    yaml_path.write_text(
        f"train: {(OUT_DIR / 'train' / 'images').as_posix()}\n"
        f"val: {(OUT_DIR / 'val' / 'images').as_posix()}\n"
        f"nc: 1\n"
        f"names: ['ball']\n"
    )

    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Missing frames (skipped): {missing_frames}")
    print(f"Train: {stats['train']['total']} total, {stats['train']['visible']} with ball")
    print(f"Val:   {stats['val']['total']} total, {stats['val']['visible']} with ball")
    print(f"Output: {OUT_DIR}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
