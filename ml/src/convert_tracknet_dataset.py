"""
TrackNet 공개 데이터셋 → SHOT 학습 형식 변환.

TrackNet Label.csv (픽셀 좌표) → ball_combined.json (정규화 좌표)
연속 프레임을 유지하면서 video_id(game+clip) 기반으로 매핑.

Usage:
    python convert_tracknet_dataset.py \
        --input "../../ball data/Dataset-001/Dataset" \
        --output ../data/ball_tracknet \
        --copy-frames
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


IMG_W = 1280
IMG_H = 720


def convert_dataset(input_dir: str, output_dir: str, copy_frames: bool = False):
    input_path = Path(input_dir)
    frames_out = Path(output_dir) / "frames"
    os.makedirs(frames_out, exist_ok=True)

    all_annotations = []
    stats = {"total": 0, "visible": 0, "not_visible": 0, "occluded": 0,
             "games": 0, "clips": 0}

    games = sorted([d for d in input_path.iterdir()
                     if d.is_dir() and d.name.startswith("game")])
    stats["games"] = len(games)

    for game_dir in games:
        game_id = game_dir.name  # e.g. "game1"
        clips = sorted([c for c in game_dir.iterdir()
                         if c.is_dir() and c.name.startswith("Clip")])

        for clip_dir in clips:
            clip_id = clip_dir.name  # e.g. "Clip1"
            video_id = f"{game_id}_{clip_id}"  # e.g. "game1_Clip1"
            stats["clips"] += 1

            label_file = clip_dir / "Label.csv"
            if not label_file.exists():
                print(f"  SKIP {video_id}: no Label.csv")
                continue

            with open(label_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row["file name"].strip()
                    vis_str = row["visibility"].strip()
                    x_str = row["x-coordinate"].strip()
                    y_str = row["y-coordinate"].strip()

                    # Handle empty or invalid values
                    if not vis_str or not fname:
                        continue
                    vis = int(vis_str)
                    try:
                        x_px = float(x_str) if x_str else 0.0
                        y_px = float(y_str) if y_str else 0.0
                    except ValueError:
                        x_px, y_px = 0.0, 0.0
                        vis = 0  # Mark as not visible

                    # New filename with video_id prefix
                    new_fname = f"{video_id}_frame_{fname}"

                    # Normalize coordinates
                    if vis == 0:
                        x_norm, y_norm = -1.0, -1.0
                    else:
                        x_norm = x_px / IMG_W
                        y_norm = y_px / IMG_H

                    ann = {
                        "image": new_fname,
                        "x": round(x_norm, 6),
                        "y": round(y_norm, 6),
                        "visibility": vis
                    }
                    all_annotations.append(ann)
                    stats["total"] += 1

                    if vis == 0:
                        stats["not_visible"] += 1
                    elif vis == 1:
                        stats["visible"] += 1
                    elif vis == 2:
                        stats["occluded"] += 1

                    # Copy frame
                    if copy_frames:
                        src = clip_dir / fname
                        dst = frames_out / new_fname
                        if src.exists() and not dst.exists():
                            shutil.copy2(str(src), str(dst))

    # Save annotations
    ann_path = Path(output_dir) / "ball_combined.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)

    print(f"\n=== Conversion Complete ===")
    print(f"Games: {stats['games']}, Clips: {stats['clips']}")
    print(f"Total frames: {stats['total']}")
    print(f"  Visible (1):     {stats['visible']}")
    print(f"  Not visible (0): {stats['not_visible']}")
    print(f"  Occluded (2):    {stats['occluded']}")
    print(f"\nAnnotations: {ann_path}")
    if copy_frames:
        frame_count = len(list(frames_out.glob("*.jpg")))
        print(f"Frames copied: {frame_count} -> {frames_out}")

    return all_annotations


def main():
    parser = argparse.ArgumentParser(description="Convert TrackNet dataset to SHOT format")
    parser.add_argument("--input", required=True, help="TrackNet Dataset root dir")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--copy-frames", action="store_true",
                        help="Copy frame images to output dir")
    args = parser.parse_args()

    convert_dataset(args.input, args.output, args.copy_frames)


if __name__ == "__main__":
    main()
