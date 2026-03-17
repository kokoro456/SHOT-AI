"""
Convert yastrebksv/TennisCourtDetector dataset to SHOT annotation format.

yastrebksv format:
    data_train.json / data_val.json with entries:
    {"id": "image_name", "kps": [[x0,y0], [x1,y1], ..., [x13,y13]]}

    14 keypoints (0-indexed):
    0:  Far baseline × doubles left    (286, 561)
    1:  Far baseline × doubles right   (1379, 561)
    2:  Near baseline × doubles left   (286, 2935)   → SHOT 12
    3:  Near baseline × doubles right  (1379, 2935)   → SHOT 16
    4:  Far singles left               (423, 561)
    5:  Near singles left              (423, 2935)    → SHOT 13
    6:  Far singles right              (1242, 561)
    7:  Near singles right             (1242, 2935)   → SHOT 15
    8:  Far service × singles left     (423, 1110)
    9:  Far service × singles right    (1242, 1110)
    10: Near service × singles left    (423, 2386)    → SHOT 9
    11: Near service × singles right   (1242, 2386)   → SHOT 11
    12: Far center service line        (832, 1110)
    13: Near center service line       (832, 2386)    → SHOT 10

SHOT format (8 near-court keypoints):
    annotations.json with entries:
    {"image": "filename.png", "keypoints": {
        "9":  {"x": ..., "y": ..., "visible": true},
        "10": {"x": ..., "y": ..., "visible": true},
        ...
        "16": {"x": ..., "y": ..., "visible": true}
    }}

    SHOT keypoint mapping:
    9:  Near service line × singles left    ← yastrebksv[10]
    10: Near service line × center mark     ← yastrebksv[13]
    11: Near service line × singles right   ← yastrebksv[11]
    12: Near baseline × doubles left        ← yastrebksv[2]
    13: Near baseline × singles left        ← yastrebksv[5]
    14: Near baseline × center mark         ← computed midpoint of yastrebksv[2] and [3]
    15: Near baseline × singles right       ← yastrebksv[7]
    16: Near baseline × doubles right       ← yastrebksv[3]

Usage:
    python convert_dataset.py --input data/raw/tennis_court --output data/raw/annotations.json
"""

import argparse
import json
import os
from pathlib import Path


# Mapping: SHOT keypoint ID → yastrebksv index
KEYPOINT_MAP = {
    9: 10,   # near service left
    10: 13,  # near center service
    11: 11,  # near service right
    12: 2,   # near baseline doubles left
    13: 5,   # near baseline singles left
    14: None,  # center mark (computed)
    15: 7,   # near baseline singles right
    16: 3,   # near baseline doubles right
}

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def convert_entry(entry):
    """Convert a single yastrebksv annotation entry to SHOT format."""
    kps = entry["kps"]  # [[x0,y0], ..., [x13,y13]]
    image_id = entry["id"]

    keypoints = {}
    for shot_id, src_idx in KEYPOINT_MAP.items():
        if shot_id == 14:
            # Center mark = midpoint of near baseline doubles left/right
            x = (kps[2][0] + kps[3][0]) / 2
            y = (kps[2][1] + kps[3][1]) / 2
        else:
            x = kps[src_idx][0]
            y = kps[src_idx][1]

        # Normalize to [0, 1]
        x_norm = x / IMAGE_WIDTH
        y_norm = y / IMAGE_HEIGHT

        # Check visibility (point must be within image bounds)
        visible = 0 <= x <= IMAGE_WIDTH and 0 <= y <= IMAGE_HEIGHT

        keypoints[str(shot_id)] = {
            "x": round(x_norm, 6),
            "y": round(y_norm, 6),
            "visible": visible,
        }

    return {
        "image": f"{image_id}.png",
        "keypoints": keypoints,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert yastrebksv dataset to SHOT format")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to extracted dataset directory (containing data_train.json, data_val.json, images/)")
    parser.add_argument("--output", type=str, default="data/raw/annotations.json",
                        help="Output annotation file path")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to data/raw/images/ directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    annotations = []

    # Load both train and val splits
    for split_file in ["data_train.json", "data_val.json"]:
        json_path = input_dir / split_file
        if not json_path.exists():
            print(f"Warning: {json_path} not found, skipping")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        print(f"Processing {split_file}: {len(data)} entries")

        for entry in data:
            converted = convert_entry(entry)

            # Verify image exists
            img_path = input_dir / "images" / converted["image"]
            if not img_path.exists():
                continue

            annotations.append(converted)

    print(f"\nTotal converted annotations: {len(annotations)}")

    # Filter out entries where near-court points are out of frame
    valid = []
    for ann in annotations:
        kps = ann["keypoints"]
        # Require at least 6 visible near-court keypoints (our minimum)
        visible_count = sum(1 for k in kps.values() if k["visible"])
        if visible_count >= 6:
            valid.append(ann)

    print(f"Valid annotations (≥6 visible keypoints): {len(valid)}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(valid, f, indent=2)

    print(f"Saved to: {args.output}")

    # Copy images if requested
    if args.copy_images:
        import shutil
        img_out_dir = Path(os.path.dirname(args.output)) / "images"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        for ann in valid:
            src = input_dir / "images" / ann["image"]
            dst = img_out_dir / ann["image"]
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        print(f"Copied {len(valid)} images to {img_out_dir}")

    # Print sample
    if valid:
        print("\nSample annotation:")
        sample = valid[0]
        print(f"  Image: {sample['image']}")
        for kid, kp in sample["keypoints"].items():
            print(f"  Point {kid}: x={kp['x']:.4f}, y={kp['y']:.4f}, visible={kp['visible']}")


if __name__ == "__main__":
    main()
