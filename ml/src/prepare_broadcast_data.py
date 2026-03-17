"""
Download and prepare yastrebksv/TennisCourtDetector broadcast dataset.

This downloads the original broadcast camera training data and converts
it to SHOT format for comparison experiments.

Usage:
    python prepare_broadcast_data.py
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# TennisCourtDetector dataset on GitHub
REPO_URL = "https://github.com/yastrebksv/TennisCourtDetector"
DATA_TRAIN_URL = "https://raw.githubusercontent.com/yastrebksv/TennisCourtDetector/main/data/data_train.json"
DATA_VAL_URL = "https://raw.githubusercontent.com/yastrebksv/TennisCourtDetector/main/data/data_val.json"

# yastrebksv keypoint index -> SHOT keypoint mapping
# Only near-court keypoints (the ones visible from behind baseline)
YASTREBKSV_TO_SHOT = {
    10: "9",   # Near service line × singles left
    13: "10",  # Near service line × center mark
    11: "11",  # Near service line × singles right
    2:  "12",  # Near baseline × doubles left
    5:  "13",  # Near baseline × singles left
    7:  "15",  # Near baseline × singles right
    3:  "16",  # Near baseline × doubles right
}
# Pt14 (baseline center) = midpoint of yastrebksv[2] and yastrebksv[3]


def download_file(url, dest):
    """Download a file with progress."""
    print(f"  Downloading {url}")
    try:
        urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def convert_entry(entry, image_w=1920, image_h=1080):
    """Convert a yastrebksv annotation to SHOT format."""
    kps_raw = entry["kps"]  # [[x0,y0], [x1,y1], ..., [x13,y13]]

    keypoints = {}

    for yidx, shot_id in YASTREBKSV_TO_SHOT.items():
        if yidx < len(kps_raw):
            x_norm = kps_raw[yidx][0] / image_w
            y_norm = kps_raw[yidx][1] / image_h
            keypoints[shot_id] = {
                "x": round(x_norm, 6),
                "y": round(y_norm, 6),
                "visible": True,
            }
        else:
            keypoints[shot_id] = {"x": 0, "y": 0, "visible": False}

    # Pt14 = midpoint of doubles left (2) and doubles right (3)
    if 2 < len(kps_raw) and 3 < len(kps_raw):
        x14 = (kps_raw[2][0] + kps_raw[3][0]) / 2 / image_w
        y14 = (kps_raw[2][1] + kps_raw[3][1]) / 2 / image_h
        keypoints["14"] = {
            "x": round(x14, 6),
            "y": round(y14, 6),
            "visible": True,
        }
    else:
        keypoints["14"] = {"x": 0, "y": 0, "visible": False}

    return {
        "image": entry["id"] + ".png",
        "keypoints": keypoints,
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    broadcast_dir = base_dir / "data" / "broadcast"
    broadcast_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloading TennisCourtDetector dataset ===\n")

    # Download annotation files
    train_json = broadcast_dir / "data_train.json"
    val_json = broadcast_dir / "data_val.json"

    if not train_json.exists():
        download_file(DATA_TRAIN_URL, str(train_json))
    if not val_json.exists():
        download_file(DATA_VAL_URL, str(val_json))

    # Load and convert
    all_annotations = []

    for json_file in [train_json, val_json]:
        if not json_file.exists():
            print(f"WARNING: {json_file} not found, skipping")
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"\n{json_file.name}: {len(data)} entries")

        for entry in data:
            converted = convert_entry(entry)
            all_annotations.append(converted)

    # Save converted annotations
    output_file = broadcast_dir / "annotations_broadcast.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)

    print(f"\n=== Results ===")
    print(f"Total broadcast annotations: {len(all_annotations)}")
    print(f"Saved to: {output_file}")

    # Download images instruction
    images_dir = broadcast_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"\n=== Image Download ===")
    print(f"Images need to be downloaded from the TennisCourtDetector repo.")
    print(f"Run: git clone {REPO_URL} /tmp/tcd")
    print(f"Then copy images to: {images_dir}")
    print(f"\nOr download automatically:")

    # Try to clone and get images
    import subprocess
    temp_clone = base_dir / "data" / "_temp_tcd"

    if not any(images_dir.iterdir()) if images_dir.exists() else True:
        print(f"Attempting to clone repo for images...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(temp_clone)],
                check=True, capture_output=True, timeout=120
            )

            # Find and copy images
            img_src = temp_clone / "data"
            count = 0
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for img_file in img_src.rglob(ext):
                    if img_file.name.startswith("data_"):
                        continue  # skip json files
                    dest = images_dir / img_file.name
                    if not dest.exists():
                        import shutil
                        shutil.copy2(img_file, dest)
                        count += 1
            print(f"Copied {count} images to {images_dir}")

            # Cleanup temp
            import shutil
            shutil.rmtree(temp_clone, ignore_errors=True)

        except Exception as e:
            print(f"Auto-download failed: {e}")
            print(f"Please manually download images from {REPO_URL}")
    else:
        print(f"Images already exist: {len(list(images_dir.iterdir()))} files")


if __name__ == "__main__":
    main()
