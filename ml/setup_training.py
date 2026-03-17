"""
Home Desktop Training Setup Script

Run this on your home desktop to download all training data
and prepare for the 3-way comparison experiment.

Usage:
    cd ml
    python setup_training.py
"""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


def download_with_progress(url, dest, description):
    """Download a file with progress indicator."""
    print(f"\n[Downloading] {description}")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb = count * block_size / (1024 * 1024)
        print(f"\r  {percent}% ({mb:.1f} MB)", end="", flush=True)

    urlretrieve(url, dest, reporthook)
    print(f"\n  Done: {os.path.getsize(dest) / (1024*1024):.1f} MB")


def main():
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    print("=" * 60)
    print("  SHOT Training Data Setup")
    print("=" * 60)

    # ========================================
    # Step 1: YouTube labeled data (from GitHub Release)
    # ========================================
    youtube_dir = base_dir / "data" / "youtube"
    review_dir = youtube_dir / "review"
    frames_dir = review_dir / "frames"

    if (youtube_dir / "labeled_annotations.json").exists() and any(frames_dir.glob("*.jpg")):
        print(f"\n[OK] YouTube labeled data already exists ({len(list(frames_dir.glob('*.jpg')))} frames)")
    else:
        zip_url = "https://github.com/kokoro456/SHOT-AI/releases/download/v0.1-data/youtube_labeled_data.zip"
        zip_path = youtube_dir / "youtube_labeled_data.zip"
        youtube_dir.mkdir(parents=True, exist_ok=True)

        download_with_progress(zip_url, str(zip_path), "YouTube labeled data (26 MB)")

        print("\n[Extracting] youtube_labeled_data.zip...")
        frames_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.startswith("frames/"):
                    # Extract to review/frames/
                    filename = os.path.basename(member)
                    if filename:
                        with zf.open(member) as src, open(frames_dir / filename, "wb") as dst:
                            dst.write(src.read())
                elif member == "labeled_annotations.json":
                    with zf.open(member) as src, open(youtube_dir / "labeled_annotations.json", "wb") as dst:
                        dst.write(src.read())

        frame_count = len(list(frames_dir.glob("*.jpg")))
        annotations = json.load(open(youtube_dir / "labeled_annotations.json"))
        print(f"  Extracted: {frame_count} frames, {len(annotations)} annotations")

    # ========================================
    # Step 2: Broadcast data (from Google Drive)
    # ========================================
    broadcast_dir = base_dir / "data" / "broadcast"
    broadcast_images = broadcast_dir / "data" / "images"
    broadcast_annotations = broadcast_dir / "annotations_broadcast.json"

    if broadcast_annotations.exists() and broadcast_images.exists() and any(broadcast_images.glob("*.png")):
        img_count = len(list(broadcast_images.glob("*.png")))
        print(f"\n[OK] Broadcast data already exists ({img_count} images)")
    else:
        print("\n[Step 2] Broadcast data (7.26 GB)")
        print("  Source: Google Drive (yastrebksv/TennisCourtDetector)")

        broadcast_dir.mkdir(parents=True, exist_ok=True)
        zip_path = broadcast_dir / "dataset.zip"

        if not zip_path.exists():
            try:
                import gdown
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
                import gdown

            print("  Downloading from Google Drive (this may take 10-20 minutes)...")
            gdown.download(
                "https://drive.google.com/uc?id=1lhAaeQCmk2y440PmagA0KmIVBIysVMwu",
                str(zip_path),
                quiet=False,
            )

        print("\n  Extracting dataset.zip (8841 images)...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(broadcast_dir))

        # Convert to SHOT format
        print("  Converting to SHOT annotation format...")
        YASTREBKSV_TO_SHOT = {10: "9", 13: "10", 11: "11", 2: "12", 5: "13", 7: "15", 3: "16"}

        all_annotations = []
        for json_file in ["data/data_train.json", "data/data_val.json"]:
            fpath = broadcast_dir / json_file
            if fpath.exists():
                data = json.load(open(fpath))
                for entry in data:
                    kps_raw = entry["kps"]
                    keypoints = {}
                    for yidx, shot_id in YASTREBKSV_TO_SHOT.items():
                        if yidx < len(kps_raw):
                            keypoints[shot_id] = {
                                "x": round(kps_raw[yidx][0] / 1280, 6),
                                "y": round(kps_raw[yidx][1] / 720, 6),
                                "visible": True,
                            }
                    if 2 < len(kps_raw) and 3 < len(kps_raw):
                        keypoints["14"] = {
                            "x": round((kps_raw[2][0] + kps_raw[3][0]) / 2 / 1280, 6),
                            "y": round((kps_raw[2][1] + kps_raw[3][1]) / 2 / 720, 6),
                            "visible": True,
                        }
                    all_annotations.append({"image": entry["id"] + ".png", "keypoints": keypoints})

        json.dump(all_annotations, open(broadcast_annotations, "w"), indent=2)
        print(f"  Converted: {len(all_annotations)} broadcast annotations")

    # ========================================
    # Step 3: Verify
    # ========================================
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)

    yt_ann = json.load(open(youtube_dir / "labeled_annotations.json"))
    yt_frames = len(list(frames_dir.glob("*")))
    print(f"\n  YouTube data:   {len(yt_ann)} annotations, {yt_frames} frames")

    if broadcast_annotations.exists():
        bc_ann = json.load(open(broadcast_annotations))
        bc_imgs = len(list(broadcast_images.glob("*.png"))) if broadcast_images.exists() else 0
        print(f"  Broadcast data: {len(bc_ann)} annotations, {bc_imgs} images")

    print(f"\n  Next step: Run comparison training")
    print(f"  python src/train_compare.py \\")
    print(f"    --broadcast-data data/broadcast/annotations_broadcast.json \\")
    print(f"    --broadcast-images data/broadcast/data/images \\")
    print(f"    --phone-data data/youtube/labeled_annotations.json \\")
    print(f"    --phone-images data/youtube/review/frames \\")
    print(f"    --epochs 100 --batch-size 32")


if __name__ == "__main__":
    main()
