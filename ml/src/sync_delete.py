"""
Sync deletions from previews folder to frames folder.

After you delete unwanted images from the previews/ folder,
run this script to automatically delete the corresponding
files from frames/.

Usage:
    python sync_delete.py --review-dir data/youtube/review
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Sync preview deletions to frames")
    parser.add_argument("--review-dir", type=str, default="data/youtube/review")
    args = parser.parse_args()

    review_dir = Path(args.review_dir)
    frames_dir = review_dir / "frames"
    previews_dir = review_dir / "previews"

    # Get surviving previews (strip "preview_" prefix to get frame name)
    surviving = set()
    for p in previews_dir.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            frame_name = p.name.replace("preview_", "", 1)
            surviving.add(frame_name)

    # Find frames to delete
    to_delete = []
    for f in frames_dir.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if f.name not in surviving:
                to_delete.append(f)

    if not to_delete:
        print("No frames to delete. Folders are already in sync.")
        return

    print(f"Will delete {len(to_delete)} frames that have no matching preview:\n")
    for f in sorted(to_delete)[:20]:
        print(f"  {f.name}")
    if len(to_delete) > 20:
        print(f"  ... and {len(to_delete) - 20} more")

    confirm = input(f"\nDelete {len(to_delete)} files? (y/n): ").strip().lower()
    if confirm == "y":
        for f in to_delete:
            f.unlink()
        print(f"Deleted {len(to_delete)} frames.")
    else:
        print("Cancelled.")

    # Summary
    remaining = len(list(frames_dir.glob("*.jpg")))
    print(f"\nRemaining: {remaining} frames, {len(surviving)} previews")


if __name__ == "__main__":
    main()
