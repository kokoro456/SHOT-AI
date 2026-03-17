"""
Training Data Review Tool

Interactive tool for reviewing extracted frames and model predictions.
Allows the user to:
- View each image with predicted keypoints
- Approve (a): Accept image + predictions as-is for training
- Reject (r): Discard image from training set
- Skip (s): Skip for now, review later
- Quit (q): Save progress and exit

After review, generates a clean annotation file containing ONLY approved data,
ready for model fine-tuning.

Requires: Pillow, numpy

Usage:
    # Interactive review with image viewer
    python review_data.py \
        --predictions data/youtube/predictions.json \
        --preview-dir data/youtube/preview \
        --output data/youtube/approved_annotations.json

    # Generate summary report of current review status
    python review_data.py \
        --predictions data/youtube/predictions.json \
        --report
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def open_image(image_path: str):
    """Open an image with the system default viewer."""
    if sys.platform == "win32":
        os.startfile(image_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", image_path])
    else:
        subprocess.run(["xdg-open", image_path])


def print_keypoint_summary(entry: Dict):
    """Print keypoint predictions for an entry."""
    kps = entry["keypoints"]
    print(f"  {'ID':>4} {'X':>8} {'Y':>8} {'Conf':>8} {'Visible':>8}")
    print(f"  {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for kp_id in ["9", "10", "11", "12", "13", "14", "15", "16"]:
        kp = kps[kp_id]
        conf = kp.get("confidence", 0)
        vis = "✓" if kp["visible"] else "✗"
        conf_indicator = "●" if conf > 0.7 else ("◐" if conf > 0.4 else "○")
        print(f"  {kp_id:>4} {kp['x']:>8.4f} {kp['y']:>8.4f} {conf:>7.3f}{conf_indicator} {vis:>8}")


def review_interactive(predictions: List[Dict], preview_dir: str) -> List[Dict]:
    """
    Interactive review loop.

    Opens preview images one by one and asks for user decision.
    """
    pending = [p for p in predictions if p.get("status") == "pending_review"]
    total = len(predictions)
    pending_count = len(pending)
    approved_count = len([p for p in predictions if p.get("status") == "approved"])
    rejected_count = len([p for p in predictions if p.get("status") == "rejected"])

    print(f"\n=== Data Review Tool ===")
    print(f"Total images: {total}")
    print(f"  Pending:  {pending_count}")
    print(f"  Approved: {approved_count}")
    print(f"  Rejected: {rejected_count}")
    print()
    print("Commands:")
    print("  a = Approve (use for training)")
    print("  r = Reject (discard)")
    print("  s = Skip (review later)")
    print("  q = Save & quit")
    print()

    if not pending:
        print("No pending images to review.")
        return predictions

    reviewed_count = 0

    for i, entry in enumerate(pending):
        image_name = entry["image"]
        preview_name = f"preview_{Path(image_name).stem}.jpg"
        preview_path = os.path.join(preview_dir, preview_name)

        print(f"\n[{i+1}/{pending_count}] {image_name}")

        # Show keypoint summary
        print_keypoint_summary(entry)

        # Open preview image
        if os.path.exists(preview_path):
            open_image(preview_path)
        else:
            print(f"  WARNING: Preview not found: {preview_path}")

        # Get user decision
        while True:
            choice = input("\n  Decision (a/r/s/q): ").strip().lower()
            if choice in ("a", "r", "s", "q"):
                break
            print("  Invalid input. Use: a(approve), r(reject), s(skip), q(quit)")

        if choice == "a":
            entry["status"] = "approved"
            reviewed_count += 1
            print("  → APPROVED")
        elif choice == "r":
            entry["status"] = "rejected"
            reviewed_count += 1
            print("  → REJECTED")
        elif choice == "s":
            print("  → SKIPPED")
        elif choice == "q":
            print(f"\n  Saving progress... ({reviewed_count} reviewed this session)")
            break

    return predictions


def generate_approved_annotations(predictions: List[Dict], output_path: str):
    """
    Generate clean annotation file from approved entries only.

    Output format matches SHOT training annotation format (same as dataset.py expects).
    """
    approved = [p for p in predictions if p.get("status") == "approved"]

    # Convert to training annotation format (strip confidence, keep only x/y/visible)
    annotations = []
    for entry in approved:
        clean_kps = {}
        for kp_id, kp_data in entry["keypoints"].items():
            clean_kps[kp_id] = {
                "x": kp_data["x"],
                "y": kp_data["y"],
                "visible": kp_data["visible"],
            }

        annotations.append({
            "image": entry["image"],
            "keypoints": clean_kps,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"\nApproved annotations: {len(annotations)} images")
    print(f"Saved to: {output_path}")
    return annotations


def print_report(predictions: List[Dict]):
    """Print review status report."""
    statuses = {}
    for p in predictions:
        status = p.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1

    total = len(predictions)
    print(f"\n=== Review Status Report ===")
    print(f"Total images: {total}")
    for status, count in sorted(statuses.items()):
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {status:>15}: {count:>4} ({pct:5.1f}%) {bar}")

    # Confidence stats for approved
    approved = [p for p in predictions if p.get("status") == "approved"]
    if approved:
        confs = []
        for p in approved:
            for kp in p["keypoints"].values():
                if "confidence" in kp:
                    confs.append(kp["confidence"])
        if confs:
            print(f"\n  Approved keypoint confidence:")
            print(f"    Mean: {sum(confs)/len(confs):.3f}")
            print(f"    Min:  {min(confs):.3f}")
            print(f"    Max:  {max(confs):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Review extracted frames and predictions")
    parser.add_argument("--predictions", type=str, default="data/youtube/predictions.json",
                        help="Predictions JSON file (from predict_and_preview.py)")
    parser.add_argument("--preview-dir", type=str, default="data/youtube/preview",
                        help="Directory containing preview images")
    parser.add_argument("--output", type=str, default="data/youtube/approved_annotations.json",
                        help="Output file for approved annotations (training-ready)")
    parser.add_argument("--report", action="store_true",
                        help="Print review status report and exit")
    args = parser.parse_args()

    # Load predictions
    if not os.path.exists(args.predictions):
        print(f"ERROR: Predictions file not found: {args.predictions}")
        print("Run predict_and_preview.py first.")
        sys.exit(1)

    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if args.report:
        print_report(predictions)
        return

    # Interactive review
    predictions = review_interactive(predictions, args.preview_dir)

    # Save updated predictions (with review status)
    with open(args.predictions, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Review progress saved to: {args.predictions}")

    # Generate approved annotations
    generate_approved_annotations(predictions, args.output)

    print(f"\nNext step:")
    print(f"  1. Continue reviewing: python review_data.py --predictions {args.predictions}")
    print(f"  2. Check status: python review_data.py --predictions {args.predictions} --report")
    print(f"  3. Train with approved data: python train.py --data {args.output} --image-dir data/youtube/frames")


if __name__ == "__main__":
    main()
