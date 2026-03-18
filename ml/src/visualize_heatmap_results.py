"""
Heatmap Model Visualization Tool

Loads the trained heatmap model and visualizes predictions on phone images.
Shows:
1. Predicted keypoints (RED circles)
2. Ground truth keypoints (GREEN circles)
3. Per-keypoint pixel error
4. Court lines connecting keypoints

Usage:
    python visualize_heatmap_results.py --model models/heatmap_stage2_best.pth
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_heatmap import HeatmapKeypointModel, create_heatmap_model, NUM_KEYPOINTS

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

KEYPOINT_IDS = [9, 10, 11, 12, 13, 14, 15, 16]
KP_NAMES = ['Pt9(SL)', 'Pt10(SC)', 'Pt11(SR)', 'Pt12(DL)',
            'Pt13(BL)', 'Pt14(BC)', 'Pt15(BR)', 'Pt16(DR)']

# Court line connections
COURT_LINES = [
    (0, 1), (1, 2),       # Service line: 9-10-11
    (4, 5), (5, 6),       # Baseline: 13-14-15
    (3, 4),               # Left doubles: 12-13
    (6, 7),               # Right doubles: 15-16
    (0, 4), (2, 6),       # Left/Right singles sidelines
]


def preprocess_image(image_path, input_size=256):
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((input_size, input_size), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, original


def visualize_single(model, image_path, gt_keypoints=None, gt_visibility=None,
                     output_path=None, device="cpu"):
    """Visualize prediction on a single image."""
    tensor, original = preprocess_image(str(image_path))
    tensor = tensor.to(device)

    with torch.no_grad():
        heatmaps = model(tensor)
        pred_coords = HeatmapKeypointModel.heatmaps_to_coords(heatmaps)

    pred = pred_coords[0].cpu().numpy()  # [8, 2]
    img_w, img_h = original.size

    preview = original.copy()
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.truetype("arial.ttf", max(11, img_h // 45))
        font_small = ImageFont.truetype("arial.ttf", max(9, img_h // 55))
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    radius = max(4, min(img_w, img_h) // 80)
    errors = []

    # Draw court lines (predicted)
    for i, j in COURT_LINES:
        px1 = pred[i, 0] * img_w
        py1 = pred[i, 1] * img_h
        px2 = pred[j, 0] * img_w
        py2 = pred[j, 1] * img_h
        draw.line([(px1, py1), (px2, py2)], fill=(255, 255, 0), width=2)

    for k in range(NUM_KEYPOINTS):
        px = pred[k, 0] * img_w
        py = pred[k, 1] * img_h

        # Draw prediction (RED)
        draw.ellipse([px-radius, py-radius, px+radius, py+radius],
                     fill=(255, 50, 50), outline=(255, 255, 255), width=1)

        label_parts = [f"{KEYPOINT_IDS[k]}"]

        # Draw ground truth (GREEN) if available
        if gt_keypoints is not None and gt_visibility is not None:
            if gt_visibility[k]:
                gx = gt_keypoints[k, 0] * img_w
                gy = gt_keypoints[k, 1] * img_h
                draw.ellipse([gx-radius, gy-radius, gx+radius, gy+radius],
                             fill=(50, 255, 50), outline=(255, 255, 255), width=1)

                # Error line
                draw.line([(px, py), (gx, gy)], fill=(255, 100, 100), width=1)

                # Calculate pixel error (on 256x256 scale)
                err = np.sqrt((pred[k, 0] - gt_keypoints[k, 0])**2 +
                              (pred[k, 1] - gt_keypoints[k, 1])**2) * 256
                errors.append(err)
                label_parts.append(f"{err:.1f}px")
            else:
                errors.append(None)

        label = " ".join(label_parts)
        draw.text((px + radius + 3, py - radius - 2), label, fill=(255, 255, 255), font=font_small)

    # Summary text
    valid_errors = [e for e in errors if e is not None]
    if valid_errors:
        mean_err = np.mean(valid_errors)
        summary = f"Mean Error: {mean_err:.2f}px"
        draw.rectangle([5, 5, 250, 28], fill=(0, 0, 0, 180))
        draw.text((8, 7), summary, fill=(255, 255, 100), font=font)
        draw.text((8, img_h - 22), "RED=Predicted  GREEN=Ground Truth", fill=(255, 255, 255), font=font_small)

    if output_path:
        preview.save(str(output_path), quality=95)

    return preview, errors


def main():
    parser = argparse.ArgumentParser(description="Visualize heatmap model predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to heatmap model checkpoint")
    parser.add_argument("--annotations", type=str,
                        default="data/youtube/labeled_annotations.json",
                        help="Path to annotation file")
    parser.add_argument("--frames", type=str,
                        default="data/youtube/review/frames",
                        help="Path to frame images")
    parser.add_argument("--output", type=str, default="data/youtube/heatmap_preview",
                        help="Output directory for previews")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of random samples to visualize")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load model
    print("Loading heatmap model...")
    model = create_heatmap_model(pretrained=False)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(args.device)

    if "mean_error" in checkpoint:
        print(f"  Checkpoint mean error: {checkpoint['mean_error']:.2f}px")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load annotations
    with open(args.annotations) as f:
        annotations = json.load(f)
    print(f"  Annotations: {len(annotations)} images")

    # Random sample
    samples = random.sample(annotations, min(args.num_samples, len(annotations)))

    # Output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_errors = []

    print(f"\nGenerating {len(samples)} previews...")
    for i, ann in enumerate(samples):
        image_path = os.path.join(args.frames, ann["image"])
        if not os.path.exists(image_path):
            print(f"  [{i+1}] SKIP: {ann['image']} not found")
            continue

        # Parse GT keypoints
        kp_dict = ann["keypoints"]
        gt_kps = np.zeros((8, 2), dtype=np.float32)
        gt_vis = np.zeros(8, dtype=bool)

        for k, kp_id in enumerate(KEYPOINT_IDS):
            kp_key = str(kp_id)
            if kp_key in kp_dict and kp_dict[kp_key].get("visible", True):
                gt_kps[k, 0] = kp_dict[kp_key]["x"]
                gt_kps[k, 1] = kp_dict[kp_key]["y"]
                gt_vis[k] = True

        out_path = output_dir / f"heatmap_{ann['image']}"
        _, errors = visualize_single(model, image_path, gt_kps, gt_vis, out_path, args.device)

        valid = [e for e in errors if e is not None]
        if valid:
            mean_e = np.mean(valid)
            all_errors.extend(valid)
            print(f"  [{i+1}/{len(samples)}] {ann['image']}: {mean_e:.2f}px")

    # Final report
    if all_errors:
        print(f"\n{'='*50}")
        print(f"  VISUALIZATION COMPLETE")
        print(f"{'='*50}")
        print(f"  Images: {len(samples)}")
        print(f"  Mean Error: {np.mean(all_errors):.2f}px")
        print(f"  Median Error: {np.median(all_errors):.2f}px")
        print(f"  Max Error: {np.max(all_errors):.2f}px")
        print(f"  Min Error: {np.min(all_errors):.2f}px")
        print(f"  Output: {output_dir}")
        print(f"\n  Open the output folder to visually inspect results!")


if __name__ == "__main__":
    main()
