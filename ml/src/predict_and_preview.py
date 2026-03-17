"""
Keypoint Prediction & Preview Generator

Runs the trained model on extracted frames to:
1. Generate initial keypoint predictions (for semi-auto labeling)
2. Create visual previews with keypoints overlaid (for human review)

This script creates a preview directory where each image shows:
- The original frame with predicted keypoints drawn as colored circles
- Keypoint confidence scores displayed next to each point
- Green (>0.7), Yellow (0.4-0.7), Red (<0.4) color coding

Requires: torch, torchvision, Pillow, numpy

Usage:
    python predict_and_preview.py \
        --frames data/youtube/frames \
        --model models/best_model.pth \
        --output data/youtube/preview
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add parent path for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CourtKeypointModel, NUM_KEYPOINTS

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Keypoint labels for display
KEYPOINT_LABELS = {
    0: "Pt9 (서비스L)",
    1: "Pt10 (서비스C)",
    2: "Pt11 (서비스R)",
    3: "Pt12 (베이스DL)",
    4: "Pt13 (베이스SL)",
    5: "Pt14 (베이스C)",
    6: "Pt15 (베이스SR)",
    7: "Pt16 (베이스DR)",
}

# SHOT keypoint IDs in order
KEYPOINT_IDS = [9, 10, 11, 12, 13, 14, 15, 16]


def load_model(model_path: str, device: str = "cpu") -> CourtKeypointModel:
    """Load trained model from checkpoint."""
    model = CourtKeypointModel(pretrained=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model


def preprocess_image(image_path: str, input_size: int = 256) -> Tuple[torch.Tensor, Image.Image]:
    """
    Preprocess image for model inference.

    Returns:
        tensor: [1, 3, 256, 256] normalized tensor
        original: Original PIL image (for drawing)
    """
    original = Image.open(image_path).convert("RGB")

    # Resize to model input size
    resized = original.resize((input_size, input_size), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0

    # ImageNet normalization
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # [H, W, C] -> [1, C, H, W]
    tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)

    return tensor, original


def predict_keypoints(model: CourtKeypointModel, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference and return keypoint predictions.

    Returns:
        coords: [8, 2] normalized (x, y) coordinates
        confidence: [8] confidence scores
    """
    with torch.no_grad():
        output = model(tensor)
        coords, confidence = CourtKeypointModel.parse_output(output)

    return coords[0].numpy(), confidence[0].numpy()


def draw_preview(
    original: Image.Image,
    coords: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
):
    """
    Draw keypoints on the original image and save as preview.

    Color coding:
    - Green: confidence > 0.7 (reliable)
    - Yellow: confidence 0.4-0.7 (uncertain)
    - Red: confidence < 0.4 (unreliable)
    """
    img_w, img_h = original.size

    # Create a copy for drawing
    preview = original.copy()
    draw = ImageDraw.Draw(preview)

    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", max(12, img_h // 40))
    except (IOError, OSError):
        font = ImageFont.load_default()

    radius = max(4, min(img_w, img_h) // 80)

    for i in range(NUM_KEYPOINTS):
        x = coords[i, 0] * img_w
        y = coords[i, 1] * img_h
        conf = confidence[i]

        # Color by confidence
        if conf > 0.7:
            color = (0, 255, 0)  # Green
        elif conf > 0.4:
            color = (255, 255, 0)  # Yellow
        else:
            color = (255, 0, 0)  # Red

        # Draw circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color, outline=(255, 255, 255), width=1
        )

        # Draw label
        label = f"{KEYPOINT_IDS[i]}({conf:.2f})"
        draw.text((x + radius + 2, y - radius), label, fill=color, font=font)

    # Draw connections (service line and baseline)
    # Service line: pt9 -> pt10 -> pt11
    for pairs in [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (6, 7), (0, 4), (2, 6)]:
        i, j = pairs
        if confidence[i] > 0.4 and confidence[j] > 0.4:
            x1, y1 = coords[i, 0] * img_w, coords[i, 1] * img_h
            x2, y2 = coords[j, 0] * img_w, coords[j, 1] * img_h
            draw.line([(x1, y1), (x2, y2)], fill=(0, 200, 0, 128), width=2)

    preview.save(output_path, quality=90)


def main():
    parser = argparse.ArgumentParser(description="Generate keypoint predictions and visual previews")
    parser.add_argument("--frames", type=str, default="data/youtube/frames",
                        help="Directory containing extracted frames")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="data/youtube/preview",
                        help="Output directory for preview images")
    parser.add_argument("--predictions-output", type=str, default="data/youtube/predictions.json",
                        help="Output JSON file for keypoint predictions (initial labels)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu/cuda)")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check model exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("Please provide the trained model checkpoint path.")
        sys.exit(1)

    # Find all frame images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not frame_files:
        print(f"ERROR: No images found in {frames_dir}")
        sys.exit(1)

    print(f"\n=== Keypoint Prediction & Preview ===")
    print(f"Frames: {len(frame_files)} images in {frames_dir}")
    print(f"Model: {args.model}")
    print(f"Preview output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    model = load_model(args.model, args.device)
    print("Model loaded.\n")

    # Process each frame
    predictions = []

    for i, frame_path in enumerate(frame_files):
        print(f"[{i+1}/{len(frame_files)}] {frame_path.name}...", end=" ")

        # Preprocess
        tensor, original = preprocess_image(str(frame_path))
        tensor = tensor.to(args.device)

        # Predict
        coords, confidence = predict_keypoints(model, tensor)

        # Save preview
        preview_path = output_dir / f"preview_{frame_path.stem}.jpg"
        draw_preview(original, coords, confidence, str(preview_path))

        # Build prediction entry (SHOT annotation format for review)
        keypoints = {}
        for j, kp_id in enumerate(KEYPOINT_IDS):
            keypoints[str(kp_id)] = {
                "x": round(float(coords[j, 0]), 6),
                "y": round(float(coords[j, 1]), 6),
                "visible": bool(confidence[j] > 0.4),
                "confidence": round(float(confidence[j]), 4),
            }

        predictions.append({
            "image": frame_path.name,
            "source": "youtube",
            "keypoints": keypoints,
            "status": "pending_review",  # Will be updated by review tool
        })

        avg_conf = confidence.mean()
        print(f"avg_conf={avg_conf:.3f}")

    # Save predictions
    pred_path = Path(args.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Previews saved to: {output_dir}")
    print(f"Predictions saved to: {pred_path}")
    print(f"\nNext step: Review previews in {output_dir}/")
    print(f"  - Check if keypoints align with actual court lines")
    print(f"  - Run review_data.py to approve/reject/correct each image")


if __name__ == "__main__":
    main()
