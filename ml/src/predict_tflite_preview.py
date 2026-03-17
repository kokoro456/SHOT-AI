"""
TFLite Keypoint Prediction & Preview Generator

Same as predict_and_preview.py but uses the TFLite model directly
(when the PyTorch .pth checkpoint is not available).

Uses the deployed TFLite model at:
  court-detection/src/main/assets/court_keypoint.tflite

Usage:
    python predict_tflite_preview.py \
        --frames data/youtube/frames \
        --output data/youtube/preview
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: tensorflow is required. Install with: pip install tensorflow")
    sys.exit(1)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_KEYPOINTS = 8
KEYPOINT_IDS = [9, 10, 11, 12, 13, 14, 15, 16]

# Default TFLite model path (relative to project root)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "court-detection", "src", "main", "assets", "court_keypoint.tflite"
)


def load_tflite_model(model_path: str):
    """Load TFLite model and return interpreter."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image_path: str, input_size: int = 256) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocess image for TFLite inference.

    TFLite model expects: [1, 256, 256, 3] NHWC float32, ImageNet normalized.
    """
    original = Image.open(image_path).convert("RGB")

    resized = original.resize((input_size, input_size), Image.BILINEAR)
    img_np = np.array(resized, dtype=np.float32) / 255.0

    # ImageNet normalization
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # NHWC format: [1, H, W, C]
    tensor = np.expand_dims(img_np, axis=0).astype(np.float32)

    return tensor, original


def predict_keypoints(interpreter, tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run TFLite inference.

    Output: [1, 24] -> 8 keypoints x (x, y, confidence), all sigmoid-activated [0,1]

    Returns:
        coords: [8, 2] normalized (x, y)
        confidence: [8] confidence scores
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])  # [1, 24]
    output = output[0]  # [24]

    # Reshape to [8, 3] -> (x, y, confidence)
    reshaped = output.reshape(NUM_KEYPOINTS, 3)
    coords = reshaped[:, :2]  # [8, 2]
    confidence = reshaped[:, 2]  # [8]

    return coords, confidence


def draw_preview(
    original: Image.Image,
    coords: np.ndarray,
    confidence: np.ndarray,
    output_path: str,
):
    """Draw keypoints on the original image and save as preview."""
    img_w, img_h = original.size

    preview = original.copy()
    draw = ImageDraw.Draw(preview)

    try:
        font = ImageFont.truetype("arial.ttf", max(12, img_h // 40))
    except (IOError, OSError):
        font = ImageFont.load_default()

    radius = max(4, min(img_w, img_h) // 80)

    for i in range(NUM_KEYPOINTS):
        x = coords[i, 0] * img_w
        y = coords[i, 1] * img_h
        conf = confidence[i]

        if conf > 0.7:
            color = (0, 255, 0)
        elif conf > 0.4:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color, outline=(255, 255, 255), width=1
        )

        label = f"{KEYPOINT_IDS[i]}({conf:.2f})"
        draw.text((x + radius + 2, y - radius), label, fill=color, font=font)

    # Draw court lines (service line and baseline connections)
    line_pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6), (6, 7), (0, 4), (2, 6)]
    for i, j in line_pairs:
        if confidence[i] > 0.4 and confidence[j] > 0.4:
            x1, y1 = coords[i, 0] * img_w, coords[i, 1] * img_h
            x2, y2 = coords[j, 0] * img_w, coords[j, 1] * img_h
            draw.line([(x1, y1), (x2, y2)], fill=(0, 200, 0), width=2)

    preview.save(output_path, quality=90)


def main():
    parser = argparse.ArgumentParser(description="TFLite keypoint prediction + preview")
    parser.add_argument("--frames", type=str, default="data/youtube/frames",
                        help="Directory containing extracted frames")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to TFLite model")
    parser.add_argument("--output", type=str, default="data/youtube/preview",
                        help="Output directory for preview images")
    parser.add_argument("--predictions-output", type=str, default="data/youtube/predictions.json",
                        help="Output JSON for predictions")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = os.path.normpath(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: TFLite model not found: {model_path}")
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_files = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not frame_files:
        print(f"ERROR: No images found in {frames_dir}")
        sys.exit(1)

    print(f"\n=== TFLite Keypoint Prediction & Preview ===")
    print(f"Frames: {len(frame_files)} images")
    print(f"Model: {model_path}")
    print(f"Preview output: {output_dir}")
    print()

    interpreter = load_tflite_model(model_path)
    print("TFLite model loaded.\n")

    predictions = []

    for i, frame_path in enumerate(frame_files):
        print(f"[{i+1}/{len(frame_files)}] {frame_path.name}...", end=" ")

        tensor, original = preprocess_image(str(frame_path))
        coords, confidence = predict_keypoints(interpreter, tensor)

        preview_path = output_dir / f"preview_{frame_path.stem}.jpg"
        draw_preview(original, coords, confidence, str(preview_path))

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
            "status": "pending_review",
        })

        avg_conf = confidence.mean()
        print(f"avg_conf={avg_conf:.3f}")

    pred_path = Path(args.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Previews: {output_dir}")
    print(f"Predictions: {pred_path}")
    print(f"\nReview the preview images, then run:")
    print(f"  python review_data.py --predictions {pred_path} --preview-dir {output_dir}")


if __name__ == "__main__":
    main()
