"""
Tennis Court Keypoint Dataset

Loads tennis court images with 8 keypoint annotations (points 9-16)
for training the court keypoint detection model.

Annotation format (JSON array):
[
    {
        "image": "filename.png",
        "keypoints": {
            "9":  {"x": 0.3305, "y": 0.9172, "visible": true},
            "10": {"x": 0.6500, "y": 0.9172, "visible": true},
            ...
            "16": {"x": 1.0773, "y": 1.0764, "visible": false}
        }
    },
    ...
]

Note: x, y are normalized to [0, 1] relative to original image dimensions.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from model import NUM_KEYPOINTS

# Keypoint IDs in order (matches model output order)
KEYPOINT_IDS = [9, 10, 11, 12, 13, 14, 15, 16]

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CourtKeypointDataset(Dataset):
    """
    Dataset for tennis court keypoint detection.

    Each item returns:
        image: [3, 256, 256] normalized RGB tensor
        keypoints: [8, 2] normalized (x, y) coordinates (0-1)
        visibility: [8] boolean visibility mask
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        input_size: int = 256,
        augmentation=None,
    ):
        self.image_dir = image_dir
        self.input_size = input_size
        self.augmentation = augmentation

        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ann = self.annotations[idx]

        # Load image
        image_path = os.path.join(self.image_dir, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        orig_h, orig_w = image_np.shape[:2]

        # Parse keypoints (dict format: {"9": {"x":..., "y":..., "visible":...}, ...})
        kp_dict = ann["keypoints"]
        keypoints_xy = []
        visibility = []

        for kp_id in KEYPOINT_IDS:
            kp_key = str(kp_id)
            if kp_key in kp_dict and kp_dict[kp_key]["visible"]:
                # Coordinates are already normalized [0, 1], scale to pixel coords
                keypoints_xy.append([
                    kp_dict[kp_key]["x"] * orig_w,
                    kp_dict[kp_key]["y"] * orig_h,
                ])
                visibility.append(True)
            else:
                keypoints_xy.append([0.0, 0.0])
                visibility.append(False)

        keypoints_np = np.array(keypoints_xy, dtype=np.float32)

        # Apply augmentation (if provided)
        if self.augmentation is not None:
            # Albumentations expects keypoints in (x, y) format
            visible_kps = []
            visible_indices = []
            for i, (kp, vis) in enumerate(zip(keypoints_np, visibility)):
                if vis:
                    visible_kps.append(tuple(kp))
                    visible_indices.append(i)

            augmented = self.augmentation(
                image=image_np,
                keypoints=visible_kps,
            )
            image_np = augmented["image"]
            aug_kps = augmented["keypoints"]

            # Update keypoints after augmentation
            for i, orig_idx in enumerate(visible_indices):
                if i < len(aug_kps):
                    keypoints_np[orig_idx] = np.array(aug_kps[i], dtype=np.float32)
                else:
                    # Keypoint was removed by augmentation (outside image)
                    visibility[orig_idx] = False

            orig_h, orig_w = image_np.shape[:2]

        # Resize image to input_size x input_size
        image_pil = Image.fromarray(image_np)
        image_resized = image_pil.resize((self.input_size, self.input_size), Image.BILINEAR)
        image_np = np.array(image_resized, dtype=np.float32) / 255.0

        # Normalize keypoints to [0, 1] relative to input size
        if orig_w > 0 and orig_h > 0:
            keypoints_norm = keypoints_np.copy()
            keypoints_norm[:, 0] /= orig_w  # x
            keypoints_norm[:, 1] /= orig_h  # y
        else:
            keypoints_norm = np.zeros_like(keypoints_np)

        # Clamp to [0, 1]
        keypoints_norm = np.clip(keypoints_norm, 0.0, 1.0)

        # ImageNet normalization
        image_np = (image_np - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)

        # Convert to tensors: [H, W, C] -> [C, H, W]
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        keypoints_tensor = torch.from_numpy(keypoints_norm).float()
        visibility_tensor = torch.tensor(visibility, dtype=torch.float32)

        return {
            "image": image_tensor,
            "keypoints": keypoints_tensor,
            "visibility": visibility_tensor,
        }


def create_sample_annotation(output_path: str, num_samples: int = 10):
    """Create a sample annotation file for testing the dataset pipeline."""
    annotations = []
    for i in range(num_samples):
        keypoints = {}
        for kp_id in KEYPOINT_IDS:
            keypoints[str(kp_id)] = {
                "x": round(float(np.random.uniform(0.1, 0.9)), 6),
                "y": round(float(np.random.uniform(0.1, 0.9)), 6),
                "visible": True,
            }
        annotations.append({
            "image": f"sample_{i:04d}.jpg",
            "keypoints": keypoints,
        })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Created sample annotation with {num_samples} entries: {output_path}")


if __name__ == "__main__":
    # Create sample annotation for testing
    create_sample_annotation("data/raw/sample_annotations.json")
