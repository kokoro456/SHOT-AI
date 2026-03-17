"""
Data augmentation pipeline for tennis court keypoint detection.

Uses Albumentations with keypoint-aware transforms.
Augmentation probabilities are tuned per the spec to prevent
overfitting to specific lighting/surface conditions.
"""

import albumentations as A
from albumentations import KeypointParams


def get_train_augmentation(input_size: int = 256):
    """
    Training augmentation pipeline.

    Probabilities per spec:
    - Brightness ±20% (p=0.5)
    - Contrast ±15% (p=0.4)
    - Hue ±10° (p=0.3)
    - Rotation ±5° (p=0.2)
    - Crop 5-10% (p=0.3)
    - Synthetic shadow (p=0.2) - approximated with RandomShadow
    - Lens distortion (p=0.3)
    """
    return A.Compose(
        [
            # Color augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.15,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3,
            ),

            # Geometric augmentations
            A.Rotate(limit=5, p=0.2, border_mode=0),
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
                p=0.3,
            ),

            # Lens distortion simulation
            A.OpticalDistortion(
                distort_limit=0.1,
                shift_limit=0.05,
                p=0.3,
            ),

            # Shadow simulation
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2,
            ),

            # Noise
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.1),

            # Final resize (if not already done by RandomResizedCrop)
            A.Resize(input_size, input_size),
        ],
        keypoint_params=KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


def get_val_augmentation(input_size: int = 256):
    """Validation augmentation: only resize, no random transforms."""
    return A.Compose(
        [A.Resize(input_size, input_size)],
        keypoint_params=KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )
