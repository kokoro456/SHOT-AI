"""
Enhanced augmentation pipeline for domain gap reduction.

Key additions vs v1:
- Strong perspective/affine transforms (simulate different camera angles)
- Lens distortion (simulate phone wide-angle)
- Motion blur, defocus blur (simulate phone shake)
- JPEG compression artifacts
- Exposure/white balance shifts
- Color jitter for different court surfaces

These augmentations make broadcast data look more like phone-filmed data,
reducing the domain gap.
"""

import albumentations as A
from albumentations import KeypointParams


def get_strong_augmentation(input_size: int = 256):
    """
    Strong augmentation for domain gap reduction.
    Designed to make broadcast images look more like phone images.
    """
    return A.Compose(
        [
            # === Geometric: simulate phone camera angles ===
            A.Perspective(scale=(0.02, 0.08), p=0.4),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-8, 8),
                shear=(-5, 5),
                p=0.4,
            ),

            # Lens distortion (phone wide-angle simulation)
            A.OpticalDistortion(distort_limit=0.15, p=0.3),

            # Random crop (simulate different framings)
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.3,
            ),

            # === Color: simulate different lighting ===
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.25,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=0.4,
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.15,
                hue=0.05,
                p=0.3,
            ),

            # Exposure/gamma shifts
            A.RandomGamma(gamma_limit=(70, 130), p=0.2),
            A.RandomToneCurve(scale=0.1, p=0.2),

            # === Blur: simulate phone shake/defocus ===
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.Defocus(radius=(2, 4), alias_blur=(0.1, 0.3), p=1.0),
            ], p=0.25),

            # === Noise & Compression: simulate phone camera ===
            A.GaussNoise(std_range=(0.01, 0.03), p=0.15),
            A.ImageCompression(quality_range=(60, 90), p=0.2),

            # Shadow simulation
            A.RandomShadow(
                shadow_roi=(0, 0.2, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.2,
            ),

            # Sun flare (outdoor)
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=80,
                p=0.05,
            ),

            # Final resize
            A.Resize(input_size, input_size),
        ],
        keypoint_params=KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


def get_phone_augmentation(input_size: int = 256):
    """
    Lighter augmentation for phone data (already in target domain).
    Don't over-augment what's already correct.
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=10, val_shift_limit=8, p=0.3),
            A.Rotate(limit=3, p=0.2, border_mode=0),
            A.RandomResizedCrop(
                size=(input_size, input_size),
                scale=(0.92, 1.0),
                ratio=(0.97, 1.03),
                p=0.2,
            ),
            A.GaussNoise(std_range=(0.005, 0.02), p=0.1),
            A.Resize(input_size, input_size),
        ],
        keypoint_params=KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


def get_val_augmentation(input_size: int = 256):
    """Validation: only resize."""
    return A.Compose(
        [A.Resize(input_size, input_size)],
        keypoint_params=KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )
