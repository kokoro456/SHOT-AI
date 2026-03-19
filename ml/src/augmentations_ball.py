"""
Ball tracking augmentations.

3프레임 연속 입력에 일관된 변환을 적용한다.
기하 변환(회전, 크롭)은 프레임 간 공 위치 관계를 깨뜨리므로 제한적으로 적용.
"""

import numpy as np


class BallAugmentor:
    """3프레임 triplet에 일관된 어그멘테이션을 적용."""

    def __init__(self, mode="train"):
        self.mode = mode

    def __call__(self, frames: np.ndarray, heatmap: np.ndarray):
        """
        Args:
            frames: [9, H, W] float32 (3 RGB frames concatenated)
            heatmap: [H, W] float32 target heatmap
        Returns:
            augmented (frames, heatmap) as numpy arrays
        """
        if self.mode != "train":
            return frames, heatmap

        # 1. Random brightness (same for all 3 frames)
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            frames = np.clip(frames * brightness, 0, 1)

        # 2. Color shift (same per-channel across all 3 frames)
        if np.random.random() < 0.3:
            channel_shift = np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
            for i in range(3):
                for c in range(3):
                    frames[i * 3 + c] += channel_shift[c]
            frames = np.clip(frames, 0, 1)

        # 3. Gaussian noise (independent per frame, simulates sensor noise)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, frames.shape).astype(np.float32)
            frames = np.clip(frames + noise, 0, 1)

        # 4. Horizontal flip (flip both frames AND heatmap)
        if np.random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()  # flip W axis [9, H, W]
            heatmap = np.flip(heatmap, axis=1).copy()  # flip W axis [H, W]

        return frames, heatmap
