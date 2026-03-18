"""
Heatmap-based Court Keypoint Detection Model

Instead of directly regressing (x,y) coordinates, this model predicts
a heatmap for each keypoint. The peak location in each heatmap gives
the keypoint position, preserving spatial information.

Architecture:
  MobileNetV3-Small backbone → upsampling decoder → 8-channel heatmap

Input:  [batch, 3, 256, 256] RGB, ImageNet normalized
Output: [batch, 8, 64, 64] heatmaps (one per keypoint)

The heatmap approach is more accurate than direct regression because:
1. Preserves spatial structure (no information loss from pooling)
2. Naturally handles uncertainty (wide peak = uncertain)
3. Easier to learn (classification-like, not regression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


NUM_KEYPOINTS = 8
HEATMAP_SIZE = 64  # Output heatmap resolution


class HeatmapKeypointModel(nn.Module):
    """
    MobileNetV3-Small + lightweight decoder for heatmap prediction.

    Uses only TFLite-compatible operations for mobile deployment.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v3_small(weights=None)

        self.features = backbone.features  # Output: [batch, 576, 8, 8]

        # Lightweight decoder: upsample 8x8 → 64x64
        self.decoder = nn.Sequential(
            # 8x8 → 16x16
            nn.ConvTranspose2d(576, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x32 → 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final 1x1 conv to get keypoint channels
            nn.Conv2d(64, NUM_KEYPOINTS, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 256, 256]
        Returns:
            heatmaps: [batch, 8, 64, 64]
        """
        features = self.features(x)       # [batch, 576, 8, 8]
        heatmaps = self.decoder(features)  # [batch, 8, 64, 64]
        return heatmaps

    @staticmethod
    def heatmaps_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract (x, y) coordinates from heatmaps using argmax.

        Simple and memory-efficient: finds the peak pixel location
        and normalizes to [0, 1].

        Args:
            heatmaps: [batch, 8, H, W]
        Returns:
            coords: [batch, 8, 2] normalized (x, y) in [0, 1]
        """
        batch_size, num_kp, h, w = heatmaps.shape

        flat = heatmaps.view(batch_size, num_kp, -1)
        max_idx = flat.argmax(dim=2)  # [batch, num_kp]

        x_coords = (max_idx % w).float() / (w - 1)
        y_coords = (max_idx // w).float() / (h - 1)

        coords = torch.stack([x_coords, y_coords], dim=2)
        return coords

    @staticmethod
    def generate_heatmap_targets(keypoints: torch.Tensor, visibility: torch.Tensor,
                                  heatmap_size: int = HEATMAP_SIZE, sigma: float = 2.0) -> torch.Tensor:
        """
        Generate Gaussian heatmap targets from keypoint coordinates.

        Args:
            keypoints: [batch, 8, 2] normalized (x, y) in [0, 1]
            visibility: [batch, 8] visibility mask
            heatmap_size: output heatmap size
            sigma: Gaussian sigma in pixels (on heatmap scale)
        Returns:
            heatmaps: [batch, 8, heatmap_size, heatmap_size]
        """
        batch_size, num_kp, _ = keypoints.shape
        device = keypoints.device

        heatmaps = torch.zeros(batch_size, num_kp, heatmap_size, heatmap_size, device=device)

        # Create coordinate grid
        y_grid = torch.arange(heatmap_size, device=device).float().view(1, 1, heatmap_size, 1)
        x_grid = torch.arange(heatmap_size, device=device).float().view(1, 1, 1, heatmap_size)

        for k in range(num_kp):
            # Scale normalized coords to heatmap pixel coords
            cx = keypoints[:, k, 0:1].unsqueeze(2) * (heatmap_size - 1)  # [batch, 1, 1]
            cy = keypoints[:, k, 1:2].unsqueeze(2) * (heatmap_size - 1)  # [batch, 1, 1]

            cx = cx.unsqueeze(3)  # [batch, 1, 1, 1]
            cy = cy.unsqueeze(3)  # [batch, 1, 1, 1]

            # Gaussian
            gaussian = torch.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
            gaussian = gaussian.squeeze(1)  # [batch, heatmap_size, heatmap_size]

            # Zero out invisible keypoints
            mask = visibility[:, k].view(-1, 1, 1)
            heatmaps[:, k] = gaussian * mask

        return heatmaps


def create_heatmap_model(pretrained: bool = True) -> HeatmapKeypointModel:
    """Factory function."""
    return HeatmapKeypointModel(pretrained=pretrained)


if __name__ == "__main__":
    model = create_heatmap_model(pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    heatmaps = model(x)
    print(f"Input: {x.shape}")
    print(f"Heatmaps: {heatmaps.shape}")

    coords = HeatmapKeypointModel.heatmaps_to_coords(heatmaps)
    print(f"Coords: {coords.shape}")
    print(f"Sample coords: {coords[0]}")

    # Test target generation
    kps = torch.rand(2, 8, 2)
    vis = torch.ones(2, 8)
    targets = HeatmapKeypointModel.generate_heatmap_targets(kps, vis)
    print(f"Targets: {targets.shape}")
    print(f"Target max: {targets.max():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
