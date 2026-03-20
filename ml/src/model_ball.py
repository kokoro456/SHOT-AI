"""
Single-Frame Ball Detector for SHOT Phase 2b

MobileNetV3-Small backbone + lightweight decoder → 48×48 heatmap.
Replaces TrackNet (3-frame, 148ms) with single-frame detection (15-25ms target).
Temporal tracking is handled by Kalman filter in Kotlin, not in the model.

Architecture:
    Input:  [B, 3, 192, 192] single RGB frame (ImageNet normalized)
    Backbone: MobileNetV3-Small (pretrained)
    Decoder: ConvTranspose2d × 3 (6×6 → 12 → 24 → 48)
    Output: [B, 1, 48, 48] heatmap (sigmoid)

Usage:
    python model_ball.py  # Test forward pass and print param count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class BallDetector(nn.Module):
    """
    Single-frame tennis ball detector.

    Input:  [B, 3, 192, 192] ImageNet-normalized RGB
    Output: [B, 1, 48, 48] ball position heatmap (sigmoid)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Backbone: MobileNetV3-Small features
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features  # Output: [B, 576, 6, 6] for 192×192 input

        # Decoder: 6×6 → 48×48
        self.decoder = nn.Sequential(
            # 6×6 → 12×12
            nn.ConvTranspose2d(576, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 12×12 → 24×24
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 24×24 → 48×48
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 1×1 conv → heatmap
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, 192, 192] ImageNet-normalized RGB
        Returns:
            [B, 1, 48, 48] heatmap (sigmoid activated)
        """
        features = self.features(x)   # [B, 576, 6, 6]
        heatmap = self.decoder(features)  # [B, 1, 48, 48]
        return torch.sigmoid(heatmap)


class BallDetectorLoss(nn.Module):
    """
    Focal loss for ball detection.
    Same approach as TrackNet but tuned for single-frame.
    """

    def __init__(self, alpha=0.97, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        class_weight = torch.where(target > 0.5, 1.0, 1.0 - self.alpha)
        loss = focal_weight * class_weight * bce
        return loss.mean()


def generate_heatmap(x, y, size=48, sigma=2.5):
    """
    Generate 2D Gaussian heatmap for 48×48 output.

    Args:
        x, y: ball center in heatmap coordinates (0~47)
        size: heatmap dimension (48)
        sigma: Gaussian std (2.5 for 48×48, smaller than TrackNet's 5.0 for 128×320)
    Returns:
        [size, size] tensor
    """
    if x < 0 or y < 0:
        return torch.zeros(size, size)

    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing='ij'
    )
    heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap


def extract_ball_position(heatmap, threshold=0.5):
    """
    Extract ball (x, y) from predicted heatmap using argmax.

    Args:
        heatmap: [H, W] predicted heatmap
        threshold: minimum peak value
    Returns:
        (x, y, confidence) or (None, None, 0) if no ball
    """
    max_val = heatmap.max().item()
    if max_val < threshold:
        return None, None, 0.0

    max_idx = heatmap.argmax()
    h, w = heatmap.shape
    y = (max_idx // w).item()
    x = (max_idx % w).item()
    return x, y, max_val


if __name__ == "__main__":
    model = BallDetector(pretrained=False)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.features.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"Total parameters:    {total_params:,}")
    print(f"  Backbone (frozen):  {backbone_params:,}")
    print(f"  Decoder (trainable): {decoder_params:,}")
    print(f"Model size (FP32):   {total_params * 4 / 1024 / 1024:.1f} MB")

    # Forward pass test
    x = torch.randn(1, 3, 192, 192)
    y = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # Heatmap test
    hm = generate_heatmap(24, 24, size=48, sigma=2.5)
    print(f"\nHeatmap shape: {hm.shape}, max: {hm.max():.4f}")
    bx, by, conf = extract_ball_position(hm)
    print(f"Extracted: ({bx}, {by}) conf={conf:.4f}")
