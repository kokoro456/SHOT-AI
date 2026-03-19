"""
TrackNet - Tennis Ball Tracking Model

Based on TrackNet paper (arXiv:1907.03698) adapted for mobile deployment.

Architecture: Encoder-Decoder with VGG-style conv blocks
Input: 3 consecutive frames concatenated → [B, 9, H, W]
Output: Heatmap of ball position → [B, 1, H, W]

Key design decisions for SHOT:
- Input resolution: 320x128 (wider for tennis court landscape)
- Lightweight variant: fewer filters for mobile TFLite deployment
- Single-channel output heatmap (not 256-class classification)
- Sub-pixel coordinate extraction via weighted average

References:
- https://github.com/yastrebksv/TrackNet
- https://arxiv.org/abs/1907.03698
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TrackNet(nn.Module):
    """
    TrackNet for tennis ball detection.

    Input: [B, 9, H, W] - 3 consecutive RGB frames concatenated
    Output: [B, 1, H, W] - ball position heatmap (sigmoid activated)
    """
    def __init__(self, input_channels=9, base_filters=32):
        super().__init__()
        f = base_filters  # 32 for mobile, 64 for full

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBlock(input_channels, f),
            ConvBlock(f, f),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            ConvBlock(f, f * 2),
            ConvBlock(f * 2, f * 2),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            ConvBlock(f * 2, f * 4),
            ConvBlock(f * 4, f * 4),
            ConvBlock(f * 4, f * 4),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(f * 4, f * 8),
            ConvBlock(f * 8, f * 8),
        )

        # Decoder (with skip connections)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = nn.Sequential(
            ConvBlock(f * 8 + f * 4, f * 4),
            ConvBlock(f * 4, f * 4),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = nn.Sequential(
            ConvBlock(f * 4 + f * 2, f * 2),
            ConvBlock(f * 2, f * 2),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            ConvBlock(f * 2 + f, f),
            ConvBlock(f, f),
        )

        # Output: single channel heatmap
        self.out_conv = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # [B, f, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 2f, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))  # [B, 4f, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))  # [B, 8f, H/8, W/8]

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)  # [B, 4f, H/4, W/4]

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)  # [B, 2f, H/2, W/2]

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)  # [B, f, H, W]

        # Output heatmap
        out = torch.sigmoid(self.out_conv(d1))  # [B, 1, H, W]
        return out


class TrackNetLoss(nn.Module):
    """
    Combined loss for ball tracking:
    - Focal loss for handling class imbalance (ball is tiny vs background)
    - Weighted BCE to focus on ball region
    """
    def __init__(self, alpha=0.97, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # weight for negative class (background >> ball)
        self.gamma = gamma

    def forward(self, pred, target):
        # Focal loss
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Class weight: more weight on ball pixels
        class_weight = torch.where(target > 0.5, 1.0, 1.0 - self.alpha)
        loss = focal_weight * class_weight * bce
        return loss.mean()


def generate_heatmap(x, y, width, height, sigma=5.0):
    """
    Generate 2D Gaussian heatmap centered at (x, y).

    Args:
        x, y: ball center coordinates (pixels)
        width, height: heatmap dimensions
        sigma: Gaussian standard deviation

    Returns:
        [H, W] tensor with Gaussian peak at (x, y)
    """
    if x < 0 or y < 0:
        return torch.zeros(height, width)

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap


def extract_ball_position(heatmap, threshold=0.5):
    """
    Extract ball (x, y) from predicted heatmap using weighted average.

    Args:
        heatmap: [H, W] predicted heatmap
        threshold: minimum peak value to consider as ball detected

    Returns:
        (x, y, confidence) or (None, None, 0) if no ball detected
    """
    max_val = heatmap.max().item()
    if max_val < threshold:
        return None, None, 0.0

    # Mask low values
    mask = (heatmap > threshold * max_val).float()
    masked = heatmap * mask

    # Weighted average for sub-pixel accuracy
    total = masked.sum()
    if total < 1e-6:
        return None, None, 0.0

    h, w = heatmap.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij'
    )

    x = (masked * xx).sum() / total
    y = (masked * yy).sum() / total

    return x.item(), y.item(), max_val


if __name__ == "__main__":
    # Test model
    model = TrackNet(input_channels=9, base_filters=32)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # Test forward pass
    x = torch.randn(1, 9, 128, 320)  # 3 frames, landscape
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # Test heatmap generation
    hm = generate_heatmap(160, 64, 320, 128, sigma=5)
    print(f"Heatmap shape: {hm.shape}, max: {hm.max():.4f}")

    # Test ball extraction
    bx, by, conf = extract_ball_position(hm)
    print(f"Extracted ball: ({bx:.1f}, {by:.1f}) conf={conf:.4f}")
