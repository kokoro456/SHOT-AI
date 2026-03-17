"""
Court Keypoint Detection Model

MobileNetV3-Small backbone + keypoint regression head.
Detects 8 keypoints (points 9-16) on the near side of a tennis court.

Input: [batch, 3, 256, 256] RGB, ImageNet normalized
Output: [batch, 24] - 8 keypoints x (x, y, confidence)
  - x, y: normalized [0, 1] coordinates
  - confidence: sigmoid-activated probability of keypoint visibility
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


NUM_KEYPOINTS = 8  # Points 9-16 (near court visible keypoints)
OUTPUT_DIM = NUM_KEYPOINTS * 3  # x, y, confidence per keypoint


class CourtKeypointModel(nn.Module):
    """
    MobileNetV3-Small based keypoint regression model.

    Uses only TFLite-compatible operations:
    Conv2d, BatchNorm2d, ReLU, HardSwish, Linear, AdaptiveAvgPool2d
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load MobileNetV3-Small backbone
        if pretrained:
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v3_small(weights=None)

        # Use feature extractor (remove classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Regression head
        # MobileNetV3-Small last channel: 576
        self.head = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 256, 256] RGB tensor, ImageNet normalized

        Returns:
            [batch, 24] tensor with 8 keypoints x (x, y, confidence)
            - x, y are sigmoid-activated (normalized 0-1)
            - confidence is sigmoid-activated (0-1 probability)
        """
        # Feature extraction
        features = self.features(x)  # [batch, 576, 8, 8]
        pooled = self.pool(features)  # [batch, 576, 1, 1]
        flat = pooled.flatten(1)  # [batch, 576]

        # Regression
        raw = self.head(flat)  # [batch, 24]

        # Apply sigmoid to all outputs (x, y coords and confidence)
        output = torch.sigmoid(raw)

        return output

    @staticmethod
    def parse_output(output: torch.Tensor):
        """
        Parse model output into structured keypoint data.

        Args:
            output: [batch, 24] tensor

        Returns:
            coords: [batch, 8, 2] - normalized (x, y) coordinates
            confidence: [batch, 8] - visibility confidence scores
        """
        batch_size = output.shape[0]
        reshaped = output.view(batch_size, NUM_KEYPOINTS, 3)
        coords = reshaped[:, :, :2]  # [batch, 8, 2]
        confidence = reshaped[:, :, 2]  # [batch, 8]
        return coords, confidence


def create_model(pretrained: bool = True) -> CourtKeypointModel:
    """Factory function to create the model."""
    return CourtKeypointModel(pretrained=pretrained)


if __name__ == "__main__":
    # Quick test
    model = create_model(pretrained=False)
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    coords, confidence = CourtKeypointModel.parse_output(output)
    print(f"Coords shape: {coords.shape}")
    print(f"Confidence shape: {confidence.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
