"""
BallDetector → ONNX conversion.

Usage:
    python export_ball_onnx.py --checkpoint models/ball_best.pth \
                               --output models/ball_detector.onnx
    python export_ball_onnx.py --dummy  # Test with untrained model
"""

import argparse
import os
import numpy as np
import torch

from model_ball import BallDetector


def export_to_onnx(model, output_path, input_size=192):
    """Export BallDetector to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes=None,  # Fixed input size for mobile
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Exported to: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")


def validate_onnx(onnx_path, input_size=192):
    """Validate ONNX model with ORT."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    result = session.run(None, {"input": dummy})

    heatmap = result[0]
    print(f"ONNX output shape: {heatmap.shape}")
    print(f"ONNX output range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    expected_hm_size = input_size // 4
    assert heatmap.shape == (1, 1, expected_hm_size, expected_hm_size), \
        f"Expected (1, 1, {expected_hm_size}, {expected_hm_size}), got {heatmap.shape}"
    print(f"[OK] Output shape verified: {heatmap.shape}")
    return heatmap


def compare_pytorch_onnx(model, onnx_path, input_size=192):
    """Verify PyTorch and ONNX outputs match."""
    import onnxruntime as ort

    model.eval()
    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy)).numpy()

    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {"input": dummy})[0]

    diff = np.abs(pt_out - ort_out)
    print(f"Max diff: {diff.max():.8f}")
    print(f"Mean diff: {diff.mean():.8f}")

    if diff.max() < 1e-5:
        print("[OK] PyTorch <-> ONNX: MATCH")
    else:
        print("[WARN] PyTorch <-> ONNX: slight difference (likely numerical)")


def main():
    parser = argparse.ArgumentParser(description="Export BallDetector to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint .pth")
    parser.add_argument("--output", type=str, default="models/ball_detector.onnx")
    parser.add_argument("--dummy", action="store_true", help="Test with untrained model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = BallDetector(pretrained=False)

    if args.dummy:
        print("Using DUMMY (untrained) model for pipeline test")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("ERROR: --checkpoint or --dummy required")
        return

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB FP32)")

    export_to_onnx(model, args.output)
    validate_onnx(args.output)
    compare_pytorch_onnx(model, args.output)

    print(f"\n[OK] Export complete: {args.output}")


if __name__ == "__main__":
    main()
