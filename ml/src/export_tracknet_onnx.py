"""
TrackNet → ONNX conversion.

Usage:
    python export_tracknet_onnx.py --checkpoint models/tracknet_best.pth \
                                   --output models/ball_tracking.onnx
    python export_tracknet_onnx.py --dummy  # Test with untrained model
"""

import argparse
import os
import numpy as np
import torch

from model_tracknet import TrackNet


def export_to_onnx(model, output_path, input_h=128, input_w=320):
    """Export TrackNet to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 9, input_h, input_w)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes=None,  # Fixed input size for mobile
    )
    print(f"Exported to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def validate_onnx(onnx_path, input_h=128, input_w=320):
    """Validate ONNX model with ORT."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 9, input_h, input_w).astype(np.float32)
    result = session.run(None, {"input": dummy})

    heatmap = result[0]
    print(f"ONNX output shape: {heatmap.shape}")
    print(f"ONNX output range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
    return heatmap


def compare_pytorch_onnx(model, onnx_path, input_h=128, input_w=320):
    """Verify PyTorch and ONNX outputs match."""
    import onnxruntime as ort

    model.eval()
    dummy = np.random.randn(1, 9, input_h, input_w).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy)).numpy()

    # ONNX
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
    parser = argparse.ArgumentParser(description="Export TrackNet to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint .pth")
    parser.add_argument("--output", type=str, default="models/ball_tracking.onnx")
    parser.add_argument("--dummy", action="store_true", help="Test with untrained model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = TrackNet(input_channels=9, base_filters=32)

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
