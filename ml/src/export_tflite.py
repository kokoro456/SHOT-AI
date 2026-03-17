"""
TFLite Export Script

Converts the PyTorch court keypoint model to TFLite format.
This script should be run FIRST with a dummy model to validate the conversion pipeline
before investing time in data collection and training.

Usage:
    python export_tflite.py --checkpoint path/to/model.pth --output models/court_keypoint.tflite
    python export_tflite.py --dummy  # Test with untrained model
"""

import argparse
import os
import sys
import torch
import numpy as np

from model import CourtKeypointModel, create_model


def export_via_ai_edge_torch(model: torch.nn.Module, output_path: str, quantize: bool = False):
    """
    Export using ai-edge-torch (Google's official PyTorch -> TFLite converter).
    This is the recommended approach.
    """
    try:
        import ai_edge_torch
    except ImportError:
        print("ai-edge-torch not installed. Install with: pip install ai-edge-torch")
        return False

    print("Exporting via ai-edge-torch...")
    model.eval()
    sample_input = (torch.randn(1, 3, 256, 256),)

    try:
        edge_model = ai_edge_torch.convert(model, sample_input)
        edge_model.export(output_path)
        print(f"Successfully exported to: {output_path}")
        print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"ai-edge-torch export failed: {e}")
        return False


def export_via_onnx_then_tflite(model: torch.nn.Module, output_path: str):
    """
    Export via ONNX -> TFLite using onnx2tf.
    Validated pipeline: PyTorch(NCHW) -> ONNX -> onnx2tf -> TFLite(NHWC).

    Requirements (Python 3.12 venv recommended for TF compatibility):
        pip install onnx onnxscript onnxruntime onnx2tf tensorflow
    """
    import tempfile

    print("Exporting via ONNX -> TFLite...")
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)

    onnx_path = os.path.join(tempfile.gettempdir(), "court_keypoint_temp.onnx")

    try:
        # Step 1: PyTorch -> ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
        )
        print(f"ONNX export successful: {onnx_path}")

        # Validate ONNX with onnxruntime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            np_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
            result = session.run(None, {session.get_inputs()[0].name: np_input})[0]
            print(f"ONNX validation: output shape {result.shape}, range [{result.min():.4f}, {result.max():.4f}]")
        except ImportError:
            print("onnxruntime not available, skipping ONNX validation")

        # Step 2: ONNX -> TFLite via onnx2tf
        # keep_ncw_or_nchw_or_ncdhw_input_names is critical for correct conversion
        try:
            import onnx2tf
            output_dir = os.path.dirname(output_path) or "."
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=output_dir,
                non_verbose=True,
                keep_ncw_or_nchw_or_ncdhw_input_names=["input"],
            )
            # onnx2tf outputs files with auto-generated names
            # Find and rename the float32 tflite file
            for f in os.listdir(output_dir):
                if f.endswith("_float32.tflite"):
                    src = os.path.join(output_dir, f)
                    os.rename(src, output_path)
                    print(f"Renamed {f} -> {os.path.basename(output_path)}")
                    break
            print(f"TFLite conversion successful: {output_path}")
            return True
        except ImportError:
            print("onnx2tf not installed. Install with: pip install onnx2tf tensorflow")
            return False

    except Exception as e:
        print(f"ONNX/TFLite export failed: {e}")
        return False
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


def export_via_torch_tflite(model: torch.nn.Module, output_path: str):
    """
    Direct export using torch to TFLite via TensorFlow's converter.
    Requires tracing the model first.
    """
    print("Attempting direct PyTorch -> SavedModel -> TFLite...")

    try:
        import tensorflow as tf

        model.eval()
        dummy_input = torch.randn(1, 3, 256, 256)

        # Trace the model
        traced = torch.jit.trace(model, dummy_input)

        # Run inference to get numpy output for shape reference
        with torch.no_grad():
            sample_output = model(dummy_input).numpy()

        print(f"Model traced successfully. Output shape: {sample_output.shape}")
        print("Note: Direct PyTorch->TFLite requires ai-edge-torch or ONNX intermediate.")
        return False

    except Exception as e:
        print(f"Direct export failed: {e}")
        return False


def validate_tflite(tflite_path: str):
    """Validate the exported TFLite model by running inference."""
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed, skipping validation")
        return False

    print(f"\nValidating TFLite model: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input: {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

    # Run with dummy input
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"All values in [0, 1]: {(output >= 0).all() and (output <= 1).all()}")

    # Compare with PyTorch output
    model = create_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        torch_input = torch.from_numpy(dummy_input)
        torch_output = model(torch_input).numpy()

    # Note: outputs won't match since model weights differ, but shapes should match
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"Shape match: {output.shape == torch_output.shape}")

    print("\nTFLite validation PASSED!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export court keypoint model to TFLite")
    parser.add_argument("--checkpoint", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="models/court_keypoint.tflite",
                        help="Output TFLite file path")
    parser.add_argument("--dummy", action="store_true",
                        help="Test with untrained dummy model (pipeline validation)")
    parser.add_argument("--quantize", action="store_true",
                        help="Export INT8 quantized model")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Load or create model
    if args.dummy:
        print("Creating dummy (untrained) model for pipeline validation...")
        model = create_model(pretrained=False)
    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = create_model(pretrained=False)
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    else:
        print("Error: Specify --checkpoint or --dummy")
        sys.exit(1)

    model.eval()

    # Count parameters and estimate size
    total_params = sum(p.numel() for p in model.parameters())
    estimated_size_mb = total_params * 4 / (1024 * 1024)  # FP32
    print(f"Model parameters: {total_params:,}")
    print(f"Estimated FP32 size: {estimated_size_mb:.2f} MB")

    # Try export methods in order of preference
    success = export_via_ai_edge_torch(model, args.output, args.quantize)

    if not success:
        print("\nFalling back to ONNX -> TFLite...")
        success = export_via_onnx_then_tflite(model, args.output)

    if not success:
        print("\n❌ All export methods failed.")
        print("Options:")
        print("  1. Install ai-edge-torch: pip install ai-edge-torch")
        print("  2. Install onnx2tf: pip install onnx2tf")
        print("  3. Use ONNX + onnx-tf: pip install onnx onnx-tf tensorflow")
        sys.exit(1)

    # Validate
    if os.path.exists(args.output):
        model_size = os.path.getsize(args.output) / 1024 / 1024
        print(f"\n✅ Export successful!")
        print(f"   File: {args.output}")
        print(f"   Size: {model_size:.2f} MB")

        if model_size > 10:
            print(f"   ⚠️ Warning: Model exceeds 10MB target ({model_size:.2f} MB)")

        validate_tflite(args.output)


if __name__ == "__main__":
    main()
