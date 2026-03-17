"""
Comparison training script: 3 experiments with different data sources.

Experiment A: Broadcast data only (yastrebksv/TennisCourtDetector)
Experiment B: YouTube phone data only (hand-labeled by user)
Experiment C: Combined (A + B)

All 3 models are evaluated on the SAME test set (phone-filmed images)
to fairly compare domain gap effects.

Usage:
    python train_compare.py

Results are saved to models/comparison/ with side-by-side metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

from dataset import CourtKeypointDataset
from model import CourtKeypointModel, create_model, NUM_KEYPOINTS
from augmentations import get_train_augmentation, get_val_augmentation

# Per-keypoint loss weights
KEYPOINT_WEIGHTS = torch.tensor([1.2, 1.2, 1.2, 1.0, 1.5, 1.5, 1.5, 1.0])
KP_NAMES = ["Pt9(SL)", "Pt10(SC)", "Pt11(SR)", "Pt12(DL)", "Pt13(BL)", "Pt14(BC)", "Pt15(BR)", "Pt16(DR)"]


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    coord_criterion = nn.SmoothL1Loss(reduction="none")
    conf_criterion = nn.BCELoss(reduction="none")
    weights = KEYPOINT_WEIGHTS.to(device)

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_kps = batch["keypoints"].to(device)
        gt_vis = batch["visibility"].to(device)

        optimizer.zero_grad()
        output = model(images)
        coords, confidence = CourtKeypointModel.parse_output(output)

        coord_loss = coord_criterion(coords, gt_kps).mean(dim=2) * gt_vis
        coord_loss = (coord_loss * weights.unsqueeze(0)).sum() / (gt_vis.sum() + 1e-8)
        conf_loss = (conf_criterion(confidence, gt_vis) * weights.unsqueeze(0)).mean()
        loss = coord_loss + 0.5 * conf_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate and return per-keypoint pixel errors + mean error."""
    model.eval()
    per_kp_errors = torch.zeros(NUM_KEYPOINTS)
    per_kp_counts = torch.zeros(NUM_KEYPOINTS)
    total_loss = 0
    num_batches = 0

    coord_criterion = nn.SmoothL1Loss(reduction="none")
    conf_criterion = nn.BCELoss(reduction="none")
    weights = KEYPOINT_WEIGHTS.to(device)

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_kps = batch["keypoints"].to(device)
        gt_vis = batch["visibility"].to(device)

        output = model(images)
        coords, confidence = CourtKeypointModel.parse_output(output)

        coord_loss = coord_criterion(coords, gt_kps).mean(dim=2) * gt_vis
        coord_loss = (coord_loss * weights.unsqueeze(0)).sum() / (gt_vis.sum() + 1e-8)
        conf_loss = (conf_criterion(confidence, gt_vis) * weights.unsqueeze(0)).mean()
        loss = coord_loss + 0.5 * conf_loss
        total_loss += loss.item()
        num_batches += 1

        pixel_error = torch.sqrt(((coords - gt_kps) ** 2).sum(dim=2)) * 256
        for i in range(NUM_KEYPOINTS):
            mask = gt_vis[:, i] > 0
            if mask.any():
                per_kp_errors[i] += pixel_error[mask, i].sum().item()
                per_kp_counts[i] += mask.sum().item()

    per_kp_avg = per_kp_errors / (per_kp_counts + 1e-8)
    mean_error = per_kp_avg.mean().item()
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, per_kp_avg, mean_error


def train_experiment(name, train_dataset, val_dataset, output_dir, device,
                     epochs=100, batch_size=32, lr=1e-3, patience=15):
    """Run a single training experiment."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT {name}")
    print(f"  Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    print(f"{'='*60}\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_per_kp = None
    best_mean_error = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, per_kp_errors, mean_error = evaluate(model, val_loader, device)
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "mean_error_px": round(mean_error, 2),
            "per_kp_errors": [round(e.item(), 2) for e in per_kp_errors],
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Mean: {mean_error:.2f}px")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_per_kp = per_kp_errors.clone()
            best_mean_error = mean_error
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "per_kp_errors": per_kp_errors.tolist(),
                "mean_error": mean_error,
            }, os.path.join(output_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Save training history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    result = {
        "name": name,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "best_val_loss": round(best_val_loss, 6),
        "best_mean_error_px": round(best_mean_error, 2),
        "per_kp_errors_px": {
            KP_NAMES[i]: round(best_per_kp[i].item(), 2) for i in range(NUM_KEYPOINTS)
        },
        "epochs_trained": len(history),
    }

    print(f"\n  Best: loss={best_val_loss:.4f}, mean_error={best_mean_error:.2f}px")
    for i, name_kp in enumerate(KP_NAMES):
        print(f"    {name_kp}: {best_per_kp[i]:.2f}px")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare 3 training experiments")
    parser.add_argument("--broadcast-data", type=str, default="data/broadcast/annotations_broadcast.json")
    parser.add_argument("--broadcast-images", type=str, default="data/broadcast/images")
    parser.add_argument("--phone-data", type=str, default="data/youtube/labeled_annotations.json")
    parser.add_argument("--phone-images", type=str, default="data/youtube/review/frames")
    parser.add_argument("--output-dir", type=str, default="models/comparison")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Ratio of phone data to hold out for testing")
    parser.add_argument("--experiments", type=str, default="A,B,C",
                        help="Comma-separated experiments to run (A,B,C)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ========================================
    # Load datasets
    # ========================================

    # Phone data (hand-labeled YouTube)
    phone_data_path = Path(args.phone_data)
    phone_images_path = Path(args.phone_images)

    if not phone_data_path.exists():
        print(f"ERROR: Phone annotation file not found: {phone_data_path}")
        sys.exit(1)

    phone_full = CourtKeypointDataset(
        annotation_file=str(phone_data_path),
        image_dir=str(phone_images_path),
        input_size=256,
    )
    print(f"Phone dataset: {len(phone_full)} images")

    # Split phone data: 80% train, 20% test (shared test set for all experiments)
    test_size = int(len(phone_full) * args.test_ratio)
    phone_train_size = len(phone_full) - test_size

    # Use fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    phone_train_dataset, phone_test_dataset = random_split(
        phone_full, [phone_train_size, test_size], generator=generator
    )

    print(f"Phone train: {len(phone_train_dataset)}, Phone test: {len(phone_test_dataset)}")

    # Broadcast data
    broadcast_data_path = Path(args.broadcast_data)
    broadcast_images_path = Path(args.broadcast_images)
    has_broadcast = broadcast_data_path.exists() and broadcast_images_path.exists()

    if has_broadcast:
        broadcast_full = CourtKeypointDataset(
            annotation_file=str(broadcast_data_path),
            image_dir=str(broadcast_images_path),
            input_size=256,
        )
        print(f"Broadcast dataset: {len(broadcast_full)} images")
    else:
        print(f"WARNING: Broadcast data not found at {broadcast_data_path}")
        print("Skipping Experiment A and C (broadcast-dependent)")

    # Apply augmentations
    phone_full.augmentation = get_train_augmentation(256)

    # ========================================
    # Run experiments
    # ========================================
    experiments_to_run = args.experiments.split(",")
    results = {}

    # Shared test loader (always phone data - this is what we care about)
    test_loader = DataLoader(phone_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Experiment A: Broadcast only
    if "A" in experiments_to_run and has_broadcast:
        broadcast_full.augmentation = get_train_augmentation(256)
        result_a = train_experiment(
            name="A: Broadcast Only",
            train_dataset=broadcast_full,  # All broadcast for training
            val_dataset=phone_test_dataset,  # Test on phone data
            output_dir=os.path.join(args.output_dir, "exp_A_broadcast"),
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        results["A"] = result_a

    # Experiment B: Phone only
    if "B" in experiments_to_run:
        result_b = train_experiment(
            name="B: Phone (YouTube) Only",
            train_dataset=phone_train_dataset,
            val_dataset=phone_test_dataset,
            output_dir=os.path.join(args.output_dir, "exp_B_phone"),
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        results["B"] = result_b

    # Experiment C: Combined
    if "C" in experiments_to_run and has_broadcast:
        broadcast_full.augmentation = get_train_augmentation(256)
        combined_train = ConcatDataset([broadcast_full, phone_train_dataset])
        result_c = train_experiment(
            name="C: Combined (Broadcast + Phone)",
            train_dataset=combined_train,
            val_dataset=phone_test_dataset,
            output_dir=os.path.join(args.output_dir, "exp_C_combined"),
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        results["C"] = result_c

    # ========================================
    # Final comparison
    # ========================================
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON (tested on phone-filmed images)")
    print("=" * 70)

    header = f"{'Experiment':<35} {'Mean Error':>10} {'Val Loss':>10} {'Train Size':>10}"
    print(header)
    print("-" * 70)

    for key in sorted(results.keys()):
        r = results[key]
        print(f"{r['name']:<35} {r['best_mean_error_px']:>8.2f}px {r['best_val_loss']:>10.6f} {r['train_size']:>10}")

    print("\nPer-keypoint errors (px on 256x256):")
    print(f"{'Keypoint':<12}", end="")
    for key in sorted(results.keys()):
        print(f"  {'Exp '+key:>10}", end="")
    print()
    print("-" * (12 + 12 * len(results)))

    for kp_name in KP_NAMES:
        print(f"{kp_name:<12}", end="")
        for key in sorted(results.keys()):
            err = results[key]["per_kp_errors_px"].get(kp_name, -1)
            print(f"  {err:>8.2f}px", end="")
        print()

    # Save comparison report
    report_path = os.path.join(args.output_dir, "comparison_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
