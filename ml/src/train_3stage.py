"""
3-Stage Training Pipeline for Domain Adaptation

Based on expert feedback:
  Stage 1: Pretrain on broadcast data (8.8K) - learn court geometry
  Stage 2: Mixed training (broadcast + phone, target oversampled) - domain adaptation
  Stage 3: Fine-tune on phone data only (low LR) - final target fit

Architecture: HeatmapKeypointModel (MobileNetV3-Small + heatmap decoder)

Usage:
    python train_3stage.py \
        --broadcast-data ../data/broadcast/annotations_broadcast.json \
        --broadcast-images ../data/broadcast/data/images \
        --phone-data ../data/youtube/labeled_annotations.json \
        --phone-images ../data/youtube/review/frames
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, WeightedRandomSampler

from dataset import CourtKeypointDataset
from model_heatmap import HeatmapKeypointModel, create_heatmap_model, NUM_KEYPOINTS, HEATMAP_SIZE
from augmentations_v2 import get_strong_augmentation, get_phone_augmentation, get_val_augmentation

KP_NAMES = ["Pt9(SL)", "Pt10(SC)", "Pt11(SR)", "Pt12(DL)",
            "Pt13(BL)", "Pt14(BC)", "Pt15(BR)", "Pt16(DR)"]


def train_one_epoch_heatmap(model, dataloader, optimizer, device, epoch_num=0):
    """Train one epoch with heatmap loss (MSE on Gaussian targets)."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_kps = batch["keypoints"].to(device)
        gt_vis = batch["visibility"].to(device)

        optimizer.zero_grad()

        # Forward
        pred_heatmaps = model(images)  # [batch, 8, 64, 64]

        # Generate target heatmaps
        target_heatmaps = HeatmapKeypointModel.generate_heatmap_targets(
            gt_kps, gt_vis, heatmap_size=HEATMAP_SIZE, sigma=3.0
        )

        # MSE loss on heatmaps (only for visible keypoints)
        vis_mask = gt_vis.unsqueeze(2).unsqueeze(3)  # [batch, 8, 1, 1]
        loss = ((pred_heatmaps - target_heatmaps) ** 2 * vis_mask).mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_heatmap(model, dataloader, device):
    """Evaluate heatmap model, returning per-keypoint pixel errors."""
    model.eval()
    per_kp_errors = torch.zeros(NUM_KEYPOINTS)
    per_kp_counts = torch.zeros(NUM_KEYPOINTS)
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        gt_kps = batch["keypoints"].to(device)
        gt_vis = batch["visibility"].to(device)

        pred_heatmaps = model(images)
        target_heatmaps = HeatmapKeypointModel.generate_heatmap_targets(
            gt_kps, gt_vis, heatmap_size=HEATMAP_SIZE, sigma=3.0
        )

        vis_mask = gt_vis.unsqueeze(2).unsqueeze(3)
        loss = ((pred_heatmaps - target_heatmaps) ** 2 * vis_mask).mean()
        total_loss += loss.item()
        num_batches += 1

        # Extract predicted coordinates
        pred_coords = HeatmapKeypointModel.heatmaps_to_coords(pred_heatmaps)

        # Per-keypoint pixel error (on 256x256 scale)
        pixel_error = torch.sqrt(((pred_coords - gt_kps) ** 2).sum(dim=2)) * 256
        for i in range(NUM_KEYPOINTS):
            mask = gt_vis[:, i] > 0
            if mask.any():
                per_kp_errors[i] += pixel_error[mask, i].sum().item()
                per_kp_counts[i] += mask.sum().item()

    per_kp_avg = per_kp_errors / (per_kp_counts + 1e-8)
    mean_error = per_kp_avg.mean().item()
    avg_loss = total_loss / max(num_batches, 1)

    return avg_loss, per_kp_avg, mean_error


def run_stage(name, model, train_loader, val_loader, device, epochs, lr, patience, output_dir):
    """Run a training stage with early stopping."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"  LR: {lr}, Epochs: {epochs}, Patience: {patience}", flush=True)
    print(f"{'='*60}", flush=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_mean_error = float("inf")
    best_per_kp = None
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch_heatmap(model, train_loader, optimizer, device, epoch)
        val_loss, per_kp_errors, mean_error = evaluate_heatmap(model, val_loader, device)
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "mean_error_px": round(mean_error, 2),
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Mean: {mean_error:.2f}px | {elapsed:.1f}s", flush=True)

        if mean_error < best_mean_error:
            best_val_loss = val_loss
            best_mean_error = mean_error
            best_per_kp = per_kp_errors.clone()
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "mean_error": mean_error,
                "per_kp_errors": per_kp_errors.tolist(),
            }, os.path.join(output_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}", flush=True)
            break

    print(f"\n  Best: mean_error={best_mean_error:.2f}px", flush=True)
    for i, name_kp in enumerate(KP_NAMES):
        print(f"    {name_kp}: {best_per_kp[i]:.2f}px", flush=True)

    # Save history
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Load best model
    ckpt = torch.load(os.path.join(output_dir, "best_model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    return best_mean_error, best_per_kp, best_val_loss


def main():
    parser = argparse.ArgumentParser(description="3-Stage Training Pipeline")
    parser.add_argument("--broadcast-data", type=str, default="data/broadcast/annotations_broadcast.json")
    parser.add_argument("--broadcast-images", type=str, default="data/broadcast/data/images")
    parser.add_argument("--phone-data", type=str, default="data/youtube/labeled_annotations.json")
    parser.add_argument("--phone-images", type=str, default="data/youtube/review/frames")
    parser.add_argument("--output-dir", type=str, default="models/3stage")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-broadcast", type=int, default=0, help="Limit broadcast data (0=all)")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--input-size", type=int, default=256)
    # Stage-specific args
    parser.add_argument("--s1-epochs", type=int, default=50, help="Stage 1 epochs")
    parser.add_argument("--s1-lr", type=float, default=1e-3)
    parser.add_argument("--s2-epochs", type=int, default=80, help="Stage 2 epochs")
    parser.add_argument("--s2-lr", type=float, default=5e-4)
    parser.add_argument("--s3-epochs", type=int, default=40, help="Stage 3 epochs")
    parser.add_argument("--s3-lr", type=float, default=1e-4)
    parser.add_argument("--target-ratio", type=float, default=0.5,
                        help="Target oversampling ratio in Stage 2 (0.5 = 50:50)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ========================================
    # Load datasets
    # ========================================
    phone_data_path = Path(args.phone_data)
    phone_images_path = Path(args.phone_images)

    phone_full = CourtKeypointDataset(
        annotation_file=str(phone_data_path),
        image_dir=str(phone_images_path),
        input_size=args.input_size,
    )
    print(f"Phone dataset: {len(phone_full)} images", flush=True)

    # Split phone: 80% train, 20% test
    test_size = int(len(phone_full) * args.test_ratio)
    phone_train_size = len(phone_full) - test_size
    generator = torch.Generator().manual_seed(42)
    phone_train, phone_test = random_split(phone_full, [phone_train_size, test_size], generator=generator)
    print(f"Phone train: {len(phone_train)}, Phone test: {len(phone_test)}", flush=True)

    # Broadcast data
    broadcast_data_path = Path(args.broadcast_data)
    broadcast_images_path = Path(args.broadcast_images)
    has_broadcast = broadcast_data_path.exists() and broadcast_images_path.exists()

    broadcast_full = None
    if has_broadcast:
        broadcast_full = CourtKeypointDataset(
            annotation_file=str(broadcast_data_path),
            image_dir=str(broadcast_images_path),
            input_size=args.input_size,
        )
        if args.max_broadcast > 0 and len(broadcast_full) > args.max_broadcast:
            indices = torch.randperm(len(broadcast_full), generator=torch.Generator().manual_seed(42))[:args.max_broadcast]
            broadcast_full = Subset(broadcast_full, indices.tolist())
        print(f"Broadcast dataset: {len(broadcast_full)} images", flush=True)

    # Validation loader (always phone test set)
    phone_test.dataset.augmentation = get_val_augmentation(args.input_size)
    val_loader = DataLoader(phone_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ========================================
    # Create model
    # ========================================
    model = create_heatmap_model(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}", flush=True)

    results = {}

    # ========================================
    # Stage 1: Pretrain on broadcast data
    # ========================================
    if has_broadcast and broadcast_full is not None:
        broadcast_full.augmentation = get_strong_augmentation(args.input_size)
        broadcast_loader = DataLoader(broadcast_full, batch_size=args.batch_size, shuffle=True, num_workers=0)

        s1_error, s1_per_kp, s1_loss = run_stage(
            name="STAGE 1: Pretrain on Broadcast Data",
            model=model,
            train_loader=broadcast_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.s1_epochs,
            lr=args.s1_lr,
            patience=15,
            output_dir=os.path.join(args.output_dir, "stage1"),
        )
        results["stage1"] = {"mean_error": s1_error, "per_kp": s1_per_kp.tolist()}
    else:
        print("WARNING: No broadcast data, skipping Stage 1", flush=True)

    # ========================================
    # Stage 2: Mixed training with target oversampling
    # ========================================
    if has_broadcast and broadcast_full is not None:
        # Apply appropriate augmentations
        broadcast_full.augmentation = get_strong_augmentation(args.input_size)
        phone_train.dataset.augmentation = get_phone_augmentation(args.input_size)

        # Create oversampled dataset
        combined = ConcatDataset([broadcast_full, phone_train])

        # WeightedRandomSampler for target oversampling
        n_broadcast = len(broadcast_full)
        n_phone = len(phone_train)
        target_ratio = args.target_ratio

        # Weight so that phone samples appear ~target_ratio of the time
        w_broadcast = (1 - target_ratio) / n_broadcast
        w_phone = target_ratio / n_phone
        weights = [w_broadcast] * n_broadcast + [w_phone] * n_phone
        sampler = WeightedRandomSampler(weights, num_samples=n_broadcast + n_phone, replacement=True)

        mixed_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        print(f"\nMixed dataset: {n_broadcast} broadcast + {n_phone} phone "
              f"(target ratio: {target_ratio:.0%})", flush=True)

        s2_error, s2_per_kp, s2_loss = run_stage(
            name="STAGE 2: Mixed Training (Target Oversampled)",
            model=model,
            train_loader=mixed_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.s2_epochs,
            lr=args.s2_lr,
            patience=20,
            output_dir=os.path.join(args.output_dir, "stage2"),
        )
        results["stage2"] = {"mean_error": s2_error, "per_kp": s2_per_kp.tolist()}
    else:
        # No broadcast: skip to stage 3 with phone-only
        phone_train.dataset.augmentation = get_phone_augmentation(args.input_size)

    # ========================================
    # Stage 3: Fine-tune on phone data only (low LR)
    # ========================================
    phone_train.dataset.augmentation = get_phone_augmentation(args.input_size)
    phone_loader = DataLoader(phone_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    s3_error, s3_per_kp, s3_loss = run_stage(
        name="STAGE 3: Fine-tune on Phone Data Only",
        model=model,
        train_loader=phone_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.s3_epochs,
        lr=args.s3_lr,
        patience=15,
        output_dir=os.path.join(args.output_dir, "stage3_final"),
    )
    results["stage3_final"] = {"mean_error": s3_error, "per_kp": s3_per_kp.tolist()}

    # ========================================
    # Final Report
    # ========================================
    print(f"\n{'='*60}", flush=True)
    print(f"  3-STAGE TRAINING COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"\n{'Stage':<35} {'Mean Error':>10}", flush=True)
    print("-" * 50, flush=True)
    for stage_name, stage_result in results.items():
        print(f"  {stage_name:<33} {stage_result['mean_error']:>8.2f}px", flush=True)

    print(f"\nFinal model per-keypoint errors:", flush=True)
    for i, name in enumerate(KP_NAMES):
        print(f"  {name}: {s3_per_kp[i]:.2f}px", flush=True)

    # Save final report
    report = {
        "device": str(device),
        "stages": results,
        "final_mean_error": s3_error,
        "final_per_kp": {KP_NAMES[i]: round(s3_per_kp[i].item(), 2) for i in range(NUM_KEYPOINTS)},
        "args": vars(args),
    }
    report_path = os.path.join(args.output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
