"""
Training script for the court keypoint detection model.

Usage:
    python train.py --config configs/train_config.yaml
    python train.py --data data/raw/annotations.json --image-dir data/raw/images
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from augmentations import get_train_augmentation, get_val_augmentation
from dataset import CourtKeypointDataset
from model import CourtKeypointModel, create_model, NUM_KEYPOINTS

# Per-keypoint loss weights (spec: baseline 1.5x, service 1.2x, doubles 1.0x)
# Order: [9, 10, 11, 12, 13, 14, 15, 16]
KEYPOINT_WEIGHTS = torch.tensor([
    1.2,  # 9  - service line
    1.2,  # 10 - service line
    1.2,  # 11 - service line
    1.0,  # 12 - doubles sideline
    1.5,  # 13 - baseline
    1.5,  # 14 - baseline (center mark)
    1.5,  # 15 - baseline
    1.0,  # 16 - doubles sideline
])


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    coord_loss_sum = 0
    conf_loss_sum = 0
    num_batches = 0

    coord_criterion = nn.SmoothL1Loss(reduction="none")
    conf_criterion = nn.BCELoss(reduction="none")

    weights = KEYPOINT_WEIGHTS.to(device)

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        gt_keypoints = batch["keypoints"].to(device)  # [B, 8, 2]
        gt_visibility = batch["visibility"].to(device)  # [B, 8]

        optimizer.zero_grad()

        output = model(images)  # [B, 24]
        coords, confidence = CourtKeypointModel.parse_output(output)
        # coords: [B, 8, 2], confidence: [B, 8]

        # Coordinate loss (only for visible keypoints)
        coord_loss = coord_criterion(coords, gt_keypoints)  # [B, 8, 2]
        coord_loss = coord_loss.mean(dim=2)  # [B, 8]
        coord_loss = coord_loss * gt_visibility  # mask invisible
        coord_loss = coord_loss * weights.unsqueeze(0)  # per-keypoint weighting
        coord_loss = coord_loss.sum() / (gt_visibility.sum() + 1e-8)

        # Confidence loss
        conf_loss = conf_criterion(confidence, gt_visibility)  # [B, 8]
        conf_loss = (conf_loss * weights.unsqueeze(0)).mean()

        # Total loss
        loss = coord_loss + 0.5 * conf_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        coord_loss_sum += coord_loss.item()
        conf_loss_sum += conf_loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_coord = coord_loss_sum / max(num_batches, 1)
    avg_conf = conf_loss_sum / max(num_batches, 1)

    return avg_loss, avg_coord, avg_conf


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    per_kp_errors = torch.zeros(NUM_KEYPOINTS)
    per_kp_counts = torch.zeros(NUM_KEYPOINTS)

    coord_criterion = nn.SmoothL1Loss(reduction="none")
    conf_criterion = nn.BCELoss(reduction="none")
    weights = KEYPOINT_WEIGHTS.to(device)

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        gt_keypoints = batch["keypoints"].to(device)
        gt_visibility = batch["visibility"].to(device)

        output = model(images)
        coords, confidence = CourtKeypointModel.parse_output(output)

        # Total loss
        coord_loss = coord_criterion(coords, gt_keypoints).mean(dim=2) * gt_visibility
        coord_loss = (coord_loss * weights.unsqueeze(0)).sum() / (gt_visibility.sum() + 1e-8)
        conf_loss = (conf_criterion(confidence, gt_visibility) * weights.unsqueeze(0)).mean()
        loss = coord_loss + 0.5 * conf_loss
        total_loss += loss.item()

        # Per-keypoint pixel error (on 256x256 scale)
        pixel_error = torch.sqrt(((coords - gt_keypoints) ** 2).sum(dim=2)) * 256  # [B, 8]
        for i in range(NUM_KEYPOINTS):
            mask = gt_visibility[:, i] > 0
            if mask.any():
                per_kp_errors[i] += pixel_error[mask, i].sum().item()
                per_kp_counts[i] += mask.sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    per_kp_avg_errors = per_kp_errors / (per_kp_counts + 1e-8)

    return avg_loss, per_kp_avg_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Annotation JSON file")
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--input-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    full_dataset = CourtKeypointDataset(
        annotation_file=args.data,
        image_dir=args.image_dir,
        input_size=args.input_size,
    )

    # Split 80/20
    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply augmentations
    train_dataset.dataset.augmentation = get_train_augmentation(args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = create_model(pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    kp_names = ["Pt9(SL)", "Pt10(SC)", "Pt11(SR)", "Pt12(DL)", "Pt13(BL)", "Pt14(BC)", "Pt15(BR)", "Pt16(DR)"]

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        train_loss, train_coord, train_conf = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, per_kp_errors = validate(model, val_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} (coord: {train_coord:.4f}, conf: {train_conf:.4f})")
        print(f"Val Loss: {val_loss:.4f}")
        print("Per-keypoint errors (px on 256x256):")
        for i, (name, err) in enumerate(zip(kp_names, per_kp_errors)):
            status = "✓" if err < (3 if i < 3 else 4 if i in [3, 7] else 2) else "✗"
            print(f"  {name}: {err:.2f} px {status}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "per_kp_errors": per_kp_errors.tolist(),
            }, save_path)
            print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={args.patience})")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
