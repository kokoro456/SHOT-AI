"""
Single-Frame Ball Detector Training Script

Trains BallDetector (MobileNetV3-Small + decoder) on single frames.
Reuses TrackNet Dataset-001 annotations converted to single-frame format.

Input data format (ball_annotations.json or ball_combined.json):
[
  {"image": "frame_001.jpg", "x": 0.45, "y": 0.32, "visibility": 1},
  {"image": "frame_002.jpg", "x": -1, "y": -1, "visibility": 0},
  ...
]

Coordinates are normalized (0~1). visibility: 0=not visible, 1=visible, 2=occluded.

Usage:
    python train_ball.py --data data/tracknet/ball_combined.json \
                         --frames data/tracknet/frames \
                         --epochs 50 --batch-size 32

    # Resume from checkpoint
    python train_ball.py --data data/tracknet/ball_combined.json \
                         --frames data/tracknet/frames \
                         --checkpoint models/ball_best.pth --resume
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model_ball import BallDetector, BallDetectorLoss, generate_heatmap


# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BallDataset(Dataset):
    """
    Single-frame ball detection dataset.

    Each sample: 1 RGB frame → target heatmap (48×48).
    No triplet grouping needed (unlike TrackNet).
    """

    INPUT_SIZE = 192
    HEATMAP_SIZE = 48  # 192 / 4
    SIGMA = 2.5

    def __init__(self, annotations, frames_dir, augment=False):
        self.frames_dir = Path(frames_dir)
        self.augment = augment

        # Filter annotations with existing frame files
        existing = set()
        if self.frames_dir.is_dir():
            existing = set(os.listdir(str(self.frames_dir)))

        self.samples = []
        skipped = 0
        for ann in annotations:
            if existing and ann["image"] not in existing:
                skipped += 1
                continue
            self.samples.append(ann)

        if skipped > 0:
            print(f"  Skipped {skipped} samples (missing frame files)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ann = self.samples[idx]

        # Load and preprocess image
        img_path = self.frames_dir / ann["image"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

        # Augmentation
        if self.augment:
            img = self._augment(img)

        # ImageNet normalization + NCHW
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # [3, H, W]

        # Generate target heatmap (48×48)
        if ann["visibility"] > 0 and ann["x"] >= 0:
            bx = ann["x"] * self.HEATMAP_SIZE
            by = ann["y"] * self.HEATMAP_SIZE
            heatmap = generate_heatmap(bx, by, self.HEATMAP_SIZE, self.SIGMA)
        else:
            heatmap = torch.zeros(self.HEATMAP_SIZE, self.HEATMAP_SIZE)

        return (
            torch.from_numpy(img).float(),
            heatmap.unsqueeze(0).float(),  # [1, 48, 48]
            torch.tensor(ann["visibility"], dtype=torch.long)
        )

    def _augment(self, img):
        """Simple augmentations for single frame."""
        # Random brightness
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 1)

        # Color shift
        if np.random.random() < 0.3:
            shift = np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
            img = np.clip(img + shift, 0, 1)

        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        # Horizontal flip (x coordinate flipped in heatmap generation)
        # NOTE: We don't flip here because the heatmap is generated from
        # original coordinates. Instead, we flip both image and coordinates
        # together in a separate pass if needed.

        return img


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets, visibility in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        for i in range(len(visibility)):
            if visibility[i] > 0:
                total += 1
                if outputs[i].max().item() > 0.5:
                    correct += 1

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1) * 100
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    position_errors = []

    with torch.no_grad():
        for inputs, targets, visibility in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            for i in range(len(visibility)):
                if visibility[i] > 0:
                    total += 1
                    pred_hm = outputs[i, 0]
                    gt_hm = targets[i, 0]

                    pred_max = pred_hm.max().item()
                    if pred_max > 0.5:
                        correct += 1

                        # Position error (in heatmap pixels)
                        pred_idx = pred_hm.argmax()
                        gt_idx = gt_hm.argmax()
                        h, w = pred_hm.shape
                        pred_y, pred_x = pred_idx // w, pred_idx % w
                        gt_y, gt_x = gt_idx // w, gt_idx % w
                        err = ((pred_x - gt_x).float() ** 2 +
                               (pred_y - gt_y).float() ** 2).sqrt().item()
                        position_errors.append(err)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1) * 100
    avg_error = np.mean(position_errors) if position_errors else float('inf')
    return avg_loss, accuracy, avg_error


def main():
    parser = argparse.ArgumentParser(description="Train single-frame ball detector")
    parser.add_argument("--data", required=True, help="Ball annotations JSON")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4,
                        help="Learning rate for backbone (lower for pretrained)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5,
                        help="Freeze backbone for first N epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load annotations
    with open(args.data) as f:
        all_annotations = json.load(f)

    print(f"Total annotations: {len(all_annotations)}")
    visible = sum(1 for a in all_annotations if a["visibility"] > 0)
    print(f"Visible balls: {visible}, Not visible: {len(all_annotations) - visible}")

    # Video-level split
    video_frames = defaultdict(list)
    for ann in all_annotations:
        name = Path(ann["image"]).stem
        match = re.match(r"(.+?)_frame_\d+$", name)
        vid_id = match.group(1) if match else name
        video_frames[vid_id].append(ann)

    video_ids = sorted(video_frames.keys())
    np.random.seed(42)
    np.random.shuffle(video_ids)
    split_idx = max(1, int(len(video_ids) * 0.8))

    train_ann = []
    for vid in video_ids[:split_idx]:
        train_ann.extend(video_frames[vid])
    val_ann = []
    for vid in video_ids[split_idx:]:
        val_ann.extend(video_frames[vid])

    print(f"Videos: {len(video_ids)} total, {split_idx} train, {len(video_ids) - split_idx} val")
    print(f"Frames: {len(train_ann)} train, {len(val_ann)} val")

    # Datasets
    train_dataset = BallDataset(train_ann, args.frames, augment=True)
    val_dataset = BallDataset(val_ann, args.frames, augment=False)

    print(f"\n=== Training Data Summary ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Input: [B, 3, 192, 192] → Output: [B, 1, 48, 48]")
    print(f"=============================\n")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    # Model
    model = BallDetector(pretrained=True).to(device)
    criterion = BallDetectorLoss()

    # Separate learning rates: backbone (pretrained, lower LR) vs decoder (higher LR)
    backbone_params = list(model.features.parameters())
    decoder_params = list(model.decoder.parameters())
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": decoder_params, "lr": args.lr},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB)")

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint and args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>8} | {'Val Err':>7} | {'LR':>10} | {'Note':>10}")
    print("-" * 95)

    patience = 15
    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        # Freeze backbone for first N epochs (train decoder only)
        if epoch < args.freeze_backbone_epochs:
            for p in model.features.parameters():
                p.requires_grad = False
            note = "frozen"
        elif epoch == args.freeze_backbone_epochs:
            for p in model.features.parameters():
                p.requires_grad = True
            note = "unfrozen"
        else:
            note = ""

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_err = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        lr = optimizer.param_groups[1]["lr"]  # decoder LR
        print(f"{epoch:>5} | {train_loss:>10.6f} | {train_acc:>8.1f}% | {val_loss:>10.6f} | {val_acc:>7.1f}% | {val_err:>6.1f}px | {lr:>10.6f} | {note:>10}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.output_dir, "ball_best.pth"))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

        if (epoch + 1) % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.output_dir, f"ball_epoch{epoch+1}.pth"))

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Best model: {args.output_dir}/ball_best.pth")


if __name__ == "__main__":
    main()
