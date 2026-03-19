"""
TrackNet Training Script

Trains the TrackNet ball detection model on labeled tennis video frames.

Input data format (ball_annotations.json):
[
  {"image": "video_frame_001.jpg", "x": 0.45, "y": 0.32, "visibility": 1},
  {"image": "video_frame_002.jpg", "x": -1, "y": -1, "visibility": 0},
  ...
]

Usage:
    python train_tracknet.py --data data/sntc/ball_annotations.json \
                             --frames data/sntc/frames \
                             --epochs 50 --batch-size 16

    # Resume from checkpoint
    python train_tracknet.py --data data/sntc/ball_annotations.json \
                             --frames data/sntc/frames \
                             --checkpoint models/tracknet_best.pth --resume
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model_tracknet import TrackNet, TrackNetLoss, generate_heatmap


class TrackNetDataset(Dataset):
    """
    Dataset for TrackNet training.

    Groups frames by video, creates triplets of consecutive frames.
    Each sample: 3 consecutive frames → target heatmap for the last frame.
    """

    INPUT_H = 128
    INPUT_W = 320
    SIGMA = 5.0  # Gaussian heatmap sigma

    def __init__(self, annotations, frames_dir, augment=False):
        self.frames_dir = Path(frames_dir)
        self.augment = augment

        # Group by video
        video_frames = defaultdict(list)
        for ann in annotations:
            name = Path(ann["image"]).stem
            match = re.match(r"(.+?)_frame_(\d+)$", name)
            if match:
                vid_id = match.group(1)
                frame_idx = int(match.group(2))
            else:
                vid_id = name
                frame_idx = 0
            video_frames[vid_id].append((frame_idx, ann))

        # Sort frames within each video
        for vid in video_frames:
            video_frames[vid].sort(key=lambda x: x[0])

        # Create triplets: (frame_t-2, frame_t-1, frame_t) with label for frame_t
        self.samples = []
        for vid, frames in video_frames.items():
            for i in range(len(frames)):
                # Get 3 consecutive frames (or repeat if not enough)
                f0 = frames[max(0, i - 2)][1]
                f1 = frames[max(0, i - 1)][1]
                f2 = frames[i][1]
                self.samples.append((f0, f1, f2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f0, f1, f2 = self.samples[idx]

        # Load 3 frames
        imgs = []
        for f in [f0, f1, f2]:
            img_path = self.frames_dir / f["image"]
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.INPUT_W, self.INPUT_H), Image.BILINEAR)
            img = np.array(img, dtype=np.float32) / 255.0
            imgs.append(img)

        # Stack: [H, W, 9] → [9, H, W]
        stacked = np.concatenate(imgs, axis=2)  # [H, W, 9]
        stacked = np.transpose(stacked, (2, 0, 1))  # [9, H, W]

        # Generate target heatmap for frame_t (last frame)
        target = f2
        if target["visibility"] > 0 and target["x"] >= 0:
            # Scale coordinates to heatmap resolution
            bx = target["x"] * self.INPUT_W
            by = target["y"] * self.INPUT_H
            heatmap = generate_heatmap(bx, by, self.INPUT_W, self.INPUT_H, self.SIGMA)
        else:
            heatmap = torch.zeros(self.INPUT_H, self.INPUT_W)

        # Data augmentation
        if self.augment:
            stacked, heatmap = self._augment(stacked, heatmap)

        return (
            torch.from_numpy(stacked).float(),
            heatmap.unsqueeze(0).float(),  # [1, H, W]
            torch.tensor(target["visibility"], dtype=torch.long)
        )

    def _augment(self, img, heatmap):
        """Simple augmentations that preserve ball position."""
        # Random brightness
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 1)

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.flip(img, axis=2).copy()  # flip W
            heatmap = torch.flip(heatmap, [1])  # flip W

        return img, heatmap


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, visibility) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Detection accuracy: did we find the ball in visible frames?
        for i in range(len(visibility)):
            if visibility[i] > 0:  # ball should be visible
                pred_max = outputs[i].max().item()
                total += 1
                if pred_max > 0.5:
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

                        # Position error
                        pred_y, pred_x = torch.where(pred_hm == pred_hm.max())
                        gt_y, gt_x = torch.where(gt_hm == gt_hm.max())
                        if len(pred_y) > 0 and len(gt_y) > 0:
                            err = ((pred_x[0] - gt_x[0]).float() ** 2 +
                                   (pred_y[0] - gt_y[0]).float() ** 2).sqrt().item()
                            position_errors.append(err)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1) * 100
    avg_error = np.mean(position_errors) if position_errors else float('inf')
    return avg_loss, accuracy, avg_error


def main():
    parser = argparse.ArgumentParser(description="Train TrackNet ball detector")
    parser.add_argument("--data", required=True, help="Ball annotations JSON")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load annotations
    with open(args.data) as f:
        all_annotations = json.load(f)

    print(f"Total annotations: {len(all_annotations)}")
    visible = sum(1 for a in all_annotations if a["visibility"] > 0)
    print(f"Visible balls: {visible}, Not visible: {len(all_annotations) - visible}")

    # Train/val split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(all_annotations))
    split = int(len(indices) * 0.8)
    train_ann = [all_annotations[i] for i in indices[:split]]
    val_ann = [all_annotations[i] for i in indices[split:]]

    print(f"Train: {len(train_ann)}, Val: {len(val_ann)}")

    # Datasets
    train_dataset = TrackNetDataset(train_ann, args.frames, augment=True)
    val_dataset = TrackNetDataset(val_ann, args.frames, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = TrackNet(input_channels=9, base_filters=32).to(device)
    criterion = TrackNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB)")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint and args.resume and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>8} | {'Val Err':>7} | {'LR':>10}")
    print("-" * 80)

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_err = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5} | {train_loss:>10.6f} | {train_acc:>8.1f}% | {val_loss:>10.6f} | {val_acc:>7.1f}% | {val_err:>6.1f}px | {lr:>10.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.output_dir, "tracknet_best.pth"))
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
            }, os.path.join(args.output_dir, f"tracknet_epoch{epoch+1}.pth"))

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Best model: {args.output_dir}/tracknet_best.pth")


if __name__ == "__main__":
    main()
