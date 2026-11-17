import os
import sys
import argparse
import math
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

# Set up paths
# This file lives in src/training_scripts/, the imports and data are relative to src/
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_dir)

from models.backbone.feature_extractor import DilatedResNet18
from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from losses.gaussian_mse import GaussianMSE_norm, get_map_kernel
from utils.exclusions import load_excluded_scenes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PretrainingModel2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DilatedResNet18(pretrained=True, use_dilation=True)

        # SynMVCrowd's image head (`img_heatmap1 = output_head(512, outfeat_dim=64, 1)` in mvdetr.py)
        self.head_2d = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        # raw output, as in the original (no final activation on the image head)
        return self.head_2d(features)


# SynMVCrowd frameDataset: head points rasterised at 360x640, img kernel variance 10 / img_reduce(4) = 2.5,
# half-size 10 (21x21), peak-normalised; GaussianMSE_norm max-pools the points to the output grid
# (90x160 at stride 8), blurs, and rescales the target to count * 100.
POINT_MAP_SHAPE = (360, 640)
IMG_SIGMA, IMG_KERNEL_SIZE = 10 / 4, 10
SCALE = 100


def create_point_maps(gt_points_list, device, shape=POINT_MAP_SHAPE, image_size=(720, 1280)):
    """[N, 1, 360, 640] binary head-point maps from the dataset's 720x1280 pixel coordinates."""
    H, W = shape
    point_maps = torch.zeros((len(gt_points_list), 1, H, W), device=device)
    for i, pts in enumerate(gt_points_list):
        pts = pts.view(-1, 2).to(device)
        if len(pts) == 0:
            continue
        x = (pts[:, 0] * W / image_size[1]).long()
        y = (pts[:, 1] * H / image_size[0]).long()
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        point_maps[i, 0, y[valid], x[valid]] = 1.0
    return point_maps


def load_split(split, excluded_scenes):
    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split)
    datasets = []
    for scene_name in sorted(os.listdir(split_dir)):
        scene_path = os.path.join(split_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue
        if scene_name in excluded_scenes:
            print(f"[EXCLUDED] {split}/{scene_name}: {excluded_scenes[scene_name]}")
            continue
        datasets.append(SynMVCrowdDataset(scene_dir=scene_path, num_sampled_cams=5, load_refs=False))
    return datasets


def train_2d(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting 2D Pretraining on {device} (Pure FP32) ---")

    # Setup directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    run_dir = os.path.join(src_dir, "runs", "train_2d", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {run_dir}")

    # Initialize model, dataset, dataloader
    model = PretrainingModel2D().to(device)

    # --- Train Dataset Setup ---
    excluded_scenes = load_excluded_scenes()
    train_datasets = load_split("train", excluded_scenes)
    full_train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    # --- Validation Dataset Setup ---
    val_datasets = load_split("validate", excluded_scenes)
    full_val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(full_val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = GaussianMSE_norm(scale=SCALE).to(device)
    img_kernel = get_map_kernel(sigma=IMG_SIGMA, kernel_size=IMG_KERNEL_SIZE).to(device)
    print(
        f"Backbone stride {model.backbone.stride}: 2D predictions at {(720 // model.backbone.stride, 1280 // model.backbone.stride)}"
    )

    start_epoch = 1
    # The epoch with the lowest val MAE is the backbone every later model uses (epoch 4 in our run); it is copied to
    # best.pth in the run directory
    best_val_mae, best_epoch = float("inf"), None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # checkpoints from before the best tracking only know their own epoch's val MAE
        best_val_mae = checkpoint.get("best_val_mae", checkpoint["val_mae"])
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"])
        print(f"Resumed from epoch {start_epoch - 1} (best so far: epoch {best_epoch}, val MAE {best_val_mae:.2f})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        # Wrap train_loader with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", mininterval=10)
        for batch_idx, batch in enumerate(train_pbar):
            # Move all 5 images to the GPU instantly in the background
            images = batch["images"].squeeze(0).to(device, non_blocking=True)
            gt_2d_xy = batch["gt_2d_xy"]

            points_list = [pts.squeeze(0) for pts in gt_2d_xy]
            point_maps = create_point_maps(points_list, device)

            optimizer.zero_grad()

            predictions = model(images)
            # one GaussianMSE_norm call over all views of the sample, as in the original trainer
            loss, _ = criterion(predictions, point_maps, img_kernel)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_train_loss += loss.item()

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_errors = []

        # Wrap val_loader with tqdm
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", mininterval=10)
        with torch.no_grad():
            for batch in val_pbar:
                images = batch["images"].squeeze(0).to(device)
                gt_2d_xy = batch["gt_2d_xy"]
                points_list = [pts.squeeze(0) for pts in gt_2d_xy]

                predictions = model(images)

                for n in range(len(images)):
                    pred_density = predictions[n, 0] / SCALE
                    pred_count = torch.sum(pred_density).item()

                    pts = points_list[n].view(-1, 2)
                    true_count = len(pts) if len(pts) > 0 else 0

                    if not math.isfinite(pred_count):
                        continue

                    val_errors.append(abs(pred_count - true_count))

        if len(val_errors) > 0:
            val_mae = np.mean(val_errors)
            val_medae = np.median(val_errors)
        else:
            val_mae = float("inf")
            val_medae = float("inf")

        # Step the scheduler based on MAE
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n>>> END OF EPOCH {epoch} <<<")
        print(f"    Train Loss: {avg_train_loss:.5f}")
        print(f"    Val MAE:    {val_mae:.2f} people")
        print(f"    Val MedAE:  {val_medae:.2f} people")
        print(f"    Current LR: {current_lr}")

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae, best_epoch = val_mae, epoch
        print(f"    Best:       epoch {best_epoch} (val MAE {best_val_mae:.2f}){'  <-- new best' if is_best else ''}")

        checkpoint_path = os.path.join(run_dir, f"resnet18_2d_pretrained_ep{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_mae": val_mae,
                "val_medae": val_medae,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
            },
            checkpoint_path,
        )
        print(f"\nSaved checkpoint: {checkpoint_path}")
        if is_best:
            shutil.copyfile(checkpoint_path, os.path.join(run_dir, "best.pth"))
            print(f"Copied to {os.path.join(run_dir, 'best.pth')}")

    print(f"\n--- Done. Best epoch {best_epoch} (val MAE {best_val_mae:.2f}): {os.path.join(run_dir, 'best.pth')} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_2d(args)
