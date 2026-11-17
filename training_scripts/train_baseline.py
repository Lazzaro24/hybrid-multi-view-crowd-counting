import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import argparse
import math
import random
import time
from datetime import datetime
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from tqdm import tqdm

# Set up paths (this file lives in src/training_scripts/, the imports and data are relative to src/)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset
from models.backbone.feature_extractor import DilatedResNet18
from models.neck.baseline.feature_selection import ViewFeatureSelector
from models.neck.baseline.projector import SpatialProjector
from models.head.baseline.decoder import BaselineDecoder
from losses.gaussian_mse import GaussianMSE_norm, get_map_kernel
from losses.ot_loss.genloss_synmvcrowd import SynMVCrowdGeneralizedLoss, patch_points
from utils.exclusions import load_excluded_scenes


class BaselineModel(nn.Module):
    def __init__(self, num_cam=5, pretrained_backbone_path=None, train_backbone=False):
        """Wraps the baseline components (backbone, view selection, projection, decoder) into one model.

        train_backbone=True trains the backbone jointly like the released SynMVCrowd main.py: its weights
        get gradients, its BatchNorm layers stay in train mode, and it shares the optimizer's learning rate.
        """
        super().__init__()
        self.train_backbone = train_backbone
        self.backbone = DilatedResNet18(pretrained=True)

        # 1. Load custom 2D pretrained weights if provided
        if pretrained_backbone_path:
            # A missing file would silently fall back to ImageNet features
            if not os.path.isfile(pretrained_backbone_path):
                raise FileNotFoundError(f"2D-pretrained backbone not found: {pretrained_backbone_path!r}")
            print(
                f"--> Loading {'trainable' if train_backbone else 'frozen'} 2D backbone from {pretrained_backbone_path}"
            )
            checkpoint = torch.load(pretrained_backbone_path, map_location="cpu", weights_only=False)

            # Filter out the temporary 2D head and remove the 'backbone.' prefix
            backbone_weights = {
                k.replace("backbone.", ""): v
                for k, v in checkpoint["model_state_dict"].items()
                if k.startswith("backbone.")
            }
            self.backbone.load_state_dict(backbone_weights)

        # 2. Freeze the backbone gradients (unless it is trained jointly)
        for param in self.backbone.parameters():
            param.requires_grad = train_backbone

        self.feature_selector = ViewFeatureSelector(in_channels=512, hidden_channels=64)
        self.projector = SpatialProjector(bev_shape=(200, 200), original_image_shape=(1080, 1920))
        self.decoder = BaselineDecoder(num_cam=num_cam, out_channel=512)

    def train(self, mode=True):
        """Override train to ensure the frozen backbone stays in eval mode."""
        super().train(mode)
        # 3. Keep BatchNorm stats frozen (a jointly trained backbone updates them, as in main.py)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def forward(self, images, camera_paras, wld_map_paras, hw_random=None, full_scene_shape=None):
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)

        with torch.set_grad_enabled(self.train_backbone and torch.is_grad_enabled()):
            features_flat = self.backbone(images_flat)

        _, C_f, H_f, W_f = features_flat.shape
        features = features_flat.view(B, N, C_f, H_f, W_f)
        features = self.feature_selector(features)

        # --- EVALUATION PATH: DYNAMIC FULL SCENE ---
        if full_scene_shape is not None:
            # Force a (0, 0) offset tensor for the full scene
            hw_zero_offset = torch.zeros((B, 2), device=images.device)
            if torch.is_grad_enabled():
                proj_bev = self.projector(
                    features, camera_paras, wld_map_paras, hw_zero_offset, target_shape=full_scene_shape
                )
            else:
                # Inference: project and max-fuse one view at a time. Identical output (the decoder's first
                # step is a max over views), but only one [C, H, W] map is held instead of all N -- a large
                # scene such as test/scene_13 otherwise needs ~2.3 GB for this tensor alone and overflows a
                # 12 GB card into shared memory, slowing evaluation from <1 s to 6-9 s per sample.
                proj_bev = None
                for v in range(N):
                    proj_v = self.projector(
                        features[:, v : v + 1].contiguous(),
                        camera_paras[:, v : v + 1].contiguous(),
                        wld_map_paras,
                        hw_zero_offset,
                        target_shape=full_scene_shape,
                    )
                    proj_bev = proj_v if proj_bev is None else torch.maximum(proj_bev, proj_v)
            decoded_bev = self.decoder(proj_bev)
            return decoded_bev.unsqueeze(1)  # [B, 1, 1, H_full, W_full]

        # --- TRAINING PATH: PATCH LOOP ---
        patch_num = hw_random.shape[-1]
        projected_patches = []
        for p in range(patch_num):
            hw_offset = hw_random[:, :, p]
            proj_bev = self.projector(features, camera_paras, wld_map_paras, hw_offset)
            decoded_bev = self.decoder(proj_bev)
            projected_patches.append(decoded_bev)

        final_output = torch.stack(projected_patches, dim=1)
        return final_output


def create_ground_truth_heatmap(gt_world_xy, wld_map_paras, hw_offset, bev_shape=(200, 200)):
    H, W = bev_shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    if gt_world_xy.dim() == 3:
        gt_world_xy = gt_world_xy.squeeze(0)
    if wld_map_paras.dim() == 2:
        wld_map_paras = wld_map_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)

    num_people = gt_world_xy.shape[0]
    if num_people == 0:
        return torch.zeros((1, 1, H, W))

    r = wld_map_paras[0].item()
    a = wld_map_paras[1].item()
    b = wld_map_paras[2].item()
    w_min = wld_map_paras[6].item()
    h_min = wld_map_paras[7].item()

    # Extract the offset for this specific patch
    h_start = hw_offset[0].item()
    w_start = hw_offset[1].item()

    for i in range(num_people):
        x, y = gt_world_xy[i, 0].item(), gt_world_xy[i, 1].item()

        # 1. Map to the full-scene coordinate
        col_full = int(round((x - w_min + a) * r))
        row_full = int(round((y - h_min + b) * r))

        # 2. Subtract the patch offset
        col = col_full - w_start
        row = row_full - h_start

        # 3. Only stamp the person if they fall inside this specific patch
        if 0 <= row < H and 0 <= col < W:
            heatmap[row, col] += 1.0

    return torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)


# ==========================================
# PAPER PROTOCOL (SynMVCrowd, IJCV 2026, "Implementation details")
# ==========================================
# - 30 train / 10 val scenes; per epoch and scene: 10 random frames x 5 random view samplings (5 views each)
#   -> 1500 iterations per epoch, batch size 1, 3 random 200x200 ground-plane patches per sample
# - SGD, momentum 0.9, weight decay 1e-4, max lr 1e-4, LambdaLR schedule, ~200 epochs
# - loss: GaussianMSE_norm on the world map (scale 100) -- "Baseline (MSE)" -- or, with --loss ot,
#   the original unbalanced OT loss (losses/ot_loss/genloss_synmvcrowd.py, 'ed' cost, 100 mass per
#   person), summed over the patches and divided by their number -- "Baseline (OT)".
#   Both yield maps that sum to count * 100; evaluation on the whole scene map
# - OT recipe used here (deviates from the paper text, follows the released code): Adam lr 1e-5 with a
#   constant rate (--optimizer adam --lr_schedule constant; SGD diverges on the OT loss), initialised
#   from the trained MSE baseline (--init_from), at most 100 epochs with early stopping (--patience 25)
# Deliberate deviation (shared with the proposed model): the 2D-pretrained backbone stays FROZEN,
# so the paper's joint image-level loss term has nothing to train and is dropped.
# The paper names LambdaLR but not its lambda; this uses the warmup + cosine lambda of MVDeTr, whose
# trainer the SynMVCrowd code is built from (it steps LambdaLR per iteration with a fractional epoch).
SCALE = 100
MAP_SIGMA, MAP_KERNEL_SIZE = 5, 20  # frameDataset: map_sigma = 20 / grid_reduce(4), map_kernel_size = 20


class PaperEpochSampler(Sampler):
    """Each epoch: per scene, `frames_per_scene` random frames, each drawn `view_samplings` times.
    The dataset samples a fresh random set of views on every __getitem__, so repeating a frame index
    gives a new view combination."""

    def __init__(self, scene_datasets, frames_per_scene=10, view_samplings=5):
        self.frames_per_scene = frames_per_scene
        self.view_samplings = view_samplings
        self.offsets, self.lengths = [], []
        offset = 0
        for ds in scene_datasets:
            self.offsets.append(offset)
            self.lengths.append(len(ds))
            offset += len(ds)

    def __iter__(self):
        indices = []
        for offset, length in zip(self.offsets, self.lengths):
            frames = random.sample(range(length), min(self.frames_per_scene, length))
            for f in frames:
                indices.extend([offset + f] * self.view_samplings)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(min(self.frames_per_scene, n) * self.view_samplings for n in self.lengths)


def warmup_cosine_lambda(epochs, warmup_ratio=0.1):
    """MVDeTr's LambdaLR lambda, evaluated on a fractional epoch."""
    warmup_epochs = warmup_ratio * epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return (math.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * math.pi) + 1) / 2

    return lr_lambda


def load_split(split, excluded_scenes):
    split_dir = os.path.join(src_dir, "datasets", "SynMVCrowd", "data", split)
    datasets = []
    for scene in sorted(os.listdir(split_dir)):
        scene_dir = os.path.join(split_dir, scene)
        if not os.path.isdir(scene_dir):
            continue
        if scene in excluded_scenes:
            print(f"[EXCLUDED] {split}/{scene}: {excluded_scenes[scene]}")
            continue
        # camera_ids=None: all cameras of the scene; 5 are drawn at random per sample
        datasets.append(SynMVCrowdDataset(scene_dir=scene_dir, camera_ids=None, num_sampled_cams=5, load_refs=False))
    return datasets


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def validate(model, val_datasets, device, frames_per_scene, view_samplings, seed, num_workers):
    """Whole-scene evaluation as in the paper: per frame, `view_samplings` random 5-view draws.
    Frame subset and view draws are seeded, so every validation sees the same samples."""
    model.eval()
    rng = np.random.RandomState(seed)
    indices, offset = [], 0
    for ds in val_datasets:
        n = min(frames_per_scene, len(ds)) if frames_per_scene else len(ds)
        frames = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
        for f in frames:
            indices.extend([offset + f] * view_samplings)
        offset += len(ds)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=1,
        sampler=indices,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    abs_err, sq_err, nae = [], [], []
    for sample in tqdm(loader, desc="[Val]", mininterval=10):
        images = sample["images"].to(device)
        cam_paras = sample["camera_paras"].to(device)
        wld_paras = sample["wld_map_paras"].to(device)
        h_full, w_full = int(wld_paras[0, 3].item()), int(wld_paras[0, 4].item())
        pred = model(images, cam_paras, wld_paras, full_scene_shape=(h_full, w_full))
        pred_count = pred.sum().item() / SCALE
        gt_count = sample["gt_world_xy"].shape[1]
        err = pred_count - gt_count
        abs_err.append(abs(err))
        sq_err.append(err**2)
        nae.append(min(abs(err) / gt_count, 1.0) if gt_count > 0 else 0.0)  # clamped at 1 as in the original
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(sq_err))),
        "nae": float(np.mean(nae)),
        "n": len(abs_err),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"--- Baseline ({args.loss.upper()}) training (SynMVCrowd paper protocol, "
        f"{'trainable' if args.train_backbone else 'frozen'} backbone) on {device} ---"
    )
    print(vars(args))

    run_dir = os.path.join(src_dir, "runs", "baseline", args.run_name or datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # 1. Data
    excluded_scenes = load_excluded_scenes()
    train_datasets = load_split("train", excluded_scenes)
    val_datasets = load_split("validate", excluded_scenes)
    train_sampler = PaperEpochSampler(train_datasets, args.frames_per_scene, args.view_samplings)
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=1,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=args.num_workers > 0,
    )
    iters_per_epoch = len(train_sampler)
    print(f"{len(train_datasets)} train scenes -> {iters_per_epoch} iterations/epoch | {len(val_datasets)} val scenes")

    # 2. Model, loss, optimizer, schedule
    model = BaselineModel(num_cam=5, pretrained_backbone_path=args.backbone, train_backbone=args.train_backbone).to(
        device
    )
    criterion = GaussianMSE_norm(scale=SCALE).to(device)
    ot_criterion = SynMVCrowdGeneralizedLoss(cost="ed", scale=SCALE)
    map_kernel = get_map_kernel(sigma=MAP_SIGMA, kernel_size=MAP_KERNEL_SIZE).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":  # paper text
        optimizer = optim.SGD(trainable, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:  # released main.py: Adam(lr=1e-5, weight_decay=1e-4); needed for the OT loss, whose gradients
        # are ~1000x the MSE loss's and make plain SGD diverge
        optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        lr_lambda = warmup_cosine_lambda(args.epochs, args.warmup_ratio)
        # stepped once per iteration; the lambda sees the fractional epoch (iteration / iters_per_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda it: lr_lambda(it / iters_per_epoch))
    else:
        # released main.py trains without a scheduler; a constant rate also makes early stopping
        # meaningful (a cosine schedule cut short never reaches its low-lr phase)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda it: 1.0)

    if args.init_from and not args.resume:
        # Like the original main.py, which loads a fully trained checkpoint into the WHOLE model
        # (decoder included) before training: the OT variant fine-tunes a working density map
        # instead of starting the decoder from random weights, where the OT loss diverges.
        init_state = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init_state.get("model_state_dict", init_state))
        print(f"Initialised the full model from {args.init_from} (epoch {init_state.get('epoch', '?')})")

    start_epoch, best_mae, best_epoch = 1, float("inf"), 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch, best_mae = ckpt["epoch"] + 1, ckpt.get("best_val_mae", float("inf"))
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, best val MAE {best_mae:.2f} at epoch {best_epoch})")

    for epoch in range(start_epoch, args.epochs + 1):
        # ==========================================
        #               TRAINING LOOP
        # ==========================================
        model.train()
        t0 = time.time()
        losses, abs_errs, skipped = [], [], 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", mininterval=10)
        for idx, sample in enumerate(train_pbar):
            images = sample["images"].to(device, non_blocking=True)
            cam_paras = sample["camera_paras"].to(device)
            wld_paras = sample["wld_map_paras"].to(device)
            hw_random = sample["hw_random"].to(device)
            # [B=1, P, H, W] -> [P, 1, H, W]: all patches of a sample share one GaussianMSE_norm call, as in the original
            target = sample["density_patches"].to(device).permute(1, 0, 2, 3).float()

            pred = model(images, cam_paras, wld_paras, hw_random)[0]  # [P, 1, H, W]
            if args.loss == "mse":
                loss, map_gt_conv = criterion(pred, target, map_kernel)
                gt_count = map_gt_conv.sum().item() / SCALE
            else:
                # original trainer: OT loss per patch against its GT points, summed, divided by B * P
                loss = sum(ot_criterion(pred[p : p + 1], patch_points(target[p, 0])) for p in range(pred.shape[0]))
                loss = loss / pred.shape[0]
                gt_count = target.sum().item()

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at iteration {idx}, skipping update")
                skipped += 1
                scheduler.step()
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_count = pred.sum().item() / SCALE
            losses.append(loss.item())
            abs_errs.append(abs(pred_count - gt_count))
            train_pbar.set_postfix(
                {
                    "loss": f"{np.mean(losses):.4f}",
                    "mae": f"{np.mean(abs_errs):.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                },
                refresh=False,
            )

        train_loss, train_mae = float(np.mean(losses)), float(np.mean(abs_errs))
        print(
            f"\n>>> TRAIN EPOCH {epoch} | loss {train_loss:.4f} | patch MAE {train_mae:.2f} | "
            f"skipped {skipped} | lr {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t0:.0f}s"
        )

        # ==========================================
        #              VALIDATION LOOP
        # ==========================================
        val = None
        if epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            val = validate(
                model,
                val_datasets,
                device,
                args.val_frames_per_scene,
                args.val_view_samplings,
                args.val_seed,
                args.num_workers,
            )
            print(
                f">>> VAL EPOCH {epoch} | MAE {val['mae']:.2f} | RMSE {val['rmse']:.2f} | NAE {val['nae']:.3f} | "
                f"{val['n']} samples | {time.time() - t1:.0f}s"
            )

        # ==========================================
        #               SAVE CHECKPOINTS
        # ==========================================
        if val is not None and val["mae"] < best_mae:
            best_mae, best_epoch = val["mae"], epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val": val,
            "best_val_mae": best_mae,
            "best_epoch": best_epoch,
            "args": vars(args),
        }
        if val is not None and best_epoch == epoch:
            torch.save(checkpoint, os.path.join(run_dir, "best.pth"))
            print(f"Saved new best model (val MAE {best_mae:.2f})")
        torch.save(checkpoint, os.path.join(run_dir, "last.pth"))

        # Early stopping, checked at validation epochs: stop once `patience` epochs have passed
        # without a new best val MAE (with --val_every 5 and --patience 25: 5 validations in a row)
        if args.patience and val is not None and epoch - best_epoch >= args.patience:
            print(
                f">>> EARLY STOP at epoch {epoch}: no val MAE improvement for {epoch - best_epoch} epochs "
                f"(best {best_mae:.2f} at epoch {best_epoch}, saved as best.pth)"
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SynMVCrowd homography baseline (paper protocol)")
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        help="Path to the stride-8 resnet18_2d_pretrained_epX.pth from train_2d_pretrain.py.",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Train the backbone jointly (weights and BatchNorm stats, same lr) as in the "
        "released main.py, instead of keeping it frozen.",
    )
    parser.add_argument("--resume", type=str, default=None, help="last.pth of a run to resume from.")
    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Checkpoint whose full model weights start the run (no optimizer state), e.g. the "
        "MSE baseline best.pth for --loss ot. Ignored when resuming.",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Folder name under runs/baseline/ (default: timestamp)."
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "ot"],
        help="World-map loss: 'mse' = Baseline (MSE), 'ot' = Baseline (OT) as in the paper.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="'sgd' = paper text (lr 1e-4); 'adam' = released code (use --lr 1e-5).",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Maximum learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="'cosine' = warmup + cosine over --epochs (paper's LambdaLR); "
        "'constant' = released code (no scheduler), pairs with --patience.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Fraction of epochs for linear warmup (cosine schedule only)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many epochs without a new best val MAE (0 = off). "
        "Checked at validation epochs, so use a multiple of --val_every.",
    )
    parser.add_argument("--frames_per_scene", type=int, default=10, help="Random frames per scene per epoch.")
    parser.add_argument("--view_samplings", type=int, default=5, help="Random 5-view draws per frame.")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument("--val_frames_per_scene", type=int, default=0, help="0 = all frames (paper).")
    parser.add_argument("--val_view_samplings", type=int, default=5)
    parser.add_argument("--val_seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    train(args)
