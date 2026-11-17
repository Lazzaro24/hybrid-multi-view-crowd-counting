"""One validation pass over the stage's cached patches: per-stage metrics, the epoch report and the
checkpoint-selection metric.

What each stage reports (all counts in persons per 200 x 200 patch, pooled over the val scenes):
  Stage 1/4  count MAE of the (unified / fused) map, localisation error, loss
  Stage 2    oracle-fused count MAE (or, with STAGE2_UNMASKED, the per-cell routing upper bound and each expert
             alone), dense mass / sparse peak errors, pedestal and shape diagnostics, sparse peak F1
  Stage 3    relative mass error and count MAE of the learned soft routing; with performance routing also the
             hard routing, the upper bound, routing accuracy and regret
  joint      Stage 4's on the fused map, plus the oracle-fused count MAE of the experts and the router's agreement
             with the oracle mask (Tversky loss, IoU, dense-zone share)
"""

import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from . import config as cfg
from .losses import mask_loss, stage2_regions
from .reporting import append_csv_row
from .targets import (
    PatchTargets,
    cell_routing_targets,
    count_matches_np,
    extract_camera_locations,
    extract_peaks_via_nms,
    region_points,
)


class Validation:
    def __init__(self, stage, losses):
        self.stage, self.L = stage, losses
        self.abs_err, self.loc_err, self.mass_err = [], [], []
        self.loss_sum, self.n_patches = 0.0, 0
        self.scene_stats = defaultdict(lambda: defaultdict(list))
        # Stage 2
        self.dense_err, self.sparse_err, self.dense_leak, self.true_out = [], [], [], []
        self.dense_alone_err, self.sparse_alone_err = [], []
        self.f1_tp = self.f1_pred = self.f1_gt = 0
        # pedestal / shape diagnostics, as eval_experts_tier0.py: bg = mean raw dense output on empty pixels,
        # corr_in = pooled Pearson correlation of raw dense output vs true density on in-zone pixels (running sums)
        self.bg_sum = self.bg_count = 0.0
        self.sx = self.sy = self.sxy = self.sx2 = self.sy2 = self.sn = 0.0
        # Stage 3, performance routing
        self.perf = {
            "upper": [],
            "dense": [],
            "sparse": [],
            "hard": [],
            "acc_num": 0.0,
            "acc_den": 0.0,
            "regret": 0.0,
            "dense_share": 0.0,
            "label_share": 0.0,
            "n": 0,
        }
        self.soft_err = []
        # joint run
        self.oracle_err, self.router_loss = [], []
        self.router_inter = self.router_union = self.router_dense_px = self.oracle_dense_px = self.n_px = 0

    @torch.no_grad()
    def run(self, model, samples, epoch, epochs, n_scenes):
        device = self.L.device
        for sample in tqdm(
            samples, desc=f"Stage {self.stage} | Epoch {epoch + 1}/{epochs} [Val, {n_scenes} scenes pooled]"
        ):
            hw_random = sample["hw_random"].to(device)
            wld_paras = sample["wld_map_paras"].to(device)
            camera_paras = sample["camera_paras"].to(device)
            cam_locations = extract_camera_locations(camera_paras).squeeze(0)
            with torch.amp.autocast("cuda"):
                predicted_patches, predicted_masks = model(
                    sample["images"].to(device),
                    sample["ref_points_full"].to(device),
                    hw_random,
                    stage=self.stage,
                    camera_paras=camera_paras,
                    wld_map_paras=wld_paras,
                )
            for p in range(hw_random.shape[-1]):
                self.n_patches += 1
                pred_patch = predicted_patches[:, p, ...].float()
                t = PatchTargets(
                    sample["gt_world_xy"], wld_paras, hw_random[:, :, p], cam_locations, self.L.map_kernel, device
                )
                sb = self.scene_stats[sample["scene_name"]]
                if self.stage == "joint":
                    self._joint(pred_patch, predicted_masks[:, p, ...].float(), t, sb)
                elif self.stage in (1, 4):
                    self._stage14(pred_patch, predicted_masks[:, p, ...].float(), t, sb)
                elif self.stage == 2:
                    self._stage2(pred_patch, t, sb)
                else:
                    self._stage3(model, pred_patch, predicted_masks[:, p, ...].float(), t, sb)
        torch.cuda.empty_cache()  # val pools variably-sized scenes too
        return self

    def _stage14(self, pred_patch, pred_mask, t, sb):
        gt_count = t.gt_count
        pred_count = pred_patch.sum().item() / self.L.count_mae_divisor
        self.abs_err.append(abs(pred_count - gt_count))
        sb["abs_err"].append(abs(pred_count - gt_count))
        sb["gt"].append(gt_count)
        sb["pred"].append(pred_count)
        if gt_count > 0:
            pred_peaks = extract_peaks_via_nms(pred_patch, threshold=self.L.dense_peak_threshold, kernel_size=3)
            gt_points_flat = t.target_dots.squeeze(0).squeeze(0)
            if pred_peaks.shape[0] > 0:
                p2g = torch.cdist(pred_peaks, gt_points_flat).min(dim=1).values.mean().item()
                g2p = torch.cdist(gt_points_flat, pred_peaks).min(dim=1).values.mean().item()
                self.loc_err.append((p2g + g2p) / 2.0)
        loss = self.L.stage1(pred_patch, t) if self.stage == 1 else self.L.stage4(pred_patch, pred_mask, t)
        self.loss_sum += loss.item()

    def _joint(self, pred_patch, pred_mask, t, sb):
        """Fused map (learned routing) as in Stage 4; the experts fused with the oracle mask as in Stage 2; the
        router against the oracle mask (pooled IoU and dense share of the hard mask > 0.5)."""
        K = self.L.kernel_mass
        self._stage14(pred_patch[:, 0:1, ...], pred_mask, t, sb)
        dense, sparse, gt_mask = pred_patch[:, 1:2, ...], pred_patch[:, 2:3, ...], t.gt_mask
        oracle_mass = (dense * gt_mask + sparse * (1.0 - gt_mask)).sum().item() / K
        self.oracle_err.append(abs(oracle_mass - t.map_gt_conv.sum().item() / K))
        self.router_loss.append(mask_loss(torch.clamp(pred_mask, min=1e-6, max=1.0 - 1e-6), gt_mask).item())
        hard, oracle = pred_mask > 0.5, gt_mask > 0.5
        self.router_inter += (hard & oracle).sum().item()
        self.router_union += (hard | oracle).sum().item()
        self.router_dense_px += hard.sum().item()
        self.oracle_dense_px += oracle.sum().item()
        self.n_px += hard.numel()

    def _stage2(self, pred_patch, t, sb):
        K = self.L.kernel_mass
        map_gt_conv = t.map_gt_conv
        dense_region, sparse_region = stage2_regions(t.gt_mask)
        dense_raw, sparse_raw = pred_patch[:, 0:1, ...], pred_patch[:, 1:2, ...]

        empty_pixels = (map_gt_conv < 0.02).float()
        if empty_pixels.sum() > 0:
            self.bg_sum += (dense_raw * empty_pixels).sum().item()
            self.bg_count += empty_pixels.sum().item()
        in_zone = dense_region.bool()
        if in_zone.sum() > 1:
            x_in, y_in = dense_raw[in_zone], map_gt_conv[in_zone]
            self.sx += x_in.sum().item()
            self.sy += y_in.sum().item()
            self.sxy += (x_in * y_in).sum().item()
            self.sx2 += (x_in * x_in).sum().item()
            self.sy2 += (y_in * y_in).sum().item()
            self.sn += x_in.numel()

        dense_target_mass = (map_gt_conv * dense_region).sum().item() / K
        gt_sparse_xy = region_points(t.target_dots, sparse_region) if t.gt_count > 0 else []
        sparse_target_count = len(gt_sparse_xy)

        dense_count = (dense_raw * dense_region).sum().item() / K
        self.dense_leak.append((dense_raw * (1.0 - dense_region)).sum().item() / K)
        self.true_out.append((map_gt_conv * (1.0 - dense_region)).sum().item() / K)
        sparse_points = self.L.sparse_peaks(sparse_raw * sparse_region)
        sparse_count = sparse_points.shape[0]
        pr_xy = (
            sparse_points.cpu().numpy() * np.array([cfg.BEV_SHAPE[1], cfg.BEV_SHAPE[0]])
            if sparse_count
            else np.zeros((0, 2))
        )
        self.f1_tp += count_matches_np(pr_xy, np.array(gt_sparse_xy).reshape(-1, 2), cfg.SPARSE_MATCH_PX)
        self.f1_pred += sparse_count
        self.f1_gt += len(gt_sparse_xy)

        self.dense_err.append(abs(dense_count - dense_target_mass))
        self.sparse_err.append(abs(sparse_count - sparse_target_count))

        combined_target_mass = map_gt_conv.sum().item() / K
        if cfg.STAGE2_UNMASKED:
            # no oracle split to fuse with: the routing upper bound (per cell, the expert closer to the GT), plus
            # each expert counting the whole patch on its own
            ct = cell_routing_targets(dense_raw, sparse_raw, map_gt_conv, cfg.ROUTE_CELL, K)
            combined_pred_mass = torch.where(ct["label"] > 0, ct["dense"], ct["sparse"]).sum().item()
            self.dense_alone_err.append(abs(ct["dense"].sum().item() - combined_target_mass))
            self.sparse_alone_err.append(abs(ct["sparse"].sum().item() - combined_target_mass))
        else:
            gt_mask = t.gt_mask
            combined_pred_mass = (dense_raw * gt_mask + sparse_raw * (1.0 - gt_mask)).sum().item() / K
        self.abs_err.append(abs(combined_pred_mass - combined_target_mass))
        sb["abs_err"].append(abs(combined_pred_mass - combined_target_mass))
        sb["gt"].append(combined_target_mass)
        sb["pred"].append(combined_pred_mass)
        sb["dense_err"].append(abs(dense_count - dense_target_mass))
        sb["sparse_err"].append(abs(sparse_count - sparse_target_count))

    def _stage3(self, model, pred_patch, pred_mask, t, sb):
        K = self.L.kernel_mass
        loss, safe_pred_mask, ct, cell_prob = self.L.stage3(model, pred_patch, pred_mask, t)
        if ct is not None:
            gt_cnt = ct["gt"].sum().item()
            hard = cell_prob > 0.5
            best_err = torch.minimum(ct["err_dense"], ct["err_sparse"])
            chosen_err = torch.where(hard, ct["err_dense"], ct["err_sparse"])
            pv = self.perf
            pv["upper"].append(abs(torch.where(ct["label"] > 0, ct["dense"], ct["sparse"]).sum().item() - gt_cnt))
            pv["dense"].append(abs(ct["dense"].sum().item() - gt_cnt))
            pv["sparse"].append(abs(ct["sparse"].sum().item() - gt_cnt))
            pv["hard"].append(abs(torch.where(hard, ct["dense"], ct["sparse"]).sum().item() - gt_cnt))
            pv["acc_num"] += (ct["weight"] * (hard.float() == ct["label"]).float()).sum().item()
            pv["acc_den"] += ct["weight"].sum().item()
            pv["regret"] += (chosen_err - best_err).sum().item()
            pv["dense_share"] += hard.float().mean().item()
            pv["label_share"] += ct["label"].mean().item()
            pv["n"] += 1
        self.loss_sum += loss.item()

        dense_pred = torch.relu(model.head.dense_expert(pred_patch))
        sparse_pred = model.head.sparse_map(pred_patch)
        learned_mass = (safe_pred_mask * dense_pred + (1.0 - safe_pred_mask) * sparse_pred).sum().item() / K
        true_mass = t.map_gt_conv.sum().item() / K
        rel_err = abs(learned_mass - true_mass) / max(true_mass, cfg.MASS_REL_ERROR_FLOOR)
        self.mass_err.append(rel_err)
        sb["abs_err"].append(abs(learned_mass - true_mass))
        self.soft_err.append(abs(learned_mass - true_mass))
        sb["gt"].append(true_mass)
        sb["pred"].append(learned_mass)
        sb["rel_err"].append(rel_err)

    # --- summaries -------------------------------------------------------------------------------------------------
    @property
    def avg_loss(self):
        return self.loss_sum / self.n_patches if self.n_patches > 0 else 0.0

    @property
    def mae(self):
        return np.mean(self.abs_err) if self.abs_err else 0.0

    @property
    def localization(self):
        return np.mean(self.loc_err) if self.loc_err else 0.0

    def sparse_f1(self):
        prec = self.f1_tp / max(self.f1_pred, 1)
        rec = self.f1_tp / max(self.f1_gt, 1)
        return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec

    def selection_metric(self, fs_res):
        """(metric, name, shown as a percentage) the stage selects its checkpoint on (lower is better)."""
        stage = self.stage
        if stage == 1:
            metric, name = self.avg_loss, "Val Loss"
        elif stage == 4:
            metric, name = self.mae, "Count MAE"
        elif stage == "joint":
            metric, name = self.mae, "Count MAE (fused map, learned routing)"
        elif stage == 3:
            if cfg.STAGE3_SELECT == "soft_mae":
                # what the hybrid actually outputs: count MAE of the learned soft routing
                assert cfg.STAGE3_TARGET == "performance" and self.perf["n"], (
                    "--stage3_select soft_mae needs --stage3_target performance"
                )
                metric, name = float(np.mean(self.soft_err)), "Learned soft-routing count MAE (persons/patch)"
            else:
                metric = np.mean(self.mass_err) if self.mass_err else float("inf")
                name = "Downstream Mass Rel. Error"
        elif cfg.STAGE2_SELECT == "dense":
            # dense expert alone on the whole patch (the dense-capacity control, doc/head_ablation.md §8)
            assert cfg.STAGE2_UNMASKED, "--stage2_select dense needs --stage2_unmasked"
            metric = float(np.mean(self.dense_alone_err))
            name = "Stage 2 dense-alone count MAE (mass-based, pooled val scenes)"
        elif cfg.STAGE2_SELECT == "experts":
            # the two experts Stage 3 routes between, each on the whole patch (the per-cell upper bound is a
            # minimum over two noisy experts and selected outliers)
            assert cfg.STAGE2_UNMASKED, "--stage2_select experts needs --stage2_unmasked"
            metric = 0.5 * (float(np.mean(self.dense_alone_err)) + float(np.mean(self.sparse_alone_err)))
            name = "Stage 2 mean of dense-alone and sparse-alone count MAE (mass-based, pooled val scenes)"
        elif cfg.STAGE2_SELECT == "sparse_f1":
            metric = 1.0 - self.sparse_f1()[0]
            name = "1 - Sparse F1 (peak-matched, sparse region, pooled val scenes)"
        else:
            metric = self.mae
            name = (
                f"Stage 2 routing upper bound MAE (per-{cfg.ROUTE_CELL}px-cell best expert, mass-based, pooled across val scenes)"
                if cfg.STAGE2_UNMASKED
                else "Stage 2 Combined MAE (oracle-routed, mass-based, pooled across val scenes)"
            )
        if fs_res is not None:
            metric = fs_res["mae"]
            name = f"Whole-scene val MAE ({'oracle' if stage == 2 else 'learned'} routing, {fs_res['n']} samples)"
        is_pct = stage == 3 and fs_res is None and cfg.STAGE3_SELECT == "mass"
        return metric, name, is_pct

    def report(self, epoch, train, metric, name, is_pct, n_scenes):
        """The epoch's summary lines (and Stage 2's diagnostics CSV row). train: the epoch's TrainStats."""
        stage = self.stage
        train_loss = train.loss / train.n_batches
        if stage == 2:
            dense_mae, sparse_mae = np.mean(self.dense_err), np.mean(self.sparse_err)
            train_dense_mae = np.mean(train.train_dense_errors) if train.train_dense_errors else 0.0
            train_sparse_mae = np.mean(train.train_sparse_errors) if train.train_sparse_errors else 0.0
            dense_bg = self.bg_sum / self.bg_count if self.bg_count > 0 else float("nan")
            if self.sn > 1:
                mean_x, mean_y = self.sx / self.sn, self.sy / self.sn
                cov = self.sxy / self.sn - mean_x * mean_y
                var_x = self.sx2 / self.sn - mean_x**2
                var_y = self.sy2 / self.sn - mean_y**2
                dense_corr_in = cov / max((var_x * var_y) ** 0.5, 1e-12)
            else:
                dense_corr_in = float("nan")
            f1, prec, rec = self.sparse_f1()
            print(
                f">>> Stage 2 Epoch {epoch + 1} | "
                f"Loss (Tot: {train_loss:.3f} | Dense: {train.dense_loss / train.n_batches:.3f} | "
                f"Sparse: {train.sparse_loss / train.n_batches:.3f}) | "
                f"Count MAE ({'routing upper bound' if cfg.STAGE2_UNMASKED else 'oracle-fused'}, {n_scenes} scenes pooled): {self.mae:.1f} <<<"
            )
            if cfg.STAGE2_UNMASKED:
                print(
                    f"    [val-unmasked] whole patch, mass-based: dense alone MAE {np.mean(self.dense_alone_err):.1f} | "
                    f"sparse alone MAE {np.mean(self.sparse_alone_err):.1f} | routing upper bound (per-{cfg.ROUTE_CELL}px-cell "
                    f"best expert) MAE {self.mae:.1f}"
                )
            print(
                f"    [val-mae] Dense(mass): {dense_mae:.1f} | Sparse(peak): {sparse_mae:.1f} | "
                f"train Dense(mass): {train_dense_mae:.1f} | train Sparse(peak): {train_sparse_mae:.1f}"
            )
            print(
                f"    [val-pedestal] dense bg (mean output on empty pixels): {dense_bg:.3f} persons/px | "
                f"dense corr_in (Pearson, in-zone, shape): {dense_corr_in:.3f} | "
                f"dense mass leaked outside mask: {np.mean(self.dense_leak):.1f} persons/patch "
                f"(true mass there: {np.mean(self.true_out):.1f})"
            )
            print(
                f"    [val-sparse] matched peak F1 {f1:.3f} (P {prec:.2f}, R {rec:.2f}, "
                f"pred/GT {self.f1_pred / max(self.f1_gt, 1):.2f}) | radius {cfg.SPARSE_MATCH_PX:g} px, "
                f"NMS mult {cfg.SPARSE_PEAK_THRESHOLD_MULTIPLIER:g}, kernel {cfg.SPARSE_NMS_KERNEL_SIZE}"
            )
            append_csv_row(
                os.path.join(cfg.VAL_LOG_DIR, f"stage2_diag_{cfg.TIMING_RUN_TAG}.csv"),
                [
                    "run_tag",
                    "epoch",
                    "count_mae_fused",
                    "dense_mae",
                    "sparse_mae",
                    "dense_bg",
                    "dense_corr_in",
                    "dense_leak_mass",
                    "true_out_mass",
                    "sparse_f1",
                    "dense_out_weight",
                ],
                [
                    cfg.TIMING_RUN_TAG,
                    epoch + 1,
                    f"{self.mae:.4f}",
                    f"{dense_mae:.4f}",
                    f"{sparse_mae:.4f}",
                    f"{dense_bg:.4f}",
                    f"{dense_corr_in:.4f}",
                    f"{np.mean(self.dense_leak):.4f}",
                    f"{np.mean(self.true_out):.4f}",
                    f"{f1:.4f}",
                    cfg.DENSE_OUT_WEIGHT,
                ],
            )
        elif stage == 1:
            print(
                f">>> Stage 1 Epoch {epoch + 1} Complete | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {self.avg_loss:.4f} | Count MAE: {self.mae:.4f} | Localization: {self.localization:.4f} | "
                f"{name}: {metric:.4f} <<<"
            )
        elif stage == "joint":
            # count MAEs per patch: fused map with the learned router, experts fused with the oracle mask
            row = {
                "count_mae_fused": self.mae,
                "count_mae_oracle": np.mean(self.oracle_err),
                "router_loss": np.mean(self.router_loss),
                "router_iou": self.router_inter / max(self.router_union, 1),
                "dense_share": self.router_dense_px / max(self.n_px, 1),
                "oracle_share": self.oracle_dense_px / max(self.n_px, 1),
                "val_loss_fused": self.avg_loss,
            }
            print(
                f">>> Joint Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | {name} ({n_scenes} scenes): {metric:.2f} <<<"
            )
            print("    [val-joint] " + " | ".join(f"{k} {v:.3f}" for k, v in row.items()))
            append_csv_row(
                os.path.join(cfg.VAL_LOG_DIR, f"val_joint_{cfg.TIMING_RUN_TAG}.csv"),
                ["run_tag", "epoch", "selection_metric", *row],
                [cfg.TIMING_RUN_TAG, epoch + 1, f"{metric:.4f}", *(f"{v:.4f}" for v in row.values())],
            )
        elif stage == 4:
            print(
                f">>> Stage 4 Epoch {epoch + 1} Complete | Train Loss: {train_loss:.4f} | "
                f"Val Loss ({'OT+MSE' if cfg.OT_WEIGHT > 0.0 else 'MSE only'}, informational): {self.avg_loss:.4f} | "
                f"Localization (informational): {self.localization:.4f} | "
                f"{name} (SELECTION metric, {n_scenes} scenes pooled): {metric:.4f} <<<"
            )
        else:
            print(
                f">>> Stage 3 Epoch {epoch + 1} Complete | Train Loss: {train_loss:.4f} | "
                f"Val Loss ({'weighted routing BCE' if cfg.STAGE3_TARGET == 'performance' else 'Tversky/Focal'}, "
                f"informational): {self.avg_loss:.4f} | "
                f"{name} ({n_scenes} scenes pooled): "
                f"{f'{metric * 100:.1f}%' if is_pct else f'{metric:.2f}'} <<<"
            )
            pv = self.perf
            if cfg.STAGE3_TARGET == "performance" and pv["n"]:
                n_ = pv["n"]
                print(
                    f"    [val-routing] count MAE (persons/patch): learned soft {np.mean(self.soft_err):.1f} | learned hard "
                    f"{np.mean(pv['hard']):.1f} | upper bound {np.mean(pv['upper']):.1f} | dense alone "
                    f"{np.mean(pv['dense']):.1f} | sparse alone {np.mean(pv['sparse']):.1f}"
                )
                print(
                    f"    [val-routing] weighted accuracy {100 * pv['acc_num'] / max(pv['acc_den'], 1e-9):.1f}% | "
                    f"regret {pv['regret'] / n_:.1f} persons/patch (sum over cells of chosen - best error) | "
                    f"cells routed dense {100 * pv['dense_share'] / n_:.1f}% (labels: {100 * pv['label_share'] / n_:.1f}%) | "
                    f"cell {cfg.ROUTE_CELL} px"
                )
