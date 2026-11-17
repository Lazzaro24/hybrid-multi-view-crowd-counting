"""
evaluation/metrics/localization.py

Point-detection metrics (Precision/Recall/F1/MODA/MODP) for crowd-counting
density-map outputs, extracted via NMS. MODE-AGNOSTIC BY DESIGN: the same
LocalizationMetricsAggregator.update() call is used whether you're scoring
one 200x200 patch (debugging) or one full scene (results) -- the class has
no notion of "patch" or "scene", only "one prediction map's extracted
points, one set of ground-truth points, optionally one region mask, per
call". The caller (evaluate.py) decides how many times to call it and with
what; this module never needs to know which mode produced its inputs.

IMPORTANT CAVEAT, worth repeating wherever these numbers get reported:
Precision/Recall/F1/MODA/MODP are fundamentally POINT-DETECTION metrics,
but the model's actual output is a continuous density map -- points are
extracted via NMS as a post-hoc step. The dense expert is specifically
designed (wide dilated convolutions -- see head.md/training.md) to blur
nearby people into one indistinguishable blob rather than resolve
individuals, so NMS-based point extraction will under-detect in dense
regions BECAUSE OF THAT DESIGN CHOICE, not necessarily because of a
training defect. Treat MAE/RMSE (counting.py) as the primary,
architecture-appropriate metric; treat this module's output as a
secondary, point-detection-style view with this caveat attached.

COORDINATE CONVENTION (load-bearing, do not silently change): every
function here uses (row, col) order throughout -- extract_peaks_from_heatmap
returns np.argwhere() output, which is (row, col); convert_gt_to_bev_coords
returns [row, col] per point to match. If you introduce a peak extractor
elsewhere that returns (x, y)/(col, row) instead (e.g. a normalized-coordinate
extractor), convert it to (row, col) at the call site before passing into
match_points/classify_region -- do not mix conventions inside this module.
"""

import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# WILDTRACK's own benchmark protocol (Chavdarova et al. 2018) uses a 0.5m
# match radius. Convert to pixels via the scene's own r (pixels-per-meter,
# wld_map_paras[0]) rather than a fixed pixel constant -- a fixed pixel
# threshold means a DIFFERENT real-world distance in every scene with a
# different BEV resolution, which is exactly the "uncalibrated placeholder"
# problem evaluate_held_out.py flagged in its own --match_threshold_px default.
WILDTRACK_MATCH_THRESHOLD_M = 0.5


def match_threshold_px_from_r(r, threshold_m=WILDTRACK_MATCH_THRESHOLD_M):
    """r = wld_map_paras[0].item() for the scene/patch being evaluated.
    Returns the WILDTRACK-standard 0.5m match radius in THIS scene's own
    pixel units. Call this per-scene if r varies across scenes -- don't
    reuse one scene's converted value for another."""
    return threshold_m * float(r)


def extract_peaks_from_heatmap(heatmap, threshold, min_distance=3, is_ot=False):
    """
    Extracts (row, col) peak coordinates via 2D non-maximum suppression.
    Works identically on a [1,1,200,200] patch or a [1,1,H_full,W_full]
    full-scene tensor -- shape-agnostic, which is what makes this reusable
    across both eval modes.

    threshold is on the RAW heatmap scale of whatever was passed in --
    this function does NOT pick a default; the caller must supply a
    threshold derived from documented calibration constants (see
    evaluate.py's peak-threshold setup), not a bare literal. The original
    localization.py's PEAK_THRESHOLD=0.05 module constant is intentionally
    NOT reproduced here -- its provenance (whether it predates or postdates
    the GAUSSIAN_SCALE units-bug fix documented elsewhere in this project)
    could not be confirmed, so it is not safe to silently inherit.
    """
    active_threshold = threshold / 10.0 if is_ot else threshold
    padding = min_distance // 2
    max_pooled = F.max_pool2d(heatmap, kernel_size=min_distance, stride=1, padding=padding)
    peaks = (heatmap == max_pooled) & (heatmap > active_threshold)
    peaks = peaks.squeeze().detach().cpu().numpy()
    return np.argwhere(peaks)  # (row, col)


def match_points(pred_rc, gt_rc, threshold_px):
    """
    Hungarian (optimal one-to-one) matching between predicted and ground-
    truth (row, col) points, disallowing matches beyond threshold_px.

    Returns (tp_distances, fp_idx, fn_idx) -- raw ingredients, NOT a
    pre-averaged MODP. Callers/aggregators must pool tp_distances across
    every call BEFORE averaging. Averaging a per-call mean and then
    averaging THOSE means under-weights calls with many matches relative
    to calls with few -- this is the exact bug found in an earlier
    evaluate.py (sum_modp / matched_batches, averaging per-patch instead
    of per-match). LocalizationMetricsAggregator below pools correctly;
    do not reintroduce a per-call average anywhere upstream of it.
    """
    n_pred, n_gt = len(pred_rc), len(gt_rc)
    if n_pred == 0 and n_gt == 0:
        return [], [], []
    if n_pred == 0:
        return [], [], list(range(n_gt))
    if n_gt == 0:
        return [], list(range(n_pred)), []

    pred_rc = np.asarray(pred_rc, dtype=float)
    gt_rc = np.asarray(gt_rc, dtype=float)
    dist_matrix = np.linalg.norm(pred_rc[:, None, :] - gt_rc[None, :, :], axis=2)

    # linear_sum_assignment needs a finite cost; use a large-but-finite
    # penalty for over-threshold pairs rather than inf, then discard any
    # assignment still over threshold after solving.
    cost_matrix = np.where(dist_matrix <= threshold_px, dist_matrix, 1e6)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp_distances = []
    matched_pred, matched_gt = set(), set()
    for pi, gi in zip(row_ind, col_ind):
        if dist_matrix[pi, gi] <= threshold_px:
            tp_distances.append(float(dist_matrix[pi, gi]))
            matched_pred.add(pi)
            matched_gt.add(gi)

    fp_idx = [i for i in range(n_pred) if i not in matched_pred]
    fn_idx = [i for i in range(n_gt) if i not in matched_gt]
    return tp_distances, fp_idx, fn_idx


def classify_region(rc_points, gt_mask, shape):
    """
    Classifies each (row, col) point as 'dense' or 'sparse' via the oracle
    density-threshold gt_mask -- the SAME mask Stage 2's experts train
    against (head.md/training.md), so this evaluates the model against
    the exact boundary it was supervised with, per the project's own
    reasoning for keeping oracle-mask evaluation around. Ported from
    evaluate_held_out.py unchanged, generalized only in variable naming
    (patch 'bev_shape' -> generic 'shape', since this runs unmodified at
    full-scene resolution too).

    gt_mask: tensor indexable as gt_mask[0, 0, row, col] -> float,
    thresholded at 0.5 (dense if > 0.5). Caller is responsible for
    building this at the correct resolution for whichever mode is active
    -- see evaluate.py's oracle-mask construction, itself an ASSUMPTION
    pending confirmation against train_proposed.py (see module docstring
    in evaluate.py).
    """
    if len(rc_points) == 0:
        return []
    H, W = shape
    regions = []
    for row, col in rc_points:
        r = min(max(int(round(float(row))), 0), H - 1)
        c = min(max(int(round(float(col))), 0), W - 1)
        regions.append("dense" if gt_mask[0, 0, r, c].item() > 0.5 else "sparse")
    return regions


class LocalizationMetricsAggregator:
    """
    Mode-agnostic aggregator. Call .update() once per patch OR once per
    scene -- identical either way, since it only ever sees "one map's
    extracted points vs. one set of ground truth". Pools ALL match
    distances across every call before computing MODP, and sums TP/FP/FN/
    GT across every call before computing Precision/Recall/F1/MODA -- see
    match_points' docstring for why per-call ratios are never averaged.

    Dense/sparse breakdown is OPTIONAL per call: pass gt_mask + shape to
    get it; omit them and only the overall metrics are computed (e.g. if
    a mode/config doesn't have a meaningful oracle mask available).
    """

    def __init__(self, match_threshold_px):
        self.match_threshold_px = match_threshold_px
        self.tp_distances = []
        self.total_fp = 0
        self.total_fn = 0
        self.total_gt = 0
        self.has_region_data = False
        self.dense_gt = self.sparse_gt = 0
        self.dense_tp = self.sparse_tp = 0
        self.dense_fp = self.sparse_fp = 0
        self.dense_fn = self.sparse_fn = 0

    def update(self, pred_rc, gt_rc, gt_mask=None, shape=None):
        tp_dists, fp_idx, fn_idx = match_points(pred_rc, gt_rc, self.match_threshold_px)
        self.tp_distances.extend(tp_dists)
        self.total_fp += len(fp_idx)
        self.total_fn += len(fn_idx)
        self.total_gt += len(gt_rc)

        if gt_mask is not None and shape is not None:
            self.has_region_data = True
            gt_regions = classify_region(gt_rc, gt_mask, shape)
            pred_regions = classify_region(pred_rc, gt_mask, shape)

            d_gt = sum(1 for reg in gt_regions if reg == "dense")
            s_gt = sum(1 for reg in gt_regions if reg == "sparse")
            d_fn = sum(1 for i in fn_idx if gt_regions[i] == "dense")
            s_fn = sum(1 for i in fn_idx if gt_regions[i] == "sparse")
            d_fp = sum(1 for i in fp_idx if pred_regions[i] == "dense")
            s_fp = sum(1 for i in fp_idx if pred_regions[i] == "sparse")

            self.dense_gt += d_gt
            self.sparse_gt += s_gt
            self.dense_tp += d_gt - d_fn
            self.sparse_tp += s_gt - s_fn
            self.dense_fp += d_fp
            self.sparse_fp += s_fp
            self.dense_fn += d_fn
            self.sparse_fn += s_fn

    @staticmethod
    def _prf(tp, fp, fn, gt):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / gt if gt > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def compute(self):
        total_tp = len(self.tp_distances)
        precision, recall, f1 = self._prf(total_tp, self.total_fp, self.total_fn, self.total_gt)
        moda = 1.0 - (self.total_fp + self.total_fn) / self.total_gt if self.total_gt > 0 else float("nan")
        modp = float(np.mean([1.0 - d / self.match_threshold_px for d in self.tp_distances])) if total_tp > 0 else 0.0

        result = {
            "tp": total_tp,
            "fp": self.total_fp,
            "fn": self.total_fn,
            "gt": self.total_gt,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "moda": moda,
            "modp": modp,
        }

        if self.has_region_data:
            dp, dr, df1 = self._prf(self.dense_tp, self.dense_fp, self.dense_fn, self.dense_gt)
            sp, sr, sf1 = self._prf(self.sparse_tp, self.sparse_fp, self.sparse_fn, self.sparse_gt)
            result.update(
                {
                    "dense_precision": dp,
                    "dense_recall": dr,
                    "dense_f1": df1,
                    "dense_gt": self.dense_gt,
                    "sparse_precision": sp,
                    "sparse_recall": sr,
                    "sparse_f1": sf1,
                    "sparse_gt": self.sparse_gt,
                }
            )
        return result


def convert_gt_to_bev_coords(gt_world_xy, wld_paras, hw_offset, shape):
    """
    World (X, Y) -> BEV pixel (row, col), clipped to shape. Unchanged math
    from the existing localization.py/evaluate_models.py -- both already
    agreed on this formula, so it's preserved exactly rather than
    "improved". hw_offset=(0, 0) for full-scene mode (no crop offset);
    non-zero for patch mode. shape=(H, W) -- pass the PATCH shape (e.g.
    200,200) in patch mode or the FULL scene shape in scene mode; this
    function has no opinion on which, it just clips to whatever you give it.
    """
    if gt_world_xy.dim() == 3:
        gt_world_xy = gt_world_xy.squeeze(0)
    if wld_paras.dim() == 2:
        wld_paras = wld_paras.squeeze(0)
    if hw_offset.dim() == 2:
        hw_offset = hw_offset.squeeze(0)

    num_people = gt_world_xy.shape[0]
    if num_people == 0:
        return np.empty((0, 2))

    r, a, b = wld_paras[0].item(), wld_paras[1].item(), wld_paras[2].item()
    w_min, h_min = wld_paras[6].item(), wld_paras[7].item()
    h_start, w_start = hw_offset[0].item(), hw_offset[1].item()
    H, W = shape

    gt_coords = []
    for i in range(num_people):
        x, y = gt_world_xy[i, 0].item(), gt_world_xy[i, 1].item()
        col = int(round((x - w_min + a) * r)) - w_start
        row = int(round((y - h_min + b) * r)) - h_start
        if 0 <= row < H and 0 <= col < W:
            gt_coords.append([row, col])
    return np.array(gt_coords) if len(gt_coords) > 0 else np.empty((0, 2))
