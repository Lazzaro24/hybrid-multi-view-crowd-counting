"""
evaluation/metrics/counting.py

Mass-regression metrics (MAE/MSE/NAE) for crowd-counting density-map
outputs. MODE-AGNOSTIC by the same principle as localization.py: .update()
takes "one prediction, one ground truth" per call, whether that call
represents one patch or one full scene.

Extended from the original CountingMetricsAggregator with OPTIONAL mass
tracking (evaluate_held_out.py's "Mass MAE" concept): pass pred_mass and
target_mass alongside pred_count/gt_count to additionally track error
against the model's OWN blurred-Gaussian training target, not just the
raw ground-truth point count. This separates two failure modes that look
identical in a single "undercounting" number:
  (a) pred_mass ~= target_mass, but target_mass << gt_count: the model is
      matching its own training target well; the gap to gt_count is a
      boundary/kernel-truncation artifact of how ground truth is
      generated (F.conv2d's zero-padding clips a person's Gaussian blob
      near a patch/scene edge), not a learning failure.
  (b) pred_mass << target_mass: genuine underprediction against what the
      model was actually trained to match.

In FULL-SCENE mode this truncation effect should be much smaller than in
patch mode (no interior patch seams to clip Gaussians against -- only the
outer scene boundary remains), but not necessarily zero -- worth checking
empirically whether Mass MAE and raw MAE converge at scene scale rather
than assuming they do.

pred_mass/target_mass are OPTIONAL per call -- omit them and only the
standard raw-count metrics are computed; mass metrics only appear in
.compute()'s output if at least one call supplied them.
"""

import math


class CountingMetricsAggregator:
    def __init__(self):
        self.total_abs_error = 0.0
        self.total_sq_error = 0.0
        self.total_norm_abs_error = 0.0
        self.num_samples = 0

        self.has_mass_data = False
        self.mass_abs_error = 0.0
        self.mass_sq_error = 0.0

    def update(self, pred_count: float, gt_count: float, pred_mass: float = None, target_mass: float = None):
        """
        pred_count/gt_count: the standard crowd-counting quantities --
        pred = final_map.sum() (already scaled to raw-count units by the
        caller, e.g. /GAUSSIAN_SCALE or /kernel_mass_per_person as
        appropriate -- this class does not rescale anything itself),
        gt = raw ground-truth point count.

        pred_mass/target_mass: OPTIONAL. Same predicted quantity as
        pred_count, but compared against the BLURRED GAUSSIAN TARGET MASS
        instead of the raw point count -- see module docstring. Both must
        be in kernel_mass_per_person-normalized units (NOT GAUSSIAN_SCALE
        -- see the units-bug note in evaluate_held_out.py's docstring if
        porting this from there) for the comparison to be meaningful.
        """
        abs_error = abs(pred_count - gt_count)
        self.total_abs_error += abs_error
        self.total_sq_error += abs_error**2
        # epsilon avoids divide-by-zero on an empty ground truth
        self.total_norm_abs_error += abs_error / (gt_count + 1e-6)
        self.num_samples += 1

        if pred_mass is not None and target_mass is not None:
            self.has_mass_data = True
            mass_err = abs(pred_mass - target_mass)
            self.mass_abs_error += mass_err
            self.mass_sq_error += mass_err**2

    def compute(self):
        if self.num_samples == 0:
            return {"MAE": 0.0, "MSE": 0.0, "NAE": 0.0}

        result = {
            "MAE": self.total_abs_error / self.num_samples,
            "MSE": math.sqrt(self.total_sq_error / self.num_samples),
            "NAE": self.total_norm_abs_error / self.num_samples,
        }
        if self.has_mass_data:
            result["Mass_MAE"] = self.mass_abs_error / self.num_samples
            result["Mass_MSE"] = math.sqrt(self.mass_sq_error / self.num_samples)
        return result
