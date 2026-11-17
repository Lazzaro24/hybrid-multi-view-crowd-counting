"""
datasets/multiview_detection.py

Shared whole-scene loader for the MVDet-style multi-view detection datasets (Wildtrack, MultiviewX),
returning samples in the same format as SynMVCrowdDataset so the SynMVCrowd-trained models and the
paper-protocol helpers (evaluation/evaluate_paper_protocol.py) run on them unchanged:

- 'images' [N, 3, 720, 1280] (ImageNet-normalised), 'camera_paras' [N, 15] = [fx, fy, u, v] normalised by
  the image width/height (the SynMVCrowd json convention the projector expects) + 5 OpenCV distortion
  coefficients + rvec + tvec in metres.
- 'wld_map_paras' [r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max, d_delta] for the
  annotated ground area only (margins a = b = 0, r = 5 px/m as in SynMVCrowd). People outside that area are
  not annotated, so predictions there could only be counted as false positives.
  min_map_px=(H, W) widens the map by equal ground margins a, b (metres) to at least H x W pixels: the
  proposed models run on 200 x 200 tiles, larger than both maps. Scoring and losses then use only
  'eval_window' [row0, col0, h, w], the annotated area inside the map.
- 'gt_world_xy': the people visible (with a box) in at least one active camera, like SynMVCrowd's
  visible-union GT. With all cameras active this is every annotated person (the MVDet GT).
- 'ref_points_full' [N, H, W, 4, 2] (ref_points=True, for the proposed models): the grid_sample coordinates
  of every map cell at the 4 z-anchors in each active camera, as SynMVCrowdDataset loads them from
  master_refs.pt; computed once per dataset with utils/reference_points.compute_reference_points.

Both datasets annotate FEET at z = 0 (z up), whereas SynMVCrowd annotates heads and the baseline projects
onto the head plane (d_mean = mean head z). d_mean is therefore set to `head_height` above the ground.
Verified 2026-10-22 by reprojecting the position IDs: feet land on the box bottoms (median 3-15 px) and
points at z = +1.75 m on the box tops.
The z-anchors are d_mean + linspace(-d_delta, 1.8 + d_delta, 4) with d_delta = 1.75 m as in every SynMVCrowd
scene (feet, head, +1.78 m, +3.55 m), also when head_height is changed. Z_SIGN corrects the behind-camera check
of compute_reference_points per dataset: +1 for Wildtrack; -1 for MultiviewX, whose calibration gives every
visible person a NEGATIVE camera depth (2026-10-29, all 6 cameras), the pattern of the SynMVCrowd z-sign bug
(Unity, left-handed world). The pixel projection is sign-invariant, so the homography projectors are unaffected.
Checked by scripts/diagnosis/verify_cross_dataset_refs.py.

Subclasses provide the calibration, the frame list and the per-frame annotations.
"""

import math
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.reference_points import compute_reference_points

D_DELTA = 1.75  # z-anchor spacing parameter, as in every SynMVCrowd scene


def read_opencv_xml(path, name):
    """Reads a matrix/vector node of an OpenCV FileStorage xml as a flat float array. Parsed by hand because
    OpenCV 5's FileStorage rejects Wildtrack's extrinsic files (plain-text nodes without <data>)."""
    node = ET.parse(path).getroot().find(name)
    data = node.find("data")
    return np.array((data if data is not None else node).text.split(), dtype=np.float64)


class MultiviewDetectionDataset(Dataset):
    R = 5.0  # BEV pixels per metre (SynMVCrowd baseline resolution, 0.2 m cells)
    IMG_SIZE = (1080, 1920)  # (H, W) of the original images
    NUM_CAMS = None
    X_RANGE = None  # (w_min, w_max) of the annotated ground area, metres
    Y_RANGE = None  # (h_min, h_max)
    TRAIN_RATIO = 0.9  # MVDet split: first 90 % of the frames train, the rest test
    Z_SIGN = 1.0  # sign of the camera depth of visible points (compute_reference_points z_sign)

    def __init__(
        self,
        root,
        split="test",
        num_sampled_cams=None,
        head_height=1.75,
        transform=None,
        min_map_px=None,
        ref_points=False,
    ):
        """num_sampled_cams=None (or >= the number of cameras) uses every camera in every sample;
        otherwise each sample draws that many distinct cameras at random (SynMVCrowd protocol).
        min_map_px=(H, W) widens the map to at least that size; ref_points=True adds 'ref_points_full'."""
        self.root = root
        self.num_sampled_cams = num_sampled_cams
        self.head_height = head_height
        self.margins = self.map_margins(min_map_px)

        self.frames = self.split_frames(split)
        if len(self.frames) == 0:
            raise ValueError(f"{type(self).__name__}: no frames in split {split!r} under {root}")
        self.camera_paras = [self.camera_vector(c) for c in range(self.NUM_CAMS)]
        # {frame: [(x_m, y_m, {cam: (xmin, ymin, xmax, ymax)}), ...]}
        self.people = {f: self.load_people(f) for f in self.frames}
        # [NUM_CAMS, H, W, 4, 2] for every camera; a sample takes the rows of its active cameras
        self.ref_points = (
            compute_reference_points(
                self.wld_map_paras(),
                torch.stack(self.camera_paras),
                img_size=(720, 1280),
                z_sign=[self.Z_SIGN] * self.NUM_CAMS,
            )
            if ref_points
            else None
        )

        if transform is None:
            # Same as SynMVCrowdDataset: resize to 720x1280 and normalise for the ImageNet backbone
            self.transform = transforms.Compose(
                [
                    transforms.Resize((720, 1280)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    # ---- dataset-specific -------------------------------------------------------------------------

    def all_frames(self):
        """Sorted frame ids with annotations."""
        raise NotImplementedError

    def load_calibration(self, cam):
        """Returns K [3, 3] in pixels, dist [5] (k1, k2, p1, p2, k3), rvec [3], tvec [3] in metres."""
        raise NotImplementedError

    def load_people(self, frame):
        """Returns [(x_m, y_m, {cam: (xmin, ymin, xmax, ymax)}), ...]: every annotated person with the
        boxes of the cameras that see them."""
        raise NotImplementedError

    def image_path(self, frame, cam):
        raise NotImplementedError

    # ---- shared -----------------------------------------------------------------------------------

    def split_frames(self, split):
        frames = self.all_frames()
        n_train = int(len(frames) * self.TRAIN_RATIO)
        return {"train": frames[:n_train], "test": frames[n_train:], "all": frames}[split]

    def camera_vector(self, cam):
        K, dist, rvec, tvec = self.load_calibration(cam)
        h, w = self.IMG_SIZE
        flat = [K[0, 0] / w, K[1, 1] / h, K[0, 2] / w, K[1, 2] / h] + list(dist) + list(rvec) + list(tvec)
        return torch.tensor(flat, dtype=torch.float32)

    def annotated_px(self):
        """(h, w) of the annotated ground area in map pixels."""
        (w_min, w_max), (h_min, h_max) = self.X_RANGE, self.Y_RANGE
        return int(round((h_max - h_min) * self.R)), int(round((w_max - w_min) * self.R))

    def map_margins(self, min_map_px):
        """(a, b): the x / y ground margin in metres on each side, whole pixels, so the map is >= min_map_px."""
        if min_map_px is None:
            return 0.0, 0.0
        h, w = self.annotated_px()
        return (
            max(0, math.ceil((min_map_px[1] - w) / 2)) / self.R,
            max(0, math.ceil((min_map_px[0] - h) / 2)) / self.R,
        )

    def eval_window(self):
        """[row0, col0, h, w] of the annotated area inside the map."""
        a, b = self.margins
        return torch.tensor([round(b * self.R), round(a * self.R), *self.annotated_px()])

    def wld_map_paras(self):
        (w_min, w_max), (h_min, h_max) = self.X_RANGE, self.Y_RANGE
        a, b = self.margins
        h, w = self.annotated_px()
        h_actual, w_actual = h + 2 * round(b * self.R), w + 2 * round(a * self.R)
        return torch.tensor(
            [self.R, a, b, h_actual, w_actual, self.head_height, w_min, h_min, w_max, h_max, D_DELTA],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.num_sampled_cams is None or self.num_sampled_cams >= self.NUM_CAMS:
            active_cams = list(range(self.NUM_CAMS))
        else:
            active_cams = sorted(random.sample(range(self.NUM_CAMS), self.num_sampled_cams))

        images = []
        for cam in active_cams:
            images.append(self.transform(Image.open(self.image_path(frame, cam)).convert("RGB")))

        # Visible-union GT and per-camera 2D head points (box top centre, scaled to 1280x720 like SynMVCrowd)
        h, w = self.IMG_SIZE
        people = self.people[frame]
        gt_world_xy = [(x, y) for x, y, boxes in people if any(c in boxes for c in active_cams)]
        gt_2d_xy = []
        for cam in active_cams:
            pts = [
                [(b[0] + b[2]) / 2 * 1280 / w, b[1] * 720 / h]
                for _, _, boxes in people
                for c, b in boxes.items()
                if c == cam
            ]
            gt_2d_xy.append(torch.tensor(pts, dtype=torch.float32).view(-1, 2))

        sample = {
            "snapshot_id": frame,
            "active_cams": torch.tensor(active_cams),
            "images": torch.stack(images, dim=0),
            "camera_paras": torch.stack([self.camera_paras[c] for c in active_cams], dim=0),
            "wld_map_paras": self.wld_map_paras(),
            "eval_window": self.eval_window(),
            "gt_world_xy": torch.tensor(gt_world_xy, dtype=torch.float32).view(-1, 2),
            "gt_2d_xy": gt_2d_xy,
        }
        if self.ref_points is not None:
            sample["ref_points_full"] = self.ref_points[active_cams]
        return sample
