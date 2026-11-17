"""
datasets/Wildtrack/wildtrack.py

Wildtrack (Chavdarova et al., CVPR 2018): 7 real cameras, 1920x1080, 400 annotated frames (ids 0-2000,
step 5), ground area 12 m x 36 m on a 2.5 cm position grid. World units are centimetres; converted to metres.

Expected layout under `root` (src/datasets/Wildtrack/data):
- ds/img/C{cam+1}_{frame:08d}.png, ds/ann/C{cam+1}_{frame:08d}.png.json: the Dataset Ninja (Supervisely)
  export: one json per image with a box and a 'position id' tag per visible person.
- calibrations/intrinsic_zero/intr_{name}.xml, calibrations/extrinsic/extr_{name}.xml: the official
  calibration (intrinsic_zero belongs to the undistorted images). Not part of the Dataset Ninja export;
  taken from the full dataset (huggingface.co/datasets/disl/my_dataset, Wildtrack_dataset_full).

Split as in MVDet (num_frame = 2000): frames < 1800 train (360), frames 1800-1995 test (40); frame 2000 is
only in 'all'.
"""

import json
import os

from datasets.multiview_detection import MultiviewDetectionDataset, read_opencv_xml

CAMERA_NAMES = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
NUM_FRAME = 2000  # MVDet's frame count for the split


def position_to_world(pos):
    """MVDet get_worldgrid_from_pos + get_worldcoord_from_worldgrid, in metres."""
    return -3.0 + 0.025 * (pos % 480), -9.0 + 0.025 * (pos // 480)


class WildtrackDataset(MultiviewDetectionDataset):
    NUM_CAMS = 7
    X_RANGE = (-3.0, 9.0)
    Y_RANGE = (-9.0, 27.0)

    def all_frames(self):
        names = os.listdir(os.path.join(self.root, "ds", "ann"))
        return sorted({int(n.split("_")[1].split(".")[0]) for n in names if n.startswith("C1_")})

    def split_frames(self, split):
        frames = self.all_frames()
        cut = int(NUM_FRAME * self.TRAIN_RATIO)
        return {
            "train": [f for f in frames if f < cut],
            "test": [f for f in frames if cut <= f < NUM_FRAME],
            "all": frames,
        }[split]

    def load_calibration(self, cam):
        name = CAMERA_NAMES[cam]
        intr = os.path.join(self.root, "calibrations", "intrinsic_zero", f"intr_{name}.xml")
        extr = os.path.join(self.root, "calibrations", "extrinsic", f"extr_{name}.xml")
        K = read_opencv_xml(intr, "camera_matrix").reshape(3, 3)
        dist = read_opencv_xml(intr, "distortion_coefficients")
        return K, dist, read_opencv_xml(extr, "rvec"), read_opencv_xml(extr, "tvec") / 100.0  # cm -> m

    def load_people(self, frame):
        people = {}
        for cam in range(self.NUM_CAMS):
            with open(os.path.join(self.root, "ds", "ann", f"C{cam + 1}_{frame:08d}.png.json")) as f:
                ann = json.load(f)
            for obj in ann["objects"]:
                pos = next(t["value"] for t in obj["tags"] if t["name"] == "position id")
                (x0, y0), (x1, y1) = obj["points"]["exterior"]
                people.setdefault(pos, {})[cam] = (x0, y0, x1, y1)
        return [(*position_to_world(pos), boxes) for pos, boxes in people.items()]

    def image_path(self, frame, cam):
        return os.path.join(self.root, "ds", "img", f"C{cam + 1}_{frame:08d}.png")
