"""
datasets/MultiviewX/multiviewx.py

MultiviewX (Hou et al., ECCV 2020, MVDet): 6 synthetic cameras, 1920x1080, 400 frames, ground area
25 m x 16 m on a 2.5 cm position grid. World units are metres.

Expected layout under `root` (src/datasets/MultiviewX/data): Image_subsets/C{cam+1}/{frame:04d}.png,
annotations_positions/{frame:05d}.json (MVDet format; xmin = -1 marks a camera that doesn't see the
person), calibrations/intrinsic/intr_Camera{cam+1}.xml, calibrations/extrinsic/extr_Camera{cam+1}.xml.

Split as in MVDet: frames 0-359 train, 360-399 test.
"""

import json
import os

from datasets.multiview_detection import MultiviewDetectionDataset, read_opencv_xml


def position_to_world(pos):
    """MVDet get_worldgrid_from_pos + get_worldcoord_from_worldgrid, in metres."""
    return 0.025 * (pos % 1000), 0.025 * (pos // 1000)


class MultiviewXDataset(MultiviewDetectionDataset):
    NUM_CAMS = 6
    X_RANGE = (0.0, 25.0)
    Y_RANGE = (0.0, 16.0)
    Z_SIGN = -1.0  # every visible person has a negative camera depth (Unity calibration, see the base class)

    def all_frames(self):
        return sorted(
            int(n.split(".")[0])
            for n in os.listdir(os.path.join(self.root, "annotations_positions"))
            if n.endswith(".json")
        )

    def load_calibration(self, cam):
        intr = os.path.join(self.root, "calibrations", "intrinsic", f"intr_Camera{cam + 1}.xml")
        extr = os.path.join(self.root, "calibrations", "extrinsic", f"extr_Camera{cam + 1}.xml")
        K = read_opencv_xml(intr, "camera_matrix").reshape(3, 3)
        dist = read_opencv_xml(intr, "distortion_coefficients")
        return K, dist, read_opencv_xml(extr, "rvec"), read_opencv_xml(extr, "tvec")

    def load_people(self, frame):
        with open(os.path.join(self.root, "annotations_positions", f"{frame:05d}.json")) as f:
            ann = json.load(f)
        people = []
        for person in ann:
            boxes = {
                v["viewNum"]: (v["xmin"], v["ymin"], v["xmax"], v["ymax"]) for v in person["views"] if v["xmin"] != -1
            }
            people.append((*position_to_world(person["positionID"]), boxes))
        return people

    def image_path(self, frame, cam):
        return os.path.join(self.root, "Image_subsets", f"C{cam + 1}", f"{frame:04d}.png")
