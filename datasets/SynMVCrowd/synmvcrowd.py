import os
import sys
import json
import random
import torch
from collections import Counter
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# The shared exclusion loaders live in src/utils/exclusions.py. Every caller of this
# class (training, evaluate.py, cache scripts, diagnostics) then gets the same per-frame
# exclusion behavior, instead of only the training script applying it.
# This file sits at src/datasets/SynMVCrowd/synmvcrowd.py, so src/ is three levels up. The
# fallback keeps the import working even for scripts that did not put src/ on sys.path.
try:
    from utils.exclusions import load_excluded_frames
    from utils.weather import load_weather_index
except ImportError:
    _SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC_DIR not in sys.path:
        sys.path.append(_SRC_DIR)
    from utils.exclusions import load_excluded_frames
    from utils.weather import load_weather_index

# Absolute sanity floor for wld_map_paras' r0. Every corrupted frame found by the
# corpus sweep (735 flagged checks across 8 scenes) had r0 in 0.19-0.56, while healthy
# scenes sit at roughly 2.7-8 (scene_71, excluded wholesale, was 0.456). Anything below 1.0
# means the world-extent computation below explodes (h_actual/w_actual up to ~15,000),
# which previously surfaced as a ~3.2 GB single-sample allocation and an OOM crash.
MIN_VALID_R0 = 1.0


def _norm_weather(value):
    """Normalises a raw JSON weather value: None -> 'UNKNOWN' (field absent), '' -> 'EMPTY', else upper-case."""
    if value is None:
        return "UNKNOWN"
    value = str(value).strip()
    return value.upper() if value else "EMPTY"


class NoFramesAfterFilter(ValueError):
    """Raised when a scene has no frames left after the weather filter (callers may skip the scene)."""


class SynMVCrowdDataset(Dataset):
    def __init__(
        self,
        scene_dir,
        camera_ids=None,
        num_sampled_cams=5,
        transform=None,
        cropped_size=(200, 200),
        patch_num=3,
        is_caching=False,
        allowed_weather=None,
        load_refs=True,
        refs_file="master_refs.pt",
    ):
        self.scene_dir = scene_dir
        self.num_sampled_cams = num_sampled_cams
        self.cropped_size = cropped_size
        self.patch_num = patch_num
        self.is_caching = is_caching

        # 1. Gather all snapshots
        snapshot_folders = [
            d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d)) and d.isdigit()
        ]
        all_snapshots = sorted(snapshot_folders, key=lambda x: int(x))

        # 1b. Drop per-frame corrupted snapshots (corrupted wld_map_paras; see excluded_frames.json)
        scene_name = os.path.basename(os.path.normpath(scene_dir))
        bad_ids = load_excluded_frames().get(scene_name, set())

        # Guard against a stale/mismatched exclusion list (IDs that don't exist on disk).
        # Checked BEFORE filtering so a typo fails loudly instead of silently excluding nothing.
        missing = bad_ids - {int(s) for s in all_snapshots}
        if missing:
            raise ValueError(f"{scene_name}: excluded_frames.json lists snapshot IDs not on disk: {sorted(missing)}")

        self.snapshots = [s for s in all_snapshots if int(s) not in bad_ids]
        self.num_excluded_frames = len(all_snapshots) - len(self.snapshots)
        if len(self.snapshots) == 0:
            raise ValueError(f"{scene_name}: all snapshots excluded.")

        # 1c. Optional weather filter + per-scene weather composition. Both need
        # src/utils/weather_index.json (build_weather_index.py). Without the index, weather is
        # still logged per sample from the JSON ('weather' key) but filtering is unavailable.
        weather_index = load_weather_index().get(scene_name, {})
        self.num_weather_filtered = 0
        if allowed_weather is not None:
            if not weather_index:
                raise FileNotFoundError(
                    f"{scene_name}: allowed_weather={list(allowed_weather)} needs utils/weather_index.json "
                    f"(run utils/build_weather_index.py first)."
                )
            allowed = {str(w).upper() for w in allowed_weather}
            before = len(self.snapshots)
            self.snapshots = [
                s for s in self.snapshots if _norm_weather(weather_index.get(s, {}).get("weather")) in allowed
            ]
            self.num_weather_filtered = before - len(self.snapshots)
            if len(self.snapshots) == 0:
                raise NoFramesAfterFilter(f"{scene_name}: no frames with weather in {sorted(allowed)}.")
        self.weather_counts = (
            dict(Counter(_norm_weather(weather_index.get(s, {}).get("weather")) for s in self.snapshots))
            if weather_index
            else {}
        )

        # 2. Find ALL available cameras in the scene
        first_snap = str(self.snapshots[0])
        sample_json_dir = os.path.join(self.scene_dir, first_snap, "jsons")

        available_cams = []
        if os.path.exists(sample_json_dir):
            for f in os.listdir(sample_json_dir):
                if f.endswith(".json"):
                    available_cams.append(int(f.split(".")[0]))

        self.all_available_cams = sorted(available_cams)

        if camera_ids is None:
            self.camera_ids = self.all_available_cams
        else:
            self.camera_ids = camera_ids

        if len(self.camera_ids) == 0:
            raise ValueError(f"CRITICAL: camera_ids is empty for {self.scene_dir}! Check optimal_cameras.json.")

        # 3. Load and safely slice the master tensor (load_refs=False skips it for models that don't
        # use reference points, e.g. the homography baseline -- ~250 MB per scene with all 50 cameras)
        if not self.is_caching and load_refs:
            # refs_file: e.g. master_refs_k8.pt (8 feet-to-head anchors with terrain margins, stored in float16)
            master_refs_path = os.path.join(self.scene_dir, refs_file)
            if not os.path.exists(master_refs_path):
                raise FileNotFoundError(f"Missing {master_refs_path}. Run cache script first!")

            loaded_master = torch.load(master_refs_path, weights_only=True)
            indices = [self.all_available_cams.index(c) for c in self.camera_ids]
            self.active_refs = loaded_master[indices].clone()

            del loaded_master

        if transform is None:
            # Transform used for ImageNet pre-trained models, resizing to 720x1280 and normalizing
            self.transform = transforms.Compose(
                [
                    transforms.Resize((720, 1280)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        snapshot_id = self.snapshots[idx]

        valid_cams = []
        for cam_id in self.camera_ids:
            img_path = os.path.join(self.scene_dir, str(snapshot_id), "jpgs", f"{cam_id}.jpg")
            json_path = os.path.join(self.scene_dir, str(snapshot_id), "jsons", f"{cam_id}.json")

            if os.path.exists(img_path) and os.path.exists(json_path):
                valid_cams.append(cam_id)

        # Fallback 1: If the folder is completely empty or corrupted, skip to the next frame
        if len(valid_cams) == 0:
            return self.__getitem__((idx + 1) % len(self))

        if getattr(self, "is_caching", False):
            active_cams = sorted(valid_cams)
        else:
            # Fallback 2: Normal training sampling
            if len(valid_cams) < self.num_sampled_cams:
                active_cams = random.choices(valid_cams, k=self.num_sampled_cams)
            else:
                active_cams = random.sample(valid_cams, self.num_sampled_cams)

        images = []
        camera_paras = []

        ground_truth_xy = None
        gt_2d_xy_all_cams = []
        # World GT as in the SynMVCrowd paper code: the union (by person idx) of people visible
        # (pixel != None) in at least one of the active cameras. Every view's image_info lists ALL
        # people in the scene, so taking one view's list unfiltered would include people no camera sees.
        visible_world_xy = {}

        wld_map_paras = None
        hw_random = None
        density_patches = None
        weather = "UNKNOWN"

        for cam_id in active_cams:
            img_path = os.path.join(self.scene_dir, str(snapshot_id), "jpgs", f"{cam_id}.jpg")
            json_path = os.path.join(self.scene_dir, str(snapshot_id), "jsons", f"{cam_id}.json")

            img = Image.open(img_path).convert("RGB")
            # Get original image dimensions for 2D scaling before transform
            orig_w, orig_h = img.size

            if self.transform:
                img = self.transform(img)
            images.append(img)

            with open(json_path, "r") as f:
                cam_data = json.load(f)

            # Camera intrinsic parameters
            cam_mat = cam_data["cameraMatrix"]
            fx, fy = cam_mat[0][0], cam_mat[1][1]
            u, v = cam_mat[0][2], cam_mat[1][2]

            # Camera extrinsic parameters
            distCoeffs = cam_data["distCoeffs"]
            rvec = cam_data["rvec"]
            tvec = cam_data["tvec"]

            flat_paras = [fx, fy, u, v] + distCoeffs + rvec + tvec
            camera_paras.append(torch.tensor(flat_paras, dtype=torch.float32))

            coords_2d = []
            for person in cam_data["image_info"]:
                # 'pixel' ([row, col]) is None when the head is outside this view
                if person.get("pixel") is not None:
                    # (x, y) at the 720x1280 input resolution
                    x_2d = person["pixel"][1] * (1280 / orig_w)
                    y_2d = person["pixel"][0] * (720 / orig_h)
                    coords_2d.append([x_2d, y_2d])
                    visible_world_xy.setdefault(person["idx"], person["world"][:2])

            # Empty tensor when nobody is in view
            gt_2d_xy_all_cams.append(torch.tensor(coords_2d, dtype=torch.float32))

            if wld_map_paras is None:
                weather = _norm_weather(cam_data.get("weather"))  # same for (almost) every camera of a frame

                # World map parameters
                raw_paras = cam_data["wld_map_paras"]
                s, r0, a0, b0, h4, w4, d_delta, d_mean, w_min, h_min = raw_paras

                # Runtime backstop for per-frame corruption. excluded_frames.json should
                # already have removed every known-bad frame, so hitting this means a NEW bad
                # frame (or a stale exclusion list). Fail here with a precise message rather
                # than letting h_actual/w_actual blow up into a multi-GB allocation below.
                if r0 < MIN_VALID_R0:
                    raise ValueError(
                        f"Corrupted wld_map_paras in {os.path.basename(os.path.normpath(self.scene_dir))} "
                        f"snapshot {snapshot_id} camera {cam_id}: r0={r0:.4f} < {MIN_VALID_R0}. "
                        f"Add this snapshot to src/utils/excluded_frames.json (and re-run the sweep "
                        f"for the rest of the scene)."
                    )

                # Baseline constants for resolution scaling
                r, a, b = 5.0, 5.0, 5.0

                # Conversion from pixel to real-world coordinates
                w_max = (4 * w4 - 2 * a0) / r0 + w_min
                h_max = (4 * h4 - 2 * b0) / r0 + h_min

                w_actual = int((w_max - w_min + 2 * a) * r)
                h_actual = int((h_max - h_min + 2 * b) * r)

                w_actual = max(w_actual, self.cropped_size[1] + self.patch_num)
                h_actual = max(h_actual, self.cropped_size[0] + self.patch_num)

                h_range = range(0, h_actual - self.cropped_size[0] + 1)
                w_range = range(0, w_actual - self.cropped_size[1] + 1)

                h_random = random.sample(h_range, k=self.patch_num)
                w_random = random.sample(w_range, k=self.patch_num)
                hw_random = torch.tensor([h_random, w_random], dtype=torch.int32)

                wld_map_paras = torch.tensor(
                    [r, a, b, h_actual, w_actual, d_mean, w_min, h_min, w_max, h_max, d_delta], dtype=torch.float32
                )

        # Visible-union world GT (needs every active camera, hence built after the loop)
        ground_truth_xy = torch.tensor(list(visible_world_xy.values()), dtype=torch.float32).view(-1, 2)

        # Build Full Map & Extract Patches
        whole_plane_map = torch.zeros((h_actual, w_actual), dtype=torch.float32)

        if len(ground_truth_xy) > 0:
            x_pixels = ((ground_truth_xy[:, 0] - w_min + a) * r).to(torch.long)
            y_pixels = ((ground_truth_xy[:, 1] - h_min + b) * r).to(torch.long)

            valid_mask = (x_pixels >= 0) & (x_pixels < w_actual) & (y_pixels >= 0) & (y_pixels < h_actual)
            x_pixels = x_pixels[valid_mask]
            y_pixels = y_pixels[valid_mask]

            whole_plane_map[y_pixels, x_pixels] = 1.0

        patches = []
        for p in range(self.patch_num):
            h_start = hw_random[0, p].item()
            w_start = hw_random[1, p].item()

            patch = whole_plane_map[h_start : h_start + self.cropped_size[0], w_start : w_start + self.cropped_size[1]]
            patches.append(patch)

        density_patches = torch.stack(patches, dim=0)

        images_tensor = torch.stack(images, dim=0)
        camera_paras_tensor = torch.stack(camera_paras, dim=0)

        return_dict = {
            "snapshot_id": snapshot_id,
            "weather": weather,
            "images": images_tensor,
            "camera_paras": camera_paras_tensor,
            "wld_map_paras": wld_map_paras,
            "hw_random": hw_random,
            "density_patches": density_patches,
            "gt_world_xy": ground_truth_xy,
            "gt_2d_xy": gt_2d_xy_all_cams,
        }

        if hasattr(self, "active_refs"):
            # 1. Sync the reference points to exactly match the randomly sampled cameras
            active_indices = [self.camera_ids.index(c) for c in active_cams]
            batch_refs = self.active_refs[active_indices].float()  # float16 caches are kept compact in RAM

            c, ref_h, ref_w, z, coords = batch_refs.shape

            # If the physical scene is smaller than our crop requirement, pad it.
            if h_actual > ref_h or w_actual > ref_w:
                pad_h = max(0, h_actual - ref_h)
                pad_w = max(0, w_actual - ref_w)

                temp_refs = batch_refs.permute(0, 3, 4, 1, 2)
                temp_refs = temp_refs.reshape(c * z * coords, 1, ref_h, ref_w)

                temp_refs = torch.nn.functional.pad(temp_refs, (0, pad_w, 0, pad_h), mode="replicate")

                temp_refs = temp_refs.reshape(c, z, coords, ref_h + pad_h, ref_w + pad_w)
                padded_refs = temp_refs.permute(0, 3, 4, 1, 2)

                return_dict["ref_points_full"] = padded_refs
            else:
                return_dict["ref_points_full"] = batch_refs

        return return_dict
