"""Scene registries (one SynMVCrowdDataset per scene, alive for the whole run), train loaders and the fixed
validation sets.

Scenes differ in grid size, so frames of two scenes cannot share a batch: training iterates scene by scene, and
validation is a list of samples materialised once per stage (a DataLoader would redraw the random crop every epoch).
"""

import hashlib
import json
import os
import random
import time
import types
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.SynMVCrowd.synmvcrowd import SynMVCrowdDataset, NoFramesAfterFilter
from losses.gaussian_mse import get_map_kernel
from utils.exclusions import load_excluded_scenes, load_excluded_frames

from . import config as cfg
from .reporting import format_duration
from .targets import oracle_mask_from_conv


def log_and_validate_camera_setup(
    scene_name,
    cam_ids,
    z_sign,
    scene_refs_for_this_run,
    scene_folder,
    min_visible_people=5,
    max_snapshots_to_try=3,
    snapshot_ids=None,
):
    """Startup check of one scene's cameras: cached reference-point coverage (dead-camera bug) and a fresh z_sign
    recalibration against the applied one. Returns the list of problems (empty if none) so the caller can collect
    them across every scene. snapshot_ids: the dataset's filtered snapshots, so no excluded frame is sampled."""
    print("\n" + "=" * 70)
    print(f"CAMERA SETUP: scene={scene_name} | N={len(cam_ids)} | cam_ids={cam_ids}")
    for cam_id, sign in zip(cam_ids, z_sign):
        flag = "  <-- Z-SIGN CORRECTED (flipped)" if sign < 0 else ""
        print(f"    camera {cam_id}: z_sign={sign:+.1f}{flag}")
    print("=" * 70)

    if snapshot_ids is None:
        snapshot_ids = sorted(
            [d for d in os.listdir(scene_folder) if os.path.isdir(os.path.join(scene_folder, d))],
            key=lambda x: int(x) if x.isdigit() else x,
        )

    problems = []

    for cam_idx, cam_id in enumerate(cam_ids):
        cam_refs = scene_refs_for_this_run[cam_idx]

        x, y = cam_refs[..., 0], cam_refs[..., 1]
        valid = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
        pct_coverage = valid.float().mean().item() * 100
        if pct_coverage < 0.5:
            problems.append(
                f"camera {cam_id}: cached coverage is only {pct_coverage:.2f}% "
                f"(expected >0.5%) -- looks like the OLD dead-camera bug is back, "
                f"or master_refs.pt is stale / wrongly indexed."
            )
        print(f"    camera {cam_id}: cached coverage = {pct_coverage:.2f}%")

        all_world_points = []
        rvec, tvec = None, None
        for snap_id in snapshot_ids[:max_snapshots_to_try]:
            json_path = os.path.join(scene_folder, str(snap_id), "jsons", f"{cam_id}.json")
            if not os.path.exists(json_path):
                continue
            with open(json_path, "r") as f:
                cam_data = json.load(f)
            if rvec is None:
                rvec = np.array(cam_data["rvec"], dtype=np.float32)
                tvec = np.array(cam_data["tvec"], dtype=np.float32)
            pts = [
                p["world"] for p in cam_data["image_info"] if p.get("pixel") is not None and p.get("world") is not None
            ]
            all_world_points.extend(pts)
            if len(all_world_points) >= min_visible_people:
                break

        if rvec is not None and len(all_world_points) > 0:
            R, _ = cv2.Rodrigues(rvec)
            z_cam = ((R @ np.array(all_world_points, dtype=np.float32).T).T + tvec)[:, 2]
            fresh_sign = -1.0 if (z_cam < 0).mean() > 0.5 else 1.0
            applied_sign = z_sign[cam_idx]
            if fresh_sign != applied_sign:
                problems.append(
                    f"camera {cam_id}: fresh recalibration says z_sign={fresh_sign:+.1f}, "
                    f"but {applied_sign:+.1f} was applied -- z_sign_corrections_full.json is "
                    f"stale or out of sync with this camera set."
                )

    if problems:
        print(f"\n!!! CAMERA SETUP VALIDATION FAILED for {scene_name} !!!")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"[OK] Camera setup validated for {scene_name} -- no dead cameras, no z_sign mismatches.\n")

    return problems


def validate_excluded_frame_keys():
    """excluded_frames.json is keyed by scene name, and SynMVCrowdDataset silently excludes nothing for a key that
    matches no scene (a typo, or a renamed scene): fail instead. Keys of excluded or test scenes are fine."""
    excluded_frames = load_excluded_frames()
    if not excluded_frames:
        print("[exclusions] No excluded_frames.json found (or empty) -- no per-frame exclusions applied.")
        return excluded_frames

    data_root = os.path.join(cfg.src_dir, "datasets", "SynMVCrowd", "data")
    on_disk = set()
    for split_name in os.listdir(data_root):
        split_path = os.path.join(data_root, split_name)
        if os.path.isdir(split_path):
            on_disk.update(d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d)))

    unknown = sorted(k for k in excluded_frames if k not in on_disk)
    if unknown:
        raise ValueError(f"excluded_frames.json has keys matching no scene folder in any split: {unknown}")

    total = sum(len(v) for v in excluded_frames.values())
    print(
        f"[exclusions] excluded_frames.json: {total} frames across {len(excluded_frames)} scenes "
        f"({', '.join(sorted(excluded_frames, key=lambda s: int(s.split('_')[1])))})."
    )
    return excluded_frames


def build_scene_registry(
    split_dir_name, k, optimal_cameras_config, z_sign_corrections, validate_setup=True, allowed_weather=None
):
    """One SynMVCrowdDataset per scene under data/<split_dir_name>, with that scene's k cameras from
    optimal_cameras_filtered.json. Skips scenes in excluded_scenes.json (per-frame exclusions apply inside the
    dataset). With validate_setup, checks every scene's cameras first and raises once, listing every problem.
    Returns a list of {'scene_name', 'scene_folder', 'cam_ids', 'dataset'}."""
    split_dir = os.path.join(cfg.src_dir, "datasets", "SynMVCrowd", "data", split_dir_name)
    scene_folders = sorted(
        os.path.join(split_dir, d) for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))
    )

    excluded_scenes = load_excluded_scenes()
    registry = []
    all_problems = {}  # scene_name -> problems, collected across every scene before deciding
    for scene_folder in scene_folders:
        scene_name = os.path.basename(scene_folder)

        if scene_name in excluded_scenes:
            print(f"[EXCLUDED] {scene_name}: {excluded_scenes[scene_name]}")
            continue

        cam_ids = optimal_cameras_config.get(scene_name, {}).get(str(k))
        if cam_ids is None:
            print(f"[skip] {scene_name}: no camera selection found for k={k} in optimal_cameras_filtered.json")
            continue

        try:
            dataset = SynMVCrowdDataset(scene_dir=scene_folder, camera_ids=cam_ids, allowed_weather=allowed_weather)
        except NoFramesAfterFilter as e:
            print(f"[skip] {e}")
            continue
        if allowed_weather is not None:
            print(
                f"[WEATHER FILTER] {scene_name}: kept {len(dataset)} frames "
                f"({dataset.num_weather_filtered} removed) -> {dataset.weather_counts}"
            )

        if dataset.num_excluded_frames > 0:
            print(
                f"[EXCLUDED FRAMES] {scene_name}: dropped {dataset.num_excluded_frames} corrupted frame(s); "
                f"{len(dataset)} remain."
            )

        if validate_setup:
            scene_corrections = z_sign_corrections.get(scene_name, {})
            z_sign_for_logging = [scene_corrections.get(str(c), 1.0) for c in cam_ids]
            problems = log_and_validate_camera_setup(
                scene_name,
                cam_ids,
                z_sign_for_logging,
                dataset.active_refs,
                scene_folder,
                snapshot_ids=dataset.snapshots,
            )
            if problems:
                all_problems[scene_name] = problems

        registry.append(
            {
                "scene_name": scene_name,
                "scene_folder": scene_folder,
                "cam_ids": cam_ids,
                "dataset": dataset,
            }
        )

    if all_problems:
        total = sum(len(v) for v in all_problems.values())
        print("\n" + "=" * 70)
        print(f"!!! CAMERA SETUP VALIDATION: {total} problem(s) across {len(all_problems)}/{len(registry)} scenes !!!")
        for scene_name, problems in all_problems.items():
            for p in problems:
                print(f"  [{scene_name}] {p}")
        print("=" * 70)
        raise AssertionError(
            f"Camera setup validation failed for {len(all_problems)} scene(s) -- see the full "
            f"list above. Refusing to start training on possibly-broken reference points. "
            f"Investigate (e.g. visualize the flagged cameras' coverage -- a healthy-looking "
            f"coverage percentage does NOT necessarily mean the flagged disagreement is a false "
            f"alarm, and vice versa) before deciding whether to fix the underlying data/z_sign "
            f"file, or re-run camera selection excluding these specific cameras -- consistent "
            f"with this project's existing preference for swapping out a disputed camera over "
            f"patching around it. Re-run with --skip_camera_validation only as a deliberate, "
            f"temporary stopgap, never as the actual fix."
        )

    return registry


def load_camera_selection(k):
    """optimal_cameras_filtered.json (K-means camera selection without fold-back cameras) and the z_sign
    corrections for the startup check: the full file (every camera on disk) if present, else the narrow one."""
    print("=" * 70)
    print(f"Loading camera selection from optimal_cameras_filtered.json (k={k})...")
    config_path = os.path.join(cfg.src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")
    with open(config_path, "r") as f:
        optimal_cameras_config = json.load(f)

    z_sign_full_path = os.path.join(cfg.src_dir, "utils", "z_sign_corrections_full.json")
    z_sign_narrow_path = os.path.join(cfg.src_dir, "utils", "z_sign_corrections.json")
    if os.path.exists(z_sign_full_path):
        z_sign_path = z_sign_full_path
    elif os.path.exists(z_sign_narrow_path):
        z_sign_path = z_sign_narrow_path
        print(
            f"WARNING: {z_sign_full_path} not found -- falling back to the NARROW "
            f"{z_sign_narrow_path}. Cameras outside its ablation subsets default to z_sign=+1.0 "
            f"in the startup validation, which can cause false mismatches or mask real ones."
        )
    else:
        z_sign_path = None
    if z_sign_path is not None:
        print(f"Using z_sign corrections from: {z_sign_path}")
        with open(z_sign_path, "r") as f:
            z_sign_corrections = json.load(f)
    else:
        print(
            f"WARNING: no z_sign corrections file found in {os.path.join(cfg.src_dir, 'utils')} -- "
            f"assuming no z-sign corrections needed anywhere."
        )
        z_sign_corrections = {}
    return optimal_cameras_config, z_sign_corrections


def build_registries(k, validate, train_weather=None, val_weather=None):
    """Train and val registries (built once for the whole run), with the frame and weather summary."""
    optimal_cameras_config, z_sign_corrections = load_camera_selection(k)
    validate_excluded_frame_keys()  # fail on a typo'd scene key before building ~40 datasets

    if train_weather or val_weather:
        print(f"[WEATHER FILTER] train={train_weather or 'all'} | val={val_weather or 'all'}")
    t_registry_start = time.perf_counter()
    print(f"Building TRAIN scene registry (validate_setup={validate})...")
    train_registry = build_scene_registry(
        "train", k, optimal_cameras_config, z_sign_corrections, validate, allowed_weather=train_weather
    )
    print(f"Building VAL scene registry (validate_setup={validate})...")
    val_registry = build_scene_registry(
        "validate", k, optimal_cameras_config, z_sign_corrections, validate, allowed_weather=val_weather
    )
    print(f"Registries built: {len(train_registry)} train scenes, {len(val_registry)} val scenes.")
    cfg.STARTUP_TIMING["registry_build_s"] = time.perf_counter() - t_registry_start
    if cfg.TIMING_ENABLED:
        print(
            f"[timing] Registry build (dataset construction"
            f"{'' if not validate else ' + camera validation'}): {format_duration(cfg.STARTUP_TIMING['registry_build_s'])}"
        )

    n_train_frames = sum(len(e["dataset"]) for e in train_registry)
    n_val_frames = sum(len(e["dataset"]) for e in val_registry)
    n_excl_train = sum(e["dataset"].num_excluded_frames for e in train_registry)
    n_excl_val = sum(e["dataset"].num_excluded_frames for e in val_registry)
    print(
        f"Frames available after exclusions: train={n_train_frames} (dropped {n_excl_train}), "
        f"val={n_val_frames} (dropped {n_excl_val})."
    )
    for split_name, reg in (("train", train_registry), ("val", val_registry)):
        comp = {}
        for e in reg:
            for w, n in e["dataset"].weather_counts.items():
                comp[w] = comp.get(w, 0) + n
        if comp:
            print(f"[weather] {split_name} frames by weather (after filters): {comp}")
    print("=" * 70)
    return train_registry, val_registry


def sample_train_subset_loader(scene_entry, num_frames):
    """DataLoader over a fresh random subset of num_frames of the scene's (non-excluded) frames; redrawn every
    RESAMPLE_EVERY_N_EPOCHS so training sees more of each scene at the same per-epoch cost."""
    dataset = scene_entry["dataset"]
    total_frames = len(dataset)  # already excludes corrupted frames
    n = min(num_frames, total_frames)
    indices = random.sample(range(total_frames), n)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)


def build_cached_val_samples(val_registry, num_frames_per_scene, val_seed=None):
    """Materialises the validation samples ONCE (iterating a loader again would redraw each frame's random crop and
    make epochs incomparable). With val_seed, each scene is seeded with "<val_seed>:<scene_name>", so every stage
    and run validates on identical frames, cameras and crops and adding a scene never shifts another; the global
    RNG states are restored afterwards. Returns (samples, manifest: scene / snapshot / crop offsets / weather)."""
    cached_samples = []
    manifest = []
    py_rng_state = random.getstate()
    torch_rng_state = torch.get_rng_state()
    try:
        for scene_entry in val_registry:
            if val_seed is not None:
                random.seed(f"{val_seed}:{scene_entry['scene_name']}")
            dataset = scene_entry["dataset"]
            total_frames = len(dataset)
            n = min(num_frames_per_scene, total_frames)
            indices = random.sample(range(total_frames), n)
            subset = Subset(dataset, indices)
            loader = DataLoader(subset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
            for sample in loader:
                sample["scene_name"] = scene_entry["scene_name"]
                manifest.append(
                    {
                        "scene": scene_entry["scene_name"],
                        "snapshot_id": str(sample["snapshot_id"][0]),
                        "hw_random": sample["hw_random"][0].tolist(),  # [[h_starts], [w_starts]]
                        "weather": str(sample["weather"][0]),
                    }
                )
                cached_samples.append(sample)
    finally:
        random.setstate(py_rng_state)
        torch.set_rng_state(torch_rng_state)
    return cached_samples, manifest


def build_full_scene_val_set(val_registry, frames_per_scene, draws_per_frame, seed):
    """Fixed whole-scene validation set, scored like evaluation/evaluate_paper_protocol.py: per val scene a
    dataset over ALL its cameras (5 drawn at random per sample, as in the paper protocol), `frames_per_scene` frames
    and `draws_per_frame` camera draws each. Only (scene, dataset, frame, draw-seed) is stored; samples are re-loaded
    each epoch with the same seed, so every epoch scores the identical samples without keeping images in memory."""
    entries = []
    for e in val_registry:
        ds = SynMVCrowdDataset(scene_dir=e["scene_folder"], camera_ids=None, num_sampled_cams=5, load_refs=True)
        rng = random.Random(f"fullscene:{seed}:{e['scene_name']}")
        frames = sorted(rng.sample(range(len(ds)), min(frames_per_scene, len(ds))))
        for f in frames:
            for d in range(draws_per_frame):
                entries.append(
                    {
                        "scene": e["scene_name"],
                        "ds": ds,
                        "idx": f,
                        "seed": f"fullscene:{seed}:{e['scene_name']}:{f}:{d}",
                    }
                )
    return entries


@torch.no_grad()
def full_scene_validation(model, entries, routing, device):
    """Whole-scene count error on the fixed set (learned or oracle routing), via the evaluator's own
    predict_proposed / gt_pixels / oracle_zone, so the numbers mean the same as the test protocol's. Global RNG
    states are restored afterwards so training randomness is unaffected."""
    from evaluation.evaluate_paper_protocol import predict_proposed, gt_pixels, oracle_zone, SCALE

    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    model.bev_shape = cfg.BEV_SHAPE
    model.to_scale = SCALE / get_map_kernel(sigma=cfg.GAUSSIAN_SIGMA, kernel_size=cfg.GAUSSIAN_KERNEL_SIZE).sum().item()
    # oracle_zone reads these off the trainer module it is given
    trainer = types.SimpleNamespace(
        get_map_kernel=get_map_kernel,
        GAUSSIAN_SIGMA=cfg.GAUSSIAN_SIGMA,
        GAUSSIAN_KERNEL_SIZE=cfg.GAUSSIAN_KERNEL_SIZE,
        oracle_mask_from_conv=oracle_mask_from_conv,
    )
    errs, per_scene = [], defaultdict(list)
    for e in entries:
        random.seed(e["seed"])
        sample = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in e["ds"][e["idx"]].items()}
        wld = sample["wld_map_paras"]
        shape = (int(wld[0, 3].item()), int(wld[0, 4].item()))
        gts = gt_pixels(sample["gt_world_xy"], wld, shape)
        zone = oracle_zone(trainer, gts, shape) if routing == "oracle" else None
        pred = predict_proposed(model, sample, device, zone)
        err = pred.sum().item() / SCALE - len(gts)
        errs.append((err, len(gts)))
        per_scene[e["scene"]].append(abs(err))
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    signed = np.array([x for x, _ in errs])
    return {
        "mae": float(np.abs(signed).mean()),
        "bias": float(signed.mean()),
        "nae": float(np.mean([min(abs(x) / g, 1.0) if g else 0.0 for x, g in errs])),
        "n": len(errs),
        "per_scene": {k: float(np.mean(v)) for k, v in per_scene.items()},
    }


def build_stage_val_sets(stage, val_registry, val_frames_per_scene, val_seed):
    """The stage's cached validation samples (fingerprint printed; manifest written by the first stage, later
    stages warn if theirs differs) and, with FULL_SCENE_VAL_FRAMES, the whole-scene set (Stages 2-4, joint).
    Returns (cached_val_samples, full_scene_val or None, build seconds)."""
    print(
        f"Caching {val_frames_per_scene} frames/scene from {len(val_registry)} val scenes "
        f"(fixed crop per frame, reused identically every epoch this stage)..."
    )
    t_cache_start = time.perf_counter()
    cached_val_samples, val_manifest = build_cached_val_samples(val_registry, val_frames_per_scene, val_seed=val_seed)
    val_cache_s = time.perf_counter() - t_cache_start
    val_fingerprint = hashlib.md5(
        json.dumps(
            [{k: m[k] for k in ("scene", "snapshot_id", "hw_random")} for m in val_manifest], sort_keys=True
        ).encode()
    ).hexdigest()[:10]
    print(f"[val] Cache fingerprint {val_fingerprint} (val_seed={val_seed}, {len(val_manifest)} samples).")
    if "fingerprint" not in cfg.VAL_CACHE_INFO:
        cfg.VAL_CACHE_INFO["fingerprint"] = val_fingerprint
        os.makedirs(cfg.VAL_LOG_DIR, exist_ok=True)
        manifest_path = os.path.join(cfg.VAL_LOG_DIR, f"val_manifest_{cfg.TIMING_RUN_TAG}.json")
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "val_seed": val_seed,
                    "val_frames_per_scene": val_frames_per_scene,
                    "fingerprint": val_fingerprint,
                    "samples": val_manifest,
                },
                f,
                indent=1,
            )
        print(f"[val] Manifest written to {manifest_path}")
    elif cfg.VAL_CACHE_INFO["fingerprint"] != val_fingerprint:
        print(
            f"[val] WARNING: this stage's validation set differs from the first stage's "
            f"({cfg.VAL_CACHE_INFO['fingerprint']}). Stage-to-stage numbers are NOT on identical patches "
            f"(val_seed=None, or the data / --val_frames_per_scene changed)."
        )
    print(f"  -> {len(cached_val_samples)} total held-out validation samples for this stage.")
    full_scene_val = None
    if cfg.FULL_SCENE_VAL_FRAMES > 0 and stage in (2, 3, 4, "joint"):
        full_scene_val = build_full_scene_val_set(
            val_registry, cfg.FULL_SCENE_VAL_FRAMES, cfg.FULL_SCENE_VAL_DRAWS, val_seed if val_seed is not None else 0
        )
        print(
            f"[val-fullscene] {len(full_scene_val)} whole-scene samples ({cfg.FULL_SCENE_VAL_FRAMES} frames x "
            f"{cfg.FULL_SCENE_VAL_DRAWS} random camera draws per val scene), "
            f"{'oracle' if stage == 2 else 'learned'} routing; checkpoint selection uses their MAE."
        )
    if cfg.TIMING_ENABLED:
        print(
            f"[timing] Validation cache built in {format_duration(val_cache_s)} "
            f"({val_cache_s / max(len(cached_val_samples), 1):.2f} s/sample)."
        )
    return cached_val_samples, full_scene_val, val_cache_s
