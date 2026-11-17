"""
run_offaxis_sweep.py

Runs an off-axis check (check_offaxis_combined.py by default, or check_offaxis_scene.py) on every scene, one
subprocess per scene so memory is released in between. The full-pool run below writes offaxis_combined_allcams.json,
the input of regenerate_optimal_cameras_filtered.py:

    uv run python utils/run_offaxis_sweep.py --script check_offaxis_combined.py --all_cameras --threshold 2.5 --closing_iterations 1
"""

import json
import os
import argparse
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
OPTIMAL_CAMERAS = os.path.join(src_dir, "datasets", "SynMVCrowd", "optimal_cameras_filtered.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="check_offaxis_combined.py")
    parser.add_argument("--num_cams", type=int, default=5)
    parser.add_argument("--min_component_size", type=int, default=None)
    parser.add_argument("--flag_median_ratio", type=float, default=None)
    parser.add_argument("--closing_iterations", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--all_cameras", action="store_true")
    args = parser.parse_args()

    script_path = os.path.join(current_dir, args.script)
    optimal_cameras_path = os.path.normpath(OPTIMAL_CAMERAS)
    with open(optimal_cameras_path, "r") as f:
        scenes = sorted(json.load(f).keys())

    print(f"Running {args.script} on {len(scenes)} scenes, one subprocess each...\n")

    first = True
    failures = []
    for i, scene in enumerate(scenes):
        cmd = ["uv", "run", "python", script_path, "--only_scene", scene, "--num_cams", str(args.num_cams)]
        if args.min_component_size is not None:
            cmd += ["--min_component_size", str(args.min_component_size)]
        if args.flag_median_ratio is not None:
            cmd += ["--flag_median_ratio", str(args.flag_median_ratio)]
        if args.closing_iterations is not None:
            cmd += ["--closing_iterations", str(args.closing_iterations)]
        if args.threshold is not None:
            cmd += ["--threshold", str(args.threshold)]
        if args.all_cameras:
            cmd.append("--all_cameras")
        if not first:
            cmd.append("--skip_calibration")
        first = False

        print(f"--- [{i + 1}/{len(scenes)}] {scene} ---")
        result = subprocess.run(cmd, cwd=src_dir)
        if result.returncode != 0:
            print(f"  !! subprocess failed for {scene} (exit code {result.returncode})")
            failures.append(scene)

    print("\n" + "=" * 70)
    print(f"Done. {len(scenes) - len(failures)}/{len(scenes)} scenes processed successfully.")
    if failures:
        print(f"Failed scenes: {failures}")
    print("=" * 70)


if __name__ == "__main__":
    main()
