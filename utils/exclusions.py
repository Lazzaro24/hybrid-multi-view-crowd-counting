import json
import os

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_excluded_scenes():
    """Whole-scene exclusions (e.g. scene_71: scene-wide geometry corruption)."""
    path = os.path.join(_UTILS_DIR, "excluded_scenes.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_excluded_frames():
    """Per-frame exclusions: {scene_name: set(int snapshot_id)}. Empty dict if file absent."""
    path = os.path.join(_UTILS_DIR, "excluded_frames.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {scene: {int(i) for i in ids} for scene, ids in raw.items()}
