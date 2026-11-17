"""
weather.py

Loader for weather_index.json, written by build_weather_index.py:
    {scene_name: {snapshot_folder_name: {<top-level scalar JSON fields>, e.g. "weather": "EXTRASUNNY"}}}
Returns {} if the file does not exist (weather logging then reports UNKNOWN; filtering by weather raises).
"""

import json
import os

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def load_weather_index(path=None):
    path = path or os.path.join(_UTILS_DIR, "weather_index.json")
    if path not in _CACHE:
        if os.path.exists(path):
            with open(path, "r") as f:
                _CACHE[path] = json.load(f)
        else:
            _CACHE[path] = {}
    return _CACHE[path]
