"""Raw perception cache: one compressed .npz per (match, half, frame range).

This is the config-independent layer of DESIGN.md §2 principle 8 — everything that needs
the video pixels or a model — so identity voting and later phases recompute from it
instead of re-running the models. Arrays (row order = frame order):

  shot_id  int32   (n_frames,)       shot id of each frame
  det      float32 (N, 7)            frame, l, t, w, h, score, class_id — the detector's
                                     output (its confidence cut applied, referees dropped);
                                     the ball is selected from these at load time
  track    float32 (M, 8)            frame, shot_id, role (0 goalkeeper, 1 player),
                                     track_id, l, t, w, h
  jersey   float16 (M, 101)          jersey-number probabilities of each track row, read
                                     on every frame; NaN where the box was too small to crop

`frame` is the frame index within the half. The file name carries a hash of everything
the arrays depend on (video, model files, perception parameters, format version), so a
change to any of them writes a new file instead of reusing a stale one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
ROLE_CODE = {"goalkeeper": 0, "player": 1}


def _file_id(path: Path) -> list:
    # Name, size and mtime identify a video or model file without hashing hundreds of MB;
    # re-exporting a model changes its mtime and so invalidates the cache.
    stat = path.stat()
    return [path.name, stat.st_size, stat.st_mtime_ns]


def cache_path(config: dict[str, Any], video: Path, match_name: str, half: int,
               start_frame: int, n_frames: int | None) -> Path:
    perception = config["perception"]
    ckpt = config["paths"]["checkpoints"]
    depends_on = {
        "format": FORMAT_VERSION,
        "video": _file_id(video),
        "detector": _file_id(ckpt / perception["detection"]["model"]),
        "jersey": _file_id(ckpt / perception["jersey"]["model"]),
        "confidence": perception["detection"]["confidence"],
        "shots": perception["shots"],
        "tracking": perception["tracking"],
    }
    key = hashlib.sha1(json.dumps(depends_on, sort_keys=True).encode()).hexdigest()[:10]
    span = "all" if n_frames is None else f"{start_frame}+{n_frames}"
    return config["paths"]["cache"] / "perception" / match_name / f"half{half}_{span}_{key}.npz"


def save_perception(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so an interrupted run never leaves a file that
    # looks complete.
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arrays)
    tmp.replace(path)


def load_perception(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    return arrays, meta
