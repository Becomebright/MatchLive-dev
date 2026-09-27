"""Offline match runner: drives the perception modules frame by frame (DESIGN.md §6.10).

Phase 0 scope: one pass over a half video that writes the raw perception cache
(matchlive.common.cache). Later phases hang the trigger, event analysis and memory off
the same loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from matchlive.common.cache import ROLE_CODE, cache_path, save_perception
from matchlive.common.match import Match
from matchlive.common.types import Detection
from matchlive.perception.detection import RFDETRDetector
from matchlive.perception.identity import kit_color
from matchlive.perception.jersey import NUM_LABELS, JerseyReader, crop_upper_body
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import MultiRoleTracker


class MatchRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        paths, perception = config["paths"], config["perception"]
        # Model sessions are expensive to create (TensorRT engines load from disk), so
        # they live as long as the runner; per-sequence state is created in perceive().
        self.detector = RFDETRDetector(perception["detection"], paths["checkpoints"], paths["cache"])
        self.jersey = JerseyReader(perception["jersey"], paths["checkpoints"], paths["cache"])

    def run_perception(self, match: Match, half: int, start_frame: int = 0,
                       n_frames: int | None = None) -> Path:
        """Write the raw perception cache for frames [start_frame, start_frame + n_frames)
        of one half (the whole half when n_frames is None); return its path. An existing
        cache for the same inputs is reused."""
        video = match.video_path(half)
        path = cache_path(self.config, video, match.name, half, start_frame, n_frames)
        if path.exists():
            return path

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        def frames():
            frame_idx = start_frame
            while n_frames is None or frame_idx < start_frame + n_frames:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame_idx, frame
                frame_idx += 1

        arrays, timing = self.perceive(frames(), fps)
        meta = {"match": match.name, "half": half, "fps": fps, "start_frame": start_frame,
                "n_frames": len(arrays["shot_id"]), "ms_per_frame": timing}
        save_perception(path, arrays, meta)
        return path

    def perceive(self, frames: Iterable[tuple[int, np.ndarray]], fps: float,
                 detections: dict[int, list[Detection]] | None = None) -> tuple[dict, dict]:
        """Run the per-frame chain over (frame_idx, BGR frame) pairs and return the raw cache
        arrays (layout in matchlive.common.cache) and mean ms/frame per stage.

        `detections` (frame_idx -> detections) replaces the detector, for data whose
        detections are already cached (SoccerNet-GSR evaluation).
        """
        perception = self.config["perception"]
        shots = ShotDetector(perception["shots"]["cut_threshold"])
        tracker = MultiRoleTracker(perception["tracking"], frame_rate=fps)

        shot_ids, det_rows, track_rows, jersey_rows, color_rows = [], [], [], [], []
        timing = {"decode": 0.0, "detect": 0.0, "shots": 0.0, "track": 0.0, "jersey": 0.0, "color": 0.0}
        t_prev = time.perf_counter()
        for frame_idx, frame in frames:
            t0 = time.perf_counter()
            dets = self.detector.predict(frame) if detections is None else detections.get(frame_idx, [])
            t1 = time.perf_counter()
            shot_id = shots.update(frame)
            t2 = time.perf_counter()
            tracks = tracker.update(dets, frame, shot_id)
            t3 = time.perf_counter()
            # Read every track on every frame; boxes too small to crop get NaN rows so the
            # jersey and colour arrays stay aligned with the track rows.
            crops = [crop_upper_body(frame, t.bbox_ltwh) for t in tracks]
            probs = np.full((len(tracks), NUM_LABELS), np.nan, dtype=np.float32)
            readable = [i for i, c in enumerate(crops) if c is not None]
            if readable:
                probs[readable] = self.jersey.predict_batch([crops[i] for i in readable])
            t4 = time.perf_counter()
            colors = np.full((len(tracks), 3), np.nan, dtype=np.float32)
            for i in readable:
                colors[i] = kit_color(crops[i])
            t5 = time.perf_counter()

            shot_ids.append(shot_id)
            det_rows += [(frame_idx, *d.bbox_ltwh, d.score, d.class_id) for d in dets]
            track_rows += [(frame_idx, shot_id, ROLE_CODE[t.role], t.track_id, *t.bbox_ltwh) for t in tracks]
            jersey_rows.append(probs)
            color_rows.append(colors)
            # "decode" is the time spent getting this frame from the iterator.
            for key, dt in zip(timing, (t0 - t_prev, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                timing[key] += dt
            t_prev = time.perf_counter()

        n = max(len(shot_ids), 1)
        arrays = {
            "shot_id": np.asarray(shot_ids, dtype=np.int32),
            "det": np.asarray(det_rows, dtype=np.float32).reshape(-1, 7),
            "track": np.asarray(track_rows, dtype=np.float32).reshape(-1, 8),
            "jersey": np.concatenate(jersey_rows or [np.zeros((0, NUM_LABELS))]).astype(np.float16),
            "color": np.concatenate(color_rows or [np.zeros((0, 3))]).astype(np.float32),
        }
        return arrays, {k: 1000 * v / n for k, v in timing.items()}
