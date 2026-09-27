"""Offline match runner: drives the perception modules frame by frame (DESIGN.md §6.10).

Phase 0 scope: one pass over a half video that writes the raw perception cache
(matchlive.common.cache). Later phases hang the trigger, event analysis and memory off
the same loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from matchlive.common.cache import ROLE_CODE, cache_path, save_perception
from matchlive.common.match import Match
from matchlive.perception.detection import RFDETRDetector
from matchlive.perception.jersey import NUM_LABELS, JerseyReader, crop_upper_body
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import MultiRoleTracker


class MatchRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        paths, perception = config["paths"], config["perception"]
        # Model sessions are expensive to create (TensorRT engines load from disk), so
        # they live as long as the runner; per-half state is created in run_perception.
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

        perception = self.config["perception"]
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        shots = ShotDetector(perception["shots"]["cut_threshold"])
        tracker = MultiRoleTracker(perception["tracking"], frame_rate=fps)

        shot_ids, det_rows, track_rows, jersey_rows = [], [], [], []
        timing = {"decode": 0.0, "detect": 0.0, "shots": 0.0, "track": 0.0, "jersey": 0.0}
        frame_idx = start_frame
        while n_frames is None or frame_idx < start_frame + n_frames:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()
            detections = self.detector.predict(frame)
            t2 = time.perf_counter()
            shot_id = shots.update(frame)
            t3 = time.perf_counter()
            tracks = tracker.update(detections, frame, shot_id)
            t4 = time.perf_counter()
            # Read every track on every frame; boxes too small to crop get a NaN row so
            # the jersey array stays aligned with the track rows.
            crops = [crop_upper_body(frame, t.bbox_ltwh) for t in tracks]
            probs = np.full((len(tracks), NUM_LABELS), np.nan, dtype=np.float32)
            readable = [i for i, c in enumerate(crops) if c is not None]
            if readable:
                probs[readable] = self.jersey.predict_batch([crops[i] for i in readable])
            t5 = time.perf_counter()

            shot_ids.append(shot_id)
            det_rows += [(frame_idx, *d.bbox_ltwh, d.score, d.class_id) for d in detections]
            track_rows += [(frame_idx, shot_id, ROLE_CODE[t.role], t.track_id, *t.bbox_ltwh) for t in tracks]
            jersey_rows.append(probs)
            for key, dt in zip(timing, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                timing[key] += dt
            frame_idx += 1

        n = frame_idx - start_frame
        arrays = {
            "shot_id": np.asarray(shot_ids, dtype=np.int32),
            "det": np.asarray(det_rows, dtype=np.float32).reshape(-1, 7),
            "track": np.asarray(track_rows, dtype=np.float32).reshape(-1, 8),
            "jersey": np.concatenate(jersey_rows or [np.zeros((0, NUM_LABELS))]).astype(np.float16),
        }
        meta = {
            "match": match.name, "half": half, "fps": fps,
            "start_frame": start_frame, "n_frames": n,
            "ms_per_frame": {k: 1000 * v / max(n, 1) for k, v in timing.items()},
        }
        save_perception(path, arrays, meta)
        return path
