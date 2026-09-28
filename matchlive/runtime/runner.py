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

from matchlive.common.cache import ROLE_CODE, cache_path, iter_frames, save_perception
from matchlive.common.match import Match
from matchlive.common.types import Detection, Track
from matchlive.memory.store import MatchMemory
from matchlive.perception.detection import RFDETRDetector, to_detections
from matchlive.perception.identity import IdentityResolver, kit_color, kit_lab, roster_index
from matchlive.perception.jersey import NUM_LABELS, UNREADABLE, JerseyReader, crop_upper_body, label_to_number
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import PlayerTracker, select_ball
from matchlive.perception.visualization import nearest_to_ball


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
                "n_frames": len(arrays["shot_id"]), **timing}
        save_perception(path, arrays, meta)
        return path

    def perceive(self, frames: Iterable[tuple[int, np.ndarray]], fps: float,
                 detections: dict[int, list[Detection]] | None = None) -> tuple[dict, dict]:
        """Run the per-frame chain over (frame_idx, BGR frame) pairs and return the raw cache
        arrays (layout in matchlive.common.cache) and a timing summary: mean ms/frame per
        stage ("ms_per_frame") and percentiles of each frame's total ("total_ms").

        `detections` (frame_idx -> detections) replaces the detector, for data whose
        detections are already cached (SoccerNet-GSR evaluation).
        """
        perception = self.config["perception"]
        shots = ShotDetector(perception["shots"]["cut_threshold"])
        tracker = PlayerTracker(perception["tracking"], frame_rate=fps)

        shot_ids, det_rows, track_rows, jersey_rows, color_rows = [], [], [], [], []
        timing = {"decode": 0.0, "detect": 0.0, "shots": 0.0, "track": 0.0, "jersey": 0.0, "color": 0.0}
        totals = []
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
            totals.append(t5 - t_prev)
            t_prev = time.perf_counter()

        n = max(len(shot_ids), 1)
        arrays = {
            "shot_id": np.asarray(shot_ids, dtype=np.int32),
            "det": np.asarray(det_rows, dtype=np.float32).reshape(-1, 7),
            "track": np.asarray(track_rows, dtype=np.float32).reshape(-1, 8),
            "jersey": np.concatenate(jersey_rows or [np.zeros((0, NUM_LABELS))]).astype(np.float16),
            "color": np.concatenate(color_rows or [np.zeros((0, 3))]).astype(np.float32),
        }
        # Real time at 25 fps means every frame, not just the average, fits in 40 ms.
        ms = 1000 * np.asarray(totals or [0.0])
        total_ms = {"p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
                    "p99": float(np.percentile(ms, 99)), "max": float(ms.max()),
                    "share_over_40": float((ms > 40).mean())}
        return arrays, {"ms_per_frame": {k: 1000 * v / n for k, v in timing.items()}, "total_ms": total_ms}


def resolve_frames(config: dict[str, Any], match: Match, arrays: dict[str, np.ndarray],
                   keep: set[int]) -> dict[int, tuple[list[Track], tuple | None, list[str | None]]]:
    """Replay identity voting over a half's raw perception cache, streaming in frame order
    (identities at frame t use only frames <= t), and keep frames `keep`:
    frame_idx -> (tracks with team / player_id, the key of the track nearest the ball or
    None, the number read clearly off each track in this frame or None). Frames without
    tracks or detections are absent.

    Phase 2 votes over the whole squad (the §7 loop.state_to_perception baseline); the
    on-pitch roster from memory comes with that switch in phase 4.
    """
    roster = roster_index(match.metadata["players"])
    resolver = IdentityResolver(config["perception"]["identity"], kit_lab(match.metadata["jersey_colors"]))
    min_ball = config["perception"]["ball"]["min_score"]
    min_prob = config["perception"]["identity"]["jersey_min_prob"]
    out = {}
    for frame_idx, tracks, jersey, color, det in iter_frames(arrays):
        tracks = resolver.update(frame_idx, tracks, jersey, color, roster, confirmed={})
        if frame_idx in keep:
            ball = select_ball(to_detections(det[:, :4], det[:, 4], det[:, 5], 0.0), min_ball)
            ball_point = None if ball is None else (ball.bbox_ltwh[0] + ball.bbox_ltwh[2] / 2,
                                                    ball.bbox_ltwh[1] + ball.bbox_ltwh[3])
            carrier = nearest_to_ball(tracks, ball_point)
            # "Clearly read" uses the same bar as a jersey vote (identity.jersey_min_prob).
            numbers = []
            for probs in jersey:
                label = -1 if np.isnan(probs[0]) else int(np.argmax(probs))
                clear = label not in (-1, UNREADABLE) and probs[label] >= min_prob
                numbers.append(label_to_number(label) if clear else None)
            out[frame_idx] = (tracks, None if carrier is None else carrier.key, numbers)
    return out


def replay_events(config: dict[str, Any], match: Match, results: dict[int, list[dict]]) -> MatchMemory:
    """Offline run of the event loop (DESIGN.md §6.10) from cached analyzer answers.

    `results` maps half -> the windows written by scripts/run_events.py (window, call_id,
    latency, candidates with their voted players). Each window's answer takes effect at
    its window end + the measured latency, so ops enter memory in knowledge-time order as
    they would live; the missing-partner constraints are judged as time passes.
    """
    from matchlive.events.analyzer import Candidate
    from matchlive.events.consistency import ConsistencyChecker

    mem = MatchMemory(match.metadata["players"], cascade=config["loop"]["cascade"])
    checker = ConsistencyChecker(config)
    arrivals = sorted(((half, r["window"][1] + r["latency"]), half, r)
                      for half, rows in results.items() for r in rows if "candidates" in r)
    for t_known, half, r in arrivals:
        checker.tick(mem, t_known)
        for i, c in enumerate(r["candidates"]):
            checker.propose(mem, t_known, half, tuple(r["window"]), Candidate.from_dict(c),
                            f"obs:{r['call_id']}#{i}", c["players"])
    # After the final whistle every missing partner has had its chance.
    checker.tick(mem, (3, 0.0))
    return mem

