"""Multi-role tracking (ported from v2), BoT-SORT or ByteTrack from the `trackers` library,
plus per-frame ball selection.

One tracker instance per role (goalkeeper / player) so track ids never cross roles.
Synchronous: feed one frame's detections, get back that frame's tracks.

Changed from v2: all trackers are reset at every shot cut and tracks carry the shot id;
the algorithm is chosen in the config (BoT-SORT adds camera-motion compensation); the
ball is no longer tracked (see select_ball).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker, ByteTrackTracker

from matchlive.common.types import Detection, Track


def _to_supervision(detections: list[Detection]):
    # The trackers consume supervision's Detections (xyxy + confidence + class_id),
    # so adapt our ltwh Detection list into that format.
    if not detections:
        return sv.Detections.empty()
    xyxy = np.asarray(
        [(x, y, x + w, y + h) for x, y, w, h in (d.bbox_ltwh for d in detections)],
        dtype=np.float32,
    )
    confidence = np.asarray([d.score for d in detections], dtype=np.float32)
    class_id = np.asarray([d.class_id for d in detections], dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


class MultiRoleTracker:
    def __init__(self, config: dict[str, Any], frame_rate: float):
        # `config` is the perception.tracking section of the YAML config.
        tracker_class = {"botsort": BoTSORTTracker, "bytetrack": ByteTrackTracker}[config["algorithm"]]
        params = config[config["algorithm"]]
        # One independent tracker per role. Association is by position/IoU only, so
        # feeding both roles to a single instance would let a keeper box inherit a nearby
        # outfield player's id (and vice versa). Isolating each role means a tracker only
        # ever sees one kind of box, so ids cannot cross roles.
        #
        # In trackers 2.6 each instance allocates its own ids from 0 (v2's 2.3.0
        # shared one global counter), so the same track_id is live in several roles at
        # once: a track is only identified by (shot_id, role, track_id).
        self.trackers = {
            role: tracker_class(frame_rate=float(frame_rate), **params)
            for role in ("goalkeeper", "player")
        }
        # BoT-SORT's camera-motion compensation (CMC) estimates the global frame-to-frame
        # motion — the same for every role — but each instance would compute it again on
        # the same frame. Hand both trackers one shared estimator that computes once per
        # frame (at 720p the per-instance version cost ~6 ms/frame per tracker).
        self.shared_cmc = None
        if getattr(self.trackers["player"], "cmc", None) is not None:
            self.shared_cmc = _SharedCMC(self.trackers["player"].cmc)
            for tracker in self.trackers.values():
                tracker.cmc = self.shared_cmc
        self.shot_id: int | None = None

    def update(self, detections: list[Detection], frame: np.ndarray, shot_id: int) -> list[Track]:
        # A new shot id means the camera cut: positions in the new shot have nothing to do
        # with the old one, so both trackers start over instead of matching across the cut.
        if shot_id != self.shot_id:
            for tracker in self.trackers.values():
                tracker.reset()
            self.shot_id = shot_id
        if self.shared_cmc is not None:
            self.shared_cmc.new_frame()

        tracks: list[Track] = []
        for role, tracker in self.trackers.items():
            # Route each detection to its role's tracker; this filter is what
            # keeps the per-role isolation described in __init__.
            role_dets = [d for d in detections if d.role == role]
            # `frame` (BGR) is only used by BoT-SORT's camera-motion compensation;
            # ByteTrack warns if it is given one.
            if self.shared_cmc is not None:
                tracked = tracker.update(_to_supervision(role_dets), frame)
            else:
                tracked = tracker.update(_to_supervision(role_dets))
            for bbox, score, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                # -1 marks a detection not yet promoted to a confirmed track
                # (still within minimum_consecutive_frames).
                if int(track_id) == -1:
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox.tolist())
                tracks.append(
                    Track(
                        shot_id=shot_id,
                        role=role,
                        track_id=int(track_id),
                        bbox_ltwh=(x1, y1, x2 - x1, y2 - y1),
                        score=float(score),
                    )
                )
        return tracks


def select_ball(detections: list[Detection], min_score: float) -> Detection | None:
    """The frame's ball: its highest-scoring ball detection above `min_score`, or None.

    There is only one ball, so there is no identity to keep across frames and nothing
    for a tracker to associate. Measured on SoccerNet-GSR (one ball per frame, hit =
    centre within 15 px): top-1 at score > 0.3 gives precision 0.84 / recall 0.76, while
    ByteTrack's ball tracks sit on the same precision-recall curve (0.89 / 0.67) and
    BoT-SORT's drop recall to 0.30 — a small fast ball rarely clears its track-start
    confidence and IoU gates.
    """
    balls = [d for d in detections if d.role == "ball" and d.score > min_score]
    return max(balls, key=lambda d: d.score, default=None)


class _SharedCMC:
    """One camera-motion estimate per frame, shared by the per-role BoT-SORT trackers.

    Stands in for each tracker's own `trackers.utils.cmc.CMC` (BoT-SORT only calls
    `estimate()` and `reset()` on it). The first role to ask on a frame computes the
    estimate; the others reuse it. With the default sparse-optical-flow method the
    estimate depends only on the frames, not on the detection boxes each role passes in;
    the feature-based methods (orb / sift) mask out those boxes and would need all roles'
    boxes here instead.
    """

    def __init__(self, cmc):
        self.cmc = cmc
        self.affine: np.ndarray | None = None

    def new_frame(self) -> None:
        self.affine = None

    def estimate(self, frame_bgr: np.ndarray, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        if self.affine is None:
            self.affine = self.cmc.estimate(frame_bgr, dets_xyxy)
        return self.affine

    def reset(self) -> None:
        self.cmc.reset()
        self.affine = None
