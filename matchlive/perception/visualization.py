"""Draw tracks onto a frame: feet ellipses, ball triangle, ball-carrier marker (ported from v2).

Same visual design as v2: with an identity, the ellipse takes the team colour and shows
the jersey number (with the player's name below); otherwise it falls back to a role
colour and the track id.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np

from matchlive.common.types import Detection, Track

# BGR colour per role.
ROLE_COLOR = {"goalkeeper": (0, 255, 255), "player": (0, 200, 0)}


def draw_tracks(frame: np.ndarray, tracks: Sequence[Track], ball: Detection | None = None,
                team_bgr: Mapping[str, tuple[int, int, int]] | None = None,
                labels: Mapping[str, tuple[str, str]] | None = None,
                detections: Sequence[Detection] = ()) -> np.ndarray:
    """`team_bgr` maps a track's team ("home"/"away") to its colour; `labels` maps a
    player_id to (jersey number, name) drawn on the ellipse and under it. The ball carrier
    is the player / keeper detection nearest the ball among `detections`, tracked or not
    (a player whose track has not started yet still holds the ball)."""
    out = frame.copy()
    team_bgr, labels = team_bgr or {}, labels or {}

    # Ball: red marker above it. ball_point (the box's bottom-center) is the reference
    # for finding the ball carrier below.
    ball_point = None
    if ball is not None:
        x, y, w, h = (int(round(v)) for v in ball.bbox_ltwh)
        ball_point = (x + w // 2, y + h)
        width = max(14, int(round(w * 1.6)))
        _inverted_triangle(out, x + w // 2, max(8, y - 4), width, (0, 0, 255))

    for t in tracks:
        x, y, w, h = (int(round(v)) for v in t.bbox_ltwh)
        color = team_bgr.get(t.team, ROLE_COLOR[t.role])
        center = (x + w // 2, y + h)  # ellipse sits at the player's feet
        axes = (max(7, int(w * 0.5)), max(2, int(w * 0.12)))

        # Translucent filled ellipse, then a black outline + coloured rim on top
        # so the marker reads clearly over any pitch background. The blend only
        # touches the ellipse's bounding box: everywhere else it reduces to
        # 0.7*v + 0.3*v == v, so blending the whole frame once per player would
        # copy a full 720p image to change a ~40x10 patch.
        x0, y0 = max(0, center[0] - axes[0]), max(0, center[1] - axes[1])
        x1 = min(out.shape[1], center[0] + axes[0] + 1)
        y1 = min(out.shape[0], center[1] + axes[1] + 1)
        if x1 > x0 and y1 > y0:
            roi = out[y0:y1, x0:x1]
            overlay = roi.copy()
            cv2.ellipse(overlay, (center[0] - x0, center[1] - y0), axes, 0, 0, 360, color, -1)
            out[y0:y1, x0:x1] = cv2.addWeighted(overlay, 0.7, roi, 0.3, 0)
        cv2.ellipse(out, center, axes, 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(out, center, axes, 0, 0, 360, color, 2)

        # Jersey number when identified, else the track id — prefixed "t" so the two
        # never look alike.
        number, name = labels.get(t.player_id, (None, None))
        _centered_text(out, number or f"t{t.track_id}", center, 0.7)
        if name:
            _centered_text(out, name, (center[0], center[1] + axes[1] + 14), 0.5)

    # Mark the ball carrier with a green triangle above the box.
    carrier = nearest_to_ball([d for d in detections if d.role != "ball"], ball_point)
    if carrier is not None:
        x, y, w, h = (int(round(v)) for v in carrier.bbox_ltwh)
        _inverted_triangle(out, x + w // 2, max(8, y - 4), max(16, int(round(w * 0.45))), (0, 255, 0))
    return out


def draw_ids(frame: np.ndarray, boxes: Sequence[tuple[str, tuple[float, float, float, float]]]) -> np.ndarray:
    """The event analyzer's light mode (DESIGN.md §6.3 `ids_text`): only an ID label ("P3")
    above each (label, ltwh box), so the picture stays uncluttered and the identities
    travel as text instead of being drawn."""
    out = frame.copy()
    for label, (x, y, w, h) in boxes:
        _centered_text(out, label, (int(round(x + w / 2)), max(10, int(round(y)) - 10)), 0.6)
    return out


def nearest_to_ball(players: Sequence[Detection], ball_point: tuple[int, int] | None) -> Detection | None:
    # Player whose feet (box bottom-center) are closest to the ball point.
    # Squared distance is enough for an argmin; no need for the sqrt.
    if ball_point is None or not players:
        return None
    bx, by = ball_point
    return min(
        players,
        key=lambda t: (t.bbox_ltwh[0] + t.bbox_ltwh[2] / 2 - bx) ** 2
        + (t.bbox_ltwh[1] + t.bbox_ltwh[3] - by) ** 2,
    )


def _inverted_triangle(img, center_x: int, tip_y: int, width: int, color):
    # Both call sites want a height of 0.9x the width, so derive it here instead
    # of repeating the factor at every call.
    half = max(4, width // 2)
    height = max(6, int(width * 0.9))
    points = np.array(
        [[center_x - half, tip_y - height], [center_x + half, tip_y - height], [center_x, tip_y]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, points, color)
    cv2.polylines(img, [points], True, (0, 0, 0), 2, cv2.LINE_AA)


def _centered_text(img, text: str, center: tuple[int, int], font_scale: float):
    # Draw white text centered on `center`. The 8 offset passes paint a black
    # outline first so the label stays legible against any background.
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    x, y = center[0] - tw // 2, center[1] + th // 2
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.putText(img, text, (x + dx, y + dy), font, font_scale, (0, 0, 0), 2)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), 2)
