"""Per-frame perception records shared across the package.

Boxes are ltwh (left, top, width, height) in pixels of the frame they came from
(DESIGN.md §4); other box formats exist only inside the module that needs them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    role: str                                   # "ball" | "goalkeeper" | "player"
    bbox_ltwh: tuple[float, float, float, float]
    score: float
    class_id: int                               # raw detector class id


@dataclass(frozen=True)
class Track:
    # A track is identified by (shot_id, role, track_id), never by track_id alone
    # (DESIGN.md §4): trackers are reset at every shot cut and each role has its own
    # tracker, so the same track_id is re-issued across shots and across roles.
    shot_id: int
    role: str                                   # "goalkeeper" | "player" (the ball is not tracked)
    track_id: int
    bbox_ltwh: tuple[float, float, float, float]
    score: float
    # Identity (DESIGN.md §5.1), filled in by perception.identity.IdentityResolver.
    team: str | None = None                     # "home" | "away" | None (no team vote yet)
    player_id: str | None = None                # roster hash, None until identified
    identity_source: str | None = None          # "vote" | "entry" (confirmed in memory)

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.shot_id, self.role, self.track_id)
