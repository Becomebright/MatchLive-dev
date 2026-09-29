"""One match on disk: its metadata, half videos, and benchmark time alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class Match:
    def __init__(self, config: dict[str, Any], name: str):
        # `name` is the match folder name, e.g. "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool";
        # it is unique across leagues/seasons in our dataset, so we search for it. It carries
        # the final score, so it is a file-system key only and never shown to a model.
        found = sorted(config["paths"]["dataset"].glob(f"*/*/{name}"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one match folder named {name!r}, found {found}")
        self.name = name
        self.dir: Path = found[0]
        with open(self.dir / "metadata.json") as f:
            # The whole file, post-match facts included (final score, clip list): for evaluation
            # scripts only. The live system reads `prematch`.
            self.metadata: dict[str, Any] = json.load(f)
        self.prematch: dict[str, Any] = prematch(self.metadata)
        # Only matches in the streaming-QA benchmark have offsets.
        self.benchmark_offsets: dict[int, int] | None = config["benchmark_offsets"].get(name)

    def video_path(self, half: int) -> Path:
        return self.dir / self.metadata["videos"][str(half)]

    def benchmark_timestamp(self, half: int, second: float) -> float:
        # (half, second) is the primary time key; the benchmark's whole-video timestamp is
        # derived through the per-match offset table (DESIGN.md §4). Half-video seconds were
        # checked against the on-screen clock and are used as match time without correction.
        return self.benchmark_offsets[half] + second


def prematch(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """What is known before kick-off — the only match information the live system gets
    (DESIGN.md §6.8): the team names, kit colours, which goal each team defends in each half,
    and the two squads (name, shirt number, starting or substitute, and a starter's position
    from the announced formation). Everything else in metadata.json (the final score, the
    clip list, nationality, referee, venue, ...) stays out."""
    # home_team_side is the side of the main camera's picture the home team defends in the
    # first half (checked on all six local halves: at kick-off the home kit stands on that
    # side); the teams swap ends at half-time.
    first = metadata["home_team_side"]
    other = {"left": "right", "right": "left"}[first]
    players = []
    for p in metadata["players"]:
        players.append({k: p[k] for k in ("hash", "team_side", "name", "short_name", "long_name",
                                          "jersey_number", "starting")}
                       | {"position": _position(metadata["tactics"][p["team_side"]], p["position"])})
    return {
        "teams": dict(metadata["teams"]),
        "jersey_colors": dict(metadata["jersey_colors"]),
        "sides": {1: {"home": first, "away": other}, 2: {"home": other, "away": first}},
        "players": players,
    }


def _position(formation: str, slot: int | None) -> str | None:
    """A starter's line in the formation: "1-4-2-3-1" with slot 7 -> "midfielder". Slots run
    through the lines in order, goalkeeper first; substitutes have no slot."""
    if not slot:
        return None
    lines = [int(n) for n in formation.split("-")]
    for i, n in enumerate(lines):
        if slot <= n:
            return "goalkeeper" if i == 0 else "defender" if i == 1 else "forward" if i == len(lines) - 1 else "midfielder"
        slot -= n
    return None
