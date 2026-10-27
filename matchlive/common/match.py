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
        # The pre-match knowledge base, if built (scripts/build_kb.py).
        kb_path = self.dir / "kb.json"
        kb = json.loads(kb_path.read_text()) if kb_path.exists() else None
        self.prematch: dict[str, Any] = prematch(self.metadata, kb)
        # Only matches in the streaming-QA benchmark have offsets.
        self.benchmark_offsets: dict[int, int] | None = config["benchmark_offsets"].get(name)

    def video_path(self, half: int) -> Path:
        return self.dir / self.metadata["videos"][str(half)]

    def match_time(self, timestamp: float) -> tuple[int, float]:
        """A streaming-QA benchmark timestamp (seconds of the whole video) -> (half, second).
        (half, second) is the primary time key; the benchmark's timestamp relates to it through
        the per-match offset table (DESIGN.md §4): timestamp = offset[half] + second. Half-video
        seconds were checked against the on-screen clock and are used as match time as is."""
        start2 = self.benchmark_offsets[2]
        return (2, float(timestamp - start2)) if timestamp >= start2 else (1, float(timestamp))


def prematch(metadata: Mapping[str, Any], kb: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What is known before kick-off — the only match information the live system gets
    (DESIGN.md §6.8): the team names, kit colours, which goal each team defends in each half,
    and the two squads (name, shirt number, starting or substitute, and a starter's position
    from the announced formation: its line and its side of it). Under "background", for the commentary: the formations,
    venue, referee, round, date and nationalities from metadata.json, and the pre-match
    knowledge base `kb` (scripts/build_kb.py) — each side's manager, captain and facts (the
    table and form going in, signings, ...), and each player's facts. Everything else in
    metadata.json (the final score, the clip list, the attendance, ...) stays out."""
    # home_team_side is the side of the main camera's picture the home team defends in the
    # first half (checked on all six local halves: at kick-off the home kit stands on that
    # side); the teams swap ends at half-time.
    first = metadata["home_team_side"]
    other = {"left": "right", "right": "left"}[first]
    players = []
    for p in metadata["players"]:
        players.append({k: p[k] for k in ("hash", "team_side", "name", "short_name", "long_name",
                                          "jersey_number", "starting")}
                       | {"position": _position(metadata["tactics"][p["team_side"]], p["position"]),
                          "side": _side(metadata["tactics"][p["team_side"]], p["position"]), "slot": p["position"]})
    kb = kb or {"teams": {}, "players": {}}
    background = {
        "venue": metadata["venue"][0].split(" (")[0] if metadata.get("venue") else None,   # "Stamford Bridge"
        "referee": metadata["referee"][0] if metadata.get("referee") else None,             # "Atkinson M."
        "round": metadata.get("round"),
        "date": metadata["game_date"].split(" - ")[0],                                      # "16/09/2016"
        "formations": {side: f.removeprefix("1-") for side, f in metadata["tactics"].items()},   # "4-2-3-1"
        "nationality": {p["hash"]: p["country"] for p in metadata["players"]},
        # Each fact as a sentence; its source and quote stay in kb.json.
        "teams": {side: {"manager": (t.get("manager") or {}).get("text"), "captain": (t.get("captain") or {}).get("text"),
                         "facts": [f["text"] for f in t.get("facts", [])]} for side, t in kb["teams"].items()},
        "players": {h: [f["text"] for f in p.get("facts", [])] for h, p in kb["players"].items()},
    }
    return {
        "teams": dict(metadata["teams"]),
        "jersey_colors": dict(metadata["jersey_colors"]),
        "sides": {1: {"home": first, "away": other}, 2: {"home": other, "away": first}},
        "players": players,
        "background": background,
    }


def _side(formation: str, slot: int | None) -> str | None:
    """A starter's side of their line in the formation: "1-4-2-3-1" with slot 5 -> "left". Each
    line's slots run from its right to its left (checked on both local matches' four line-ups:
    slot 2 is the right-back, the last defender the left-back, the first of the attacking three
    the right winger); a line of one is "centre", and the goalkeeper has none."""
    if not slot:
        return None
    lines = [int(n) for n in formation.split("-")]
    for i, n in enumerate(lines):
        if slot <= n:
            return None if i == 0 else "centre" if n == 1 else "right" if slot == 1 else "left" if slot == n else "centre"
        slot -= n


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
