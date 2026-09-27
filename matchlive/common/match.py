"""One match on disk: its metadata, half videos, and benchmark time alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Match:
    def __init__(self, config: dict[str, Any], name: str):
        # `name` is the match folder name, e.g. "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool";
        # it is unique across leagues/seasons in our dataset, so we search for it.
        found = sorted(config["paths"]["dataset"].glob(f"*/*/{name}"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one match folder named {name!r}, found {found}")
        self.name = name
        self.dir: Path = found[0]
        with open(self.dir / "metadata.json") as f:
            self.metadata: dict[str, Any] = json.load(f)
        # Only matches in the streaming-QA benchmark have offsets.
        self.benchmark_offsets: dict[int, int] | None = config["benchmark_offsets"].get(name)

    def video_path(self, half: int) -> Path:
        return self.dir / self.metadata["videos"][str(half)]

    def benchmark_timestamp(self, half: int, second: float) -> float:
        # (half, second) is the primary time key; the benchmark's whole-video timestamp is
        # derived through the per-match offset table (DESIGN.md §4). Half-video seconds were
        # checked against the on-screen clock and are used as match time without correction.
        return self.benchmark_offsets[half] + second
