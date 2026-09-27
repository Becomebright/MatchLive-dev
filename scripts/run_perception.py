"""Run perception over a match half (or a segment of it) and write the raw cache.

    python scripts/run_perception.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1
    python scripts/run_perception.py --match "..." --half 1 --start 600 --seconds 120
"""

from __future__ import annotations

import argparse

from matchlive.common.cache import load_perception
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.runtime.runner import MatchRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, required=True)
    ap.add_argument("--start", type=float, default=0.0, help="start second within the half")
    ap.add_argument("--seconds", type=float, help="segment length (default: to the end of the half)")
    args = ap.parse_args()

    config = load_config()
    match = Match(config, args.match)
    # Our videos are 25 fps (DESIGN.md §4); the runner reads the real rate from the video.
    start_frame = int(round(args.start * 25))
    n_frames = None if args.seconds is None else int(round(args.seconds * 25))
    path = MatchRunner(config).run_perception(match, args.half, start_frame, n_frames)

    arrays, meta = load_perception(path)
    print(f"{path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{meta['n_frames']} frames, {arrays['shot_id'][-1] - arrays['shot_id'][0]} cuts, "
          f"{len(arrays['det'])} detections, {len(arrays['track'])} track boxes")
    print("ms/frame: " + "  ".join(f"{k} {v:.1f}" for k, v in meta["ms_per_frame"].items())
          + f"  | total {sum(meta['ms_per_frame'].values()):.1f}")
    print("frame total ms: " + "  ".join(f"{k} {v:.3f}" if k == "share_over_40" else f"{k} {v:.1f}"
                                        for k, v in meta["total_ms"].items()))


if __name__ == "__main__":
    main()
