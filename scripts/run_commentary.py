"""Live commentary over a live run (runtime.runner.run_commentary, DESIGN.md §6.9).

    python scripts/run_commentary.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--run full]
        [--clip 2 900 1020]

Reads the run's memory (scripts/run_live.py, outputs/live/<match>/<run>/) and writes
outputs/commentary/<match>/<run>/:
  lines.jsonl       one line per record (runtime.runner.run_commentary), plus its display time;
  half<h>.srt       the lines as subtitles on each half video: shown at (ready time - output_delay)
                    — the broadcast runs that far behind the feed (DESIGN.md §6.9 D3) — but never
                    before the event a line speaks of is on screen, nor over the previous line;
  clip_*.mp4        with --clip HALF START END: that stretch of video with the subtitles burned in,
                    for a look by eye.
Each line is one cached model call (api section), so a re-run replays in seconds.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter

import cv2
import numpy as np

from matchlive.app.commentary import speaking_time
from matchlive.common.api import APIClient
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.runtime.runner import run_commentary


def display_times(lines: list[dict], cfg: dict) -> None:
    """Set each line's "show" (half, start, end) on its half video, in place."""
    prev = {1: 0.0, 2: 0.0}
    for line in lines:
        half, ready = line["t_ready"]
        start = ready - cfg["output_delay"]
        if line["t_event"]:
            start = max(start, line["t_event"][1] + 1.0)       # not before the event is on screen
        start = max(start, prev[half], 0.0)
        end = start + speaking_time(line["text"], cfg["words_per_second"])
        prev[half] = end
        line["show"] = (half, start, end)


def srt(lines: list[dict], half: int) -> str:
    def stamp(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    out = []
    for i, line in enumerate(l for l in lines if l["show"][0] == half):
        out.append(f"{i + 1}\n{stamp(line['show'][1])} --> {stamp(line['show'][2])}\n{line['text']}\n")
    return "\n".join(out)


def render_clip(match: Match, lines: list[dict], half: int, start: float, end: float, path) -> None:
    cap = cv2.VideoCapture(str(match.video_path(half)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    shown = [l for l in lines if l["show"][0] == half and l["show"][2] >= start and l["show"][1] <= end]
    for k in range(int((end - start) * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        s = start + k / fps
        for line in shown:
            if line["show"][1] <= s <= line["show"][2]:
                wrapped = textwrap.wrap(f"[{line['kind']}] {line['text']}", 70)
                y0 = h - 30 - 34 * len(wrapped)
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, y0 - 10), (w - 20, h - 20), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                for i, text in enumerate(wrapped):
                    cv2.putText(frame, text, (35, y0 + 24 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{half}H {int(s // 60):02d}:{s % 60:04.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
    out.release()
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full", help="live run name (scripts/run_live.py --name)")
    ap.add_argument("--clip", nargs=3, type=float, metavar=("HALF", "START", "END"),
                    help="also write this stretch of video with the subtitles burned in")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config setting on top of the run's own (e.g. api.max_attempts=40)")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir) + args.set)
    match = Match(config, args.match)
    out_dir = config["paths"]["outputs"] / "commentary" / args.match / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = run_commentary(config, match, run_dir, APIClient(config["api"], config["paths"]["cache"]))
    display_times(lines, config["commentary"])
    with open(out_dir / "lines.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    for half in (1, 2):
        (out_dir / f"half{half}.srt").write_text(srt(lines, half))
    if args.clip:
        half, start, end = int(args.clip[0]), args.clip[1], args.clip[2]
        path = out_dir / f"clip_{half}H_{int(start)}-{int(end)}.mp4"
        render_clip(match, lines, half, start, end, path)
        print(path)

    print(out_dir)
    kinds = Counter(l["kind"] for l in lines)
    lag = [l["show"][1] - l["t_event"][1] for l in lines if l["t_event"]]
    live_lag = [l["t_ready"][1] - l["t_event"][1] for l in lines if l["t_event"]]
    print(f"{len(lines)} lines: {dict(kinds)}; generation latency p50 "
          f"{np.median([l['latency'] for l in lines]):.1f} s")
    if lag:
        print(f"event -> line ready (live): p50 {np.median(live_lag):.0f} s, p90 {np.percentile(live_lag, 90):.0f} s; "
              f"event on screen -> line shown (output delay {config['commentary']['output_delay']} s): "
              f"p50 {np.median(lag):.0f} s, p90 {np.percentile(lag, 90):.0f} s")


if __name__ == "__main__":
    main()
