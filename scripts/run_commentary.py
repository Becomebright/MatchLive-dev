"""Live commentary over a live run (runtime.runner.run_commentary, DESIGN.md §6.9): the lines, and
the commentator's voice.

    python scripts/run_commentary.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--run full]
        [--set api.max_attempts=40 ...]

Reads the run's memory (scripts/run_live.py, outputs/live/<match>/<run>/) and writes
outputs/commentary/<match>/<run>/:
  lines.jsonl       one record per line (runtime.runner.run_commentary): text, tone, audio (an MP3 in
                    the speech cache) and its duration, and "show", when it is on air on the half
                    video (app.commentary.schedule: ready time - output_delay, never before its event
                    is on screen, never over the previous line);
  half<h>.srt       the lines as subtitles on each half video.
scripts/render_stream.py then puts them on the video. Every line and its audio is one cached model
call each, so a re-run replays in seconds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from matchlive.app.commentary import schedule
from matchlive.common.api import APIClient
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.runtime.runner import run_commentary


def srt(lines: list[dict], half: int) -> str:
    def stamp(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    out = []
    for i, line in enumerate(l for l in lines if l["show"][0] == half):
        out.append(f"{i + 1}\n{stamp(line['show'][1])} --> {stamp(line['show'][2])}\n{line['text']}\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full", help="live run name (scripts/run_live.py --name)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config setting on top of the run's own (e.g. api.max_attempts=40)")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir) + args.set)
    match = Match(config, args.match)
    out_dir = config["paths"]["outputs"] / "commentary" / args.match / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = run_commentary(config, match, run_dir, APIClient(config["api"], config["paths"]["cache"]))
    schedule(lines, config["commentary"])
    with open(out_dir / "lines.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    for half in (1, 2):
        (out_dir / f"half{half}.srt").write_text(srt(lines, half))

    print(out_dir)
    kinds = Counter(l["kind"] for l in lines)
    lag = [l["show"][1] - l["t_event"][1] for l in lines if l["t_event"]]
    live_lag = [l["t_ready"][1] - l["t_event"][1] for l in lines if l["t_event"]]
    voiced = [l for l in lines if l["audio"]]
    print(f"{len(lines)} lines: {dict(kinds)}; tones {dict(Counter(l['tone'] for l in lines))}; voiced {len(voiced)}; "
          f"generation p50 {np.median([l['latency'] for l in lines]):.1f} s, speech p50 "
          f"{np.median([l['tts_latency'] for l in voiced]) if voiced else 0:.1f} s, audio p50 "
          f"{np.median([l['duration'] for l in voiced]) if voiced else 0:.1f} s")
    if lag:
        print(f"event -> line ready (live): p50 {np.median(live_lag):.0f} s, p90 {np.percentile(live_lag, 90):.0f} s; "
              f"event on screen -> line on air (output delay {config['commentary']['output_delay']} s): "
              f"p50 {np.median(lag):.0f} s, p90 {np.percentile(lag, 90):.0f} s")


if __name__ == "__main__":
    main()
