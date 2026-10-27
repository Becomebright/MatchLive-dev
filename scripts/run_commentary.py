"""Live commentary over a live run (runtime.runner.run_commentary, DESIGN.md §6.9): the script of
timed phrases, stretch by stretch, and the commentator's voice.

    python scripts/run_commentary.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--run full]
        [--name NAME] [--only HALF START END] [--dry] [--set api.max_attempts=40 ...]

Reads the run's memory (scripts/run_live.py, outputs/live/<match>/<run>/) and writes
outputs/commentary/<match>/<run>/:
  lines.jsonl       one record per phrase on air (runtime.runner.run_commentary): text, tone (its
                    energy), audio (an MP3 in the speech cache) and its duration, the frame it was
                    written for ("at") and "show", when it is on air on the half video (the output
                    runs output_delay s behind the feed);
  half<h>.srt       the phrases as subtitles on each half video.
scripts/render_stream.py then puts them on the video. Every script and every phrase's audio is one
cached model call each, so a re-run replays in seconds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from types import SimpleNamespace

import numpy as np

from matchlive.common.api import APIClient
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.app.tts import NoVoice
from matchlive.runtime.runner import run_commentary


class DryRun:
    """Stands in for the API client in a dry run: a stretch's script says each must-say item, its
    first fact cut to 12 words, at its frame, in 4 s of made-up latency; there is no voice, so a
    phrase's speaking time is estimated."""

    def generate(self, parts, json_output=False, reasoning=None):
        # A confirmation is "at a pause": said on the first frame.
        must = re.findall(r"^- (M\d+), (?:at frame (\d+)|at a pause in these seconds) \(\w+\): (.*?) How:", parts[0], flags=re.M)
        script = [{"frame": int(k or 1), "say": " ".join(facts.split()[:12]), "energy": "lively", "material": m}
                  for m, k, facts in must]
        return SimpleNamespace(text=json.dumps({"script": script}), latency=4.0, call_id="dry")

    def speech(self, *args):
        raise NoVoice


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
    ap.add_argument("--name", default=None, help="output folder under outputs/commentary/<match>/ (default: the run)")
    ap.add_argument("--dry", action="store_true",
                    help="no model call: a stub script (each must-say item's first fact) with estimated speaking times, to try policies")
    ap.add_argument("--only", nargs=3, type=float, metavar=("HALF", "START", "END"),
                    help="only this stretch of video time (a trial)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config setting on top of the run's own (e.g. api.max_attempts=40)")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir) + args.set)
    match = Match(config, args.match)
    out_dir = config["paths"]["outputs"] / "commentary" / args.match / (args.name or args.run)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = DryRun() if args.dry else APIClient(config["api"], config["paths"]["cache"])
    # The scripts through a model of their own (commentary.api: its settings over `api`).
    own = config["commentary"]["api"]
    writer = APIClient({**config["api"], **own}, config["paths"]["cache"]) if not args.dry else None
    only = (int(args.only[0]), args.only[1], args.only[2]) if args.only else None
    lines = run_commentary(config, match, run_dir, client, only, writer)
    (out_dir / "run.json").write_text(json.dumps({"run": args.run, "dry": args.dry, "set": args.set, "only": only}, indent=1))
    with open(out_dir / "lines.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    for half in (1, 2):
        (out_dir / f"half{half}.srt").write_text(srt(lines, half))

    print(out_dir)
    kinds = Counter(l["kind"] for l in lines)
    events = [l for l in lines if l["kind"] in ("news", "confirmation", "correction")]
    voiced = [l for l in lines if l["audio"]]
    stretches = {tuple(l["segment"]): l for l in lines}.values()
    print(f"{len(lines)} phrases: {dict(kinds)}; energy {dict(Counter(l['tone'] for l in lines))}; voiced {len(voiced)}; "
          f"script p50 {np.median([l['latency'] for l in stretches]):.1f} s, voicing p50 "
          f"{np.median([l['tts_latency'] for l in stretches]):.1f} s, phrase audio p50 "
          f"{np.median([l['duration'] for l in voiced]) if voiced else 0:.1f} s")
    late = [l["show"][1] - l["at"][1] for l in lines]
    print(f"phrase on air after its frame: p50 {np.median(late):.1f} s, p90 {np.percentile(late, 90):.1f} s")
    if events:
        lag = [l["show"][1] - l["t_event"][1] for l in events]
        print(f"event on screen -> its phrase on air (output delay {config['commentary']['output_delay']} s): "
              f"p50 {np.median(lag):.0f} s, p90 {np.percentile(lag, 90):.0f} s")


if __name__ == "__main__":
    main()
