"""Play a whole match live through the event loop (runtime.runner.run_live).

    python scripts/run_live.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--name full]
        [--set loop.arbiter=false ...]

Needs the raw perception cache of both halves (scripts/run_perception.py). The live system is
given only the pre-match facts (Match.prematch: team names, squads with shirt numbers,
starting flags and positions, kit colours, the ends each team defends); memory, windows and
cases are written to outputs/live/<match>/<name>/ for scripts/run_qa.py. API calls are cached
(cache/api/), so a re-run with the same config replays in minutes.

--set changes a setting of configs/default.yaml for this run (an ablation turns a loop switch
off, DESIGN.md §7); it is recorded in run.json, and the scripts reading the run apply it too.
"""

from __future__ import annotations

import argparse
import json
import time

from matchlive.common.api import APIClient
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.runtime.runner import run_live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--name", default="full", help="run name (e.g. an ablation)")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config setting, e.g. loop.arbiter=false (repeatable)")
    args = ap.parse_args()

    config = load_config(overrides=args.set)
    match = Match(config, args.match)
    out_dir = config["paths"]["outputs"] / "live" / args.match / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = {"set": args.set, "loop": config["loop"], "model": config["api"]["model"]}
    # Written first, so the QA can read a run's settings while it is still playing.
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))
    t0 = time.time()
    mem = run_live(config, match, out_dir, APIClient(config["api"], config["paths"]["cache"]))
    (out_dir / "run.json").write_text(json.dumps(run | {"wall_s": time.time() - t0}, indent=1))

    print(out_dir)
    print(f"wall {time.time() - t0:.0f} s; final score in memory {mem.at((3, 0.0)).score}")
    print(f"summary: python scripts/eval_live.py --match \"{args.match}\" --run {args.name}")


if __name__ == "__main__":
    main()
