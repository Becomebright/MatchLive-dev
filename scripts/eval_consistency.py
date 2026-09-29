"""Replay a match's cached event candidates through the consistency check and report what
it did (DESIGN.md §10, phase 2: rule firing counts and false alarms).

    python scripts/eval_consistency.py --match "2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea" [--run v1]

Reads scripts/run_events.py output (outputs/events/<match>/[<run>/]half{1,2}.jsonl; the API
answers come from cache/api/) and replays it through memory as the offline runner would.

Scored against the match's ground truth on the types the captions cover completely (goal,
yellow card, substitution): for each type, how many entries sit on a true event, how many
are extra (duplicates or inventions), how many of the extras a constraint holds in an open
case, and how many entries end up confirmed; plus the final score in memory's view.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from eval_events import ground_truth   # scripts/eval_events.py
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.memory.types import Event
from matchlive.runtime.runner import replay_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="", help="subdirectory of an earlier run, e.g. v1")
    ap.add_argument("--slack", type=float, default=10.0, help="± seconds around an entry's interval")
    args = ap.parse_args()

    config = load_config()
    match = Match(config, args.match)
    run_dir = config["paths"]["outputs"] / "events" / args.match / args.run
    results = {h: [json.loads(line) for line in open(run_dir / f"half{h}.jsonl")] for h in (1, 2)}
    # The consistency check alone: every case is left open (the arbiter is evaluated separately).
    mem, _ = replay_events(config, match, results, decide=lambda *_: None)

    events = {eid: e for eid, e in mem.entries.items() if isinstance(e, Event) and mem.status(eid) != "retracted"}
    n_cand = sum(len(r.get("candidates", [])) for rows in results.values() for r in rows)
    print(f"{n_cand} candidates -> {len(events)} event entries: "
          + ", ".join(f"{k} {v}" for k, v in Counter(mem.status(e) for e in events).items()))
    print("cases per constraint: " + ", ".join(f"{k} {v}" for k, v in
                                               Counter(c.constraint for c in mem.cases.values()).most_common()))

    truth = ground_truth(match.dir)
    in_case = {m for c in mem.cases.values() if c.status == "open" for m in c.members}
    print(f"\n{'type':13s} {'true':>5s} {'entries':>8s} {'on truth':>9s} {'extra':>6s} {'extra in case':>14s} {'confirmed':>10s}")
    for ty in ("goal", "yellow_card", "substitution"):
        times = [t["time"] for t in truth if t["type"] == ty]
        ents = [eid for eid, e in events.items() if e.type == ty]
        # One entry per true event: the narrowest covering entry (a live sighting before a
        # replay's wide interval), nearest first; every other entry is extra.
        on_truth = set()
        for h, s in times:
            covering = []
            for eid in ents:
                (h0, lo), (_, hi) = events[eid].t_interval
                if eid not in on_truth and h0 == h and lo - args.slack <= s <= hi + args.slack:
                    covering.append((hi - lo, abs((lo + hi) / 2 - s), eid))
            if covering:
                on_truth.add(min(covering)[2])
        extra = [eid for eid in ents if eid not in on_truth]
        print(f"{ty:13s} {len(times):5d} {len(ents):8d} {len(on_truth):9d} {len(extra):6d} "
              f"{sum(e in in_case for e in extra):14d} {sum(mem.status(e) == 'confirmed' for e in ents):10d}")
    print(f"\nfinal score in memory: {mem.at((3, 0.0)).score}  (true: {match.metadata['score']})")


if __name__ == "__main__":
    main()
