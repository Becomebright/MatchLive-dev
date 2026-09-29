"""Visual check of what the arbiter would see for one case, without calling any model.

    python scripts/check_arbiter.py --match "2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea" [--case C3]

Replays the cached event candidates through the consistency check (cases left open), lists
the cases, and for the chosen one (default: the first) writes outputs/check_arbiter/<case>/:
packet.txt (round 1 of the arbiter's input) and one file per evidence tool run on its first
member (frames around it, and the crops / jersey reads of its first participant track).
"""

from __future__ import annotations

import argparse
import json

from matchlive.arbiter.agent import build_packet
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.memory.types import Event
from matchlive.runtime.evidence import EvidenceTools
from matchlive.runtime.runner import observations, replay_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--case")
    args = ap.parse_args()

    config = load_config()
    match = Match(config, args.match)
    run_dir = config["paths"]["outputs"] / "events" / args.match
    results = {h: [json.loads(line) for line in open(run_dir / f"half{h}.jsonl")] for h in (1, 2)}
    mem, _ = replay_events(config, match, results, decide=lambda *_: None)
    for c in mem.cases.values():
        print(f"{c.case_id:5s} {c.constraint:30s} {', '.join(c.members)}")

    case = mem.cases[args.case] if args.case else next(iter(mem.cases.values()))
    # The case opened with its op; knowledge up to then is what the arbiter may use.
    t_bound = next(op.t_known for op in mem.ops if op.kind == "open_case" and op.payload["case_id"] == case.case_id)
    out = config["paths"]["outputs"] / "check_arbiter" / case.case_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "packet.txt").write_text(build_packet(mem, case, t_bound, observations(results), match.prematch,
                                                 config["arbiter"]["context"]))
    print(f"\n{case.case_id} opened at {t_bound}; packet -> {out / 'packet.txt'}")

    tools = EvidenceTools(config, match)
    first = next((mem.entries[m] for m in case.members if isinstance(mem.entries.get(m), Event)), None)
    requests = []
    if first is not None:
        (h, lo), (_, hi) = first.t_interval
        requests.append(("get_frames", {"half": h, "start": max(0, lo - 5), "end": hi + 5}))
        requests.append(("list_events", {"half": h, "start": lo - 60, "end": hi + 60}))
        track = next((p.track_key for p in first.participants if p.track_key), None)
        if track:
            requests += [("get_track_crops", {"track": list(track)}), ("jersey_reads", {"track": list(track)})]
        requests.append(("why", {"entry": case.members[0]}))
    for n, (tool, targs) in enumerate(requests, 1):
        ev = tools.call(mem, t_bound, tool, targs)
        (out / f"{n}_{tool}.txt").write_text(ev.text)
        for k, jpg in enumerate(ev.images):
            (out / f"{n}_{tool}_{k:02d}.jpg").write_bytes(jpg)
        print(f"{tool} {targs}: {ev.text[:150]} ({len(ev.images)} images)")


if __name__ == "__main__":
    main()
