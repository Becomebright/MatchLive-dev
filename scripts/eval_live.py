"""Summarise a live run (scripts/run_live.py): model calls, memory, cases and verdicts.

    python scripts/eval_live.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--run full]

Reads outputs/live/<match>/<run>/ and prints: analyzer windows (answered, failed, latency
percentiles, candidates, replay-flagged ones); event entries by type and status at the final
whistle; cases by constraint with what became of them (verdict applied, void, no verdict); the
score and the players on the pitch in memory at the end, next to the final score from metadata
(evaluation only — the run never saw it).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full")
    args = ap.parse_args()

    run_dir = load_config()["paths"]["outputs"] / "live" / args.match / args.run
    config = load_config(overrides=run_overrides(run_dir))
    match = Match(config, args.match)
    rows = [json.loads(line) for line in open(run_dir / "windows.jsonl")]
    # Re-checks (loop.recheck) are logged with the windows; counted apart here.
    windows = [w for w in rows if "recheck" not in w]
    rechecks = [w for w in rows if "recheck" in w]
    ok = [w for w in windows if "error" not in w]
    lat = np.array([w["latency"] for w in ok])
    cands = [c for w in ok for c in w["candidates"]]
    print(f"windows {len(windows)}: answered {len(ok)}, failed {len(windows) - len(ok)}; latency s p50 "
          f"{np.percentile(lat, 50):.1f} p90 {np.percentile(lat, 90):.1f} max {lat.max():.1f}; "
          f"candidates {len(cands)} ({sum(c['replay'] for c in cands)} flagged replay)")

    mem = MatchMemory(match.prematch["players"], log_path=run_dir / "ops.jsonl", cascade=config["loop"]["cascade"])
    if rechecks:
        seen = sum(any(j.kind == "recheck" for _, j in js) for js in mem.justifications.values())
        print(f"re-checks {len(rechecks)}: saw the entry again {seen}, failed {sum('error' in w for w in rechecks)}")
    by_type = defaultdict(Counter)
    for eid, e in mem.entries.items():
        if isinstance(e, Event):
            by_type[e.type][mem.status(eid)] += 1
    print("event entries (confirmed / pending / retracted):")
    for ty, c in sorted(by_type.items()):
        print(f"  {ty:13s} {c['confirmed']:3d} / {c['pending']:3d} / {c['retracted']:3d}")

    records = {r["case"]: r for r in json.load(open(run_dir / "cases.json"))} if (run_dir / "cases.json").exists() else {}
    outcome = defaultdict(Counter)
    for case in mem.cases.values():
        r = records.get(case.case_id)
        if case.status == "closed":
            what = "verdict applied"
        elif r is None:
            what = "not sent (below k)" if case.constraint == "off_roster_read" else "not sent"
        elif r.get("void"):
            what = "void: " + str(r["void"]).split(":")[0]
        else:
            what = "no verdict"
        outcome[case.constraint][what] += 1
    print(f"cases {len(mem.cases)}:")
    for constraint, c in sorted(outcome.items()):
        print(f"  {constraint:30s} " + ", ".join(f"{k} {v}" for k, v in c.items()))
    ops = Counter(a.get("op") for r in records.values() if r.get("verdict") and not r.get("void")
                  for a in r["verdict"].get("actions", []) if isinstance(a, dict))
    calls = [r["calls"] for r in records.values()]
    print(f"verdict writes: {dict(ops)}; arbiter calls per case {np.mean(calls) if calls else 0:.1f}, "
          f"latency s p50 {np.median([r['latency'] for r in records.values()]) if records else 0:.0f}")

    end = mem.at((3, 0.0))
    teams = match.prematch["teams"]
    print(f"score in memory at the final whistle: {teams['home']} {end.score['home']} - {end.score['away']} "
          f"{teams['away']} (final score: {match.metadata['score']})")
    names = {p["hash"]: p["short_name"] for p in match.prematch["players"]}
    for r in end.events(type="substitution"):
        print(f"  substitution {r.status}: {teams.get(r.event.team, '?')} at {r.event.t_interval[1][0]}H "
              f"{np.mean([r.event.t_interval[0][1], r.event.t_interval[1][1]]) / 60:.1f}' "
              + ", ".join(f"{p.role} {names.get(pid, '?') if pid else '?'}" for p, pid in zip(r.event.participants, r.players)))


if __name__ == "__main__":
    main()
