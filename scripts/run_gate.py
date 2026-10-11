"""The confidence-gating baseline (DESIGN.md §10, MemTX-like) on the candidates of a live run.

    python scripts/run_gate.py [--source confidence] [--gates 0.85 0.9 0.95 0.98] [--match M ...]

The source run (scripts/run_live.py --set events.ask_confidence=true) is the full system whose
event analyzer also gave a 0-1 confidence per event. Its analyzer answers are replayed through the
consistency check in gating mode (consistency.gate = τ, runtime.runner.replay_events): a sighting
with confidence ≥ τ is written as confirmed, any other is dropped; no statuses from
justifications, no constraints, no arbiter. Same candidates, same timing (each answer at its
window end + measured latency), so the only difference from the source run is how memory decides
what to believe. Writes outputs/live/<match>/<source>_gate<τ>/ (ops.jsonl, windows.jsonl,
run.json) for scripts/eval_memory.py; no model is called.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict

from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.runtime.runner import replay_events

MATCHES = ("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="confidence", help="live run whose analyzer gave confidences")
    # The analyzer's confidences cluster at 0.85-0.95 (203 of 313 are 0.95), so gates below 0.85 keep
    # everything and 0.98 keeps almost nothing.
    ap.add_argument("--gates", nargs="+", type=float, default=[0.85, 0.9, 0.95, 0.98])
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    args = ap.parse_args()

    for name in args.match or MATCHES:
        source = load_config()["paths"]["outputs"] / "live" / name / args.source
        rows = [json.loads(line) for line in open(source / "windows.jsonl")]
        # Windows only: a re-check's row (loop.recheck) is no window of its own.
        rows = [r for r in rows if "recheck" not in r]
        results = {h: [r for r in rows if r["half"] == h and "candidates" in r] for h in (1, 2)}
        confs = [c["confidence"] for r in rows for c in r.get("candidates", [])]
        print(f"{name}: {len(confs)} candidates, {sum(c is None for c in confs)} without a confidence")
        for gate in args.gates:
            overrides = run_overrides(source) + [f"consistency.gate={gate}"]
            config = load_config(overrides=overrides)
            mem, _ = replay_events(config, Match(config, name), results)
            out = source.parent / f"{args.source}_gate{gate}"
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "ops.jsonl", "w") as f:
                for op in mem.ops:
                    f.write(json.dumps(asdict(op)) + "\n")
            shutil.copy(source / "windows.jsonl", out / "windows.jsonl")
            (out / "run.json").write_text(json.dumps({"set": overrides, "source": args.source}, indent=1))
            kept = sum(c is not None and c >= gate for c in confs)
            print(f"  gate {gate}: {kept} candidates kept, {len(mem.entries)} entries -> {out.name}")


if __name__ == "__main__":
    main()
