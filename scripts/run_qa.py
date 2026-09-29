"""Run the streaming-QA benchmark on one match (DESIGN.md §10 phase 5).

    python scripts/run_qa.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" [--run full] [--mode tools]

Each question is asked at its benchmark timestamp, turned into match time (half, second) through the
match's offset table (DESIGN.md §4), and answered from what the live run (scripts/run_live.py,
outputs/live/<match>/<run>/) knew at that moment: memory as known by then, perception and video up
to then. The model sees the question and its options and nothing else from the benchmark (its match
key carries the final score).

Writes outputs/qa/<match>/<run>/<mode>.jsonl, one line per question, and prints accuracy per
subcategory with the predicted-letter distribution and the always-"A" baseline (dim1's answers
lean to A by construction, docs.md §8).
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from matchlive.app.qa import QAAgent, question_text
from matchlive.common.api import APIClient
from matchlive.common.config import REPO_ROOT, load_config
from matchlive.common.match import Match
from matchlive.events.analyzer import clock
from matchlive.memory.store import MatchMemory
from matchlive.runtime.qa_tools import QATools, minute_phrase


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--run", default="full", help="live run name (scripts/run_live.py --name)")
    ap.add_argument("--mode", default=None,
                    help="tools | dump_log (default: qa.mode) | blind (a sanity check: no memory, no video)")
    ap.add_argument("--benchmark", default="dataset/streamingqa/part1/benchmark-0926.jsonl")
    ap.add_argument("--workers", type=int, default=8, help="questions answered at once")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (a smoke test)")
    ap.add_argument("--half", type=int, default=None,
                    help="only the questions asked in this half; the first half can be answered while the live "
                         "run is still playing the second (memory as known by a first-half question is complete)")
    args = ap.parse_args()

    config = load_config()
    mode = args.mode or config["qa"]["mode"]
    match = Match(config, args.match)
    run_dir = config["paths"]["outputs"] / "live" / args.match / args.run
    # Read from a copy of the complete lines: the live run may still be appending to the log.
    lines = open(run_dir / "ops.jsonl").read().split("\n")[:-1]
    snapshot = Path(tempfile.mkdtemp()) / "ops.jsonl"
    snapshot.write_text("".join(line + "\n" for line in lines))
    mem = MatchMemory(match.prematch["players"], log_path=snapshot, cascade=config["loop"]["cascade"])
    perceived = {h: pickle.load(open(run_dir / f"perceived_half{h}.pkl", "rb"))
                 for h in (1, 2) if (run_dir / f"perceived_half{h}.pkl").exists()}
    sources = {f"obs:{r['call_id']}#{i}": {"window": r["window"], "candidate": c}
               for r in map(json.loads, open(run_dir / "windows.jsonl")) for i, c in enumerate(r.get("candidates", []))}
    tools = QATools(config, match, mem, perceived, sources)
    client = APIClient(config["api"], config["paths"]["cache"])
    agent = QAAgent(config["qa"], lambda parts: client.generate(parts, json_output=True))

    offsets = match.benchmark_offsets

    def asked_at(q):
        ts = q["timestamp"]
        return (2, float(ts - offsets[2])) if ts >= offsets[2] else (1, float(ts))

    rows = [(i, q) for i, q in enumerate(map(json.loads, open(REPO_ROOT / args.benchmark)))
            if q["match_key"] == match.metadata["match_key"] and args.half in (None, asked_at(q)[0])][: args.limit]
    teams, kits = match.prematch["teams"], match.prematch["jersey_colors"]
    match_text = (f"The match: {teams['home']} (home, {kits['home'][0]} kit) vs {teams['away']} "
                  f"(away, {kits['away'][0]} kit).")

    def ask(item):
        i, q = item
        now = asked_at(q)
        text = question_text(q["question"], q["options"],
                             f"half {now[0]}, {clock(now[1])} ({minute_phrase(*now)})", match_text)
        letters = [o.split(".")[0] for o in q["options"]]
        try:
            if mode == "tools":
                a = agent.answer(text, letters, lambda tool, targs: tools.call(now, tool, targs))
            elif mode == "dump_log":
                a = agent.answer_from_log(text, letters, tools.dump_log(now))
            else:   # blind: nothing about the match — what the question and options alone give away
                a = agent.answer_from_log(text, letters, "(nothing has been recorded)")
            out = {"choice": a.choice, "calls": a.calls, "latency": a.latency, "tools": a.tools}
        except Exception as e:  # noqa: BLE001 — out of API retries: the question goes unanswered
            out = {"choice": None, "error": f"{type(e).__name__}: {e}"}
        return {"line": i, "category": q["category"], "subcategory": q["subcategory"], "question": q["question"],
                "game_time": q["game_time"], "correct": q["correct_choice"], **out}

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = []
        for n, r in enumerate(pool.map(ask, rows), 1):
            results.append(r)
            if n % 50 == 0:
                print(f"{n}/{len(rows)} answered, {time.time() - t0:.0f} s", flush=True)

    name = mode if args.half is None else f"{mode}_half{args.half}"
    out_path = config["paths"]["outputs"] / "qa" / args.match / args.run / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(out_path)
    by = defaultdict(list)
    for r in results:
        by[f"{r['category']}/{r['subcategory']}"].append(r)
    print(f"{'subcategory':32s} {'n':>4s} {'acc':>6s} {'always A':>9s}  predicted letters")
    for key, rs in sorted(by.items()) + [("all", results)]:
        acc = sum(r["choice"] == r["correct"] for r in rs) / len(rs)
        base = sum(r["correct"] == "A" for r in rs) / len(rs)
        print(f"{key:32s} {len(rs):4d} {acc:6.3f} {base:9.3f}  {dict(Counter(r['choice'] for r in rs))}")
    errors = sum("error" in r for r in results)
    calls = [r["calls"] for r in results if "calls" in r]
    print(f"unanswered {sum(r['choice'] is None for r in results)} (API errors {errors}); "
          f"calls per question {sum(calls) / max(len(calls), 1):.2f}; tools used "
          f"{dict(Counter(t for r in results for t in r.get('tools', [])).most_common())}")


if __name__ == "__main__":
    main()
