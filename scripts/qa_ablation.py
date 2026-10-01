"""Streaming-QA ablation table (docs/results.md §17): accuracy of each QA method under each QA model,
both matches together, per question group.

    python scripts/qa_ablation.py [--tag 1001] [--models gemini-3.8-flash gpt-6.1-sol claude-sonnet-5-5]

Reads the result files scripts/run_qa.py writes with --tag and --model. Each method reads one live
run per match: events reads the loop.consistency=false run, the others the full system's run
full_pool (claims that agree pool their participants, DESIGN.md §6.4). A missing result file
leaves its row out. Prints a Markdown table.
"""

from __future__ import annotations

import argparse
import json

from matchlive.common.config import load_config

# (match, run of the full system); the events method reads no_consistency instead.
MATCHES = [("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "full_pool"),
           ("2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool", "full_pool")]
METHODS = ["blind", "video", "events", "dump_log", "tools"]
# Question groups as in results §16; the last four subcategories are the 1001 version's new types.
GROUPS = [("counting", ["counting"]), ("recall time", ["recall_event_time"]), ("recall player", ["recall_event_player"]),
          ("current player", ["event_player"]), ("current team", ["event_team"]),
          ("new types", ["event_outcome", "infraction_actor", "substitution_pair", "event_sequence"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1001")
    ap.add_argument("--models", nargs="+", default=["gemini-3.8-flash", "gpt-6.1-sol", "claude-sonnet-5-5"])
    args = ap.parse_args()
    qa_dir = load_config()["paths"]["outputs"] / "qa"

    print("| model | method | " + " | ".join(g for g, _ in GROUPS) + " | **all** | MU | LIV | unanswered |")
    print("|---" * (len(GROUPS) + 6) + "|")
    for model in args.models:
        for method in METHODS:
            per_match = []
            for match, run in MATCHES:
                path = qa_dir / match / ("no_consistency" if method == "events" else run) / f"{method}_{args.tag}_{model}.jsonl"
                if path.exists():
                    per_match.append([json.loads(line) for line in open(path)])
            if len(per_match) < len(MATCHES):
                continue
            rows = [r for rs in per_match for r in rs]

            def acc(rs):
                return f"{sum(r['choice'] == r['correct'] for r in rs) / len(rs):.3f}"

            cells = [acc([r for r in rows if r["subcategory"] in subs]) for _, subs in GROUPS]
            cells += [f"**{acc(rows)}**", *(acc(rs) for rs in per_match), str(sum(r["choice"] is None for r in rows))]
            print(f"| {model} | {method} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
