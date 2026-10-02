"""Streaming-QA ablation table (docs/results.md §17): accuracy of each QA method under each QA model,
both matches together, per question group.

    python scripts/qa_ablation.py [--tag 1001] [--models gemini-3.8-flash gpt-6.1-sol claude-sonnet-5-5]

Reads the result files scripts/run_qa.py writes with --tag and --model. Each method reads one live
run per match: events reads the loop.consistency=false run, the others the full system's run
full_pool (claims that agree pool their participants, DESIGN.md §6.4). A missing result file
leaves its row out. Prints two Markdown tables: one column per subcategory of the report's question
taxonomy (plus each match's accuracy and the unanswered count), and one per dimension (D1-D4). Each
table starts with the question count of every column and ends with the always-"A" baseline.
"""

from __future__ import annotations

import argparse
import json

from matchlive.common.config import load_config

# (match, run of the full system); the events method reads no_consistency instead.
MATCHES = [("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "full_pool"),
           ("2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool", "full_pool")]
METHODS = ["blind", "video", "events", "dump_log", "tools"]
# The report's question taxonomy (Table 1): benchmark subcategories grouped as the report counts them.
# Player attribution includes the foul committer and substitution pair questions.
SUBCATEGORIES = [("D1 event count", ["counting"]),
                 ("D2 event time", ["recall_event_time"]),
                 ("D2 player identity", ["recall_event_player"]),
                 ("D3 player attribution", ["event_player", "infraction_actor", "substitution_pair"]),
                 ("D3 team attribution", ["event_team"]),
                 ("D3 event outcome", ["event_outcome"]),
                 ("D4 causal sequence", ["event_sequence"])]
# The four dimensions are the benchmark's own categories.
DIMENSIONS = [("D1", "dim1"), ("D2", "dim2"), ("D3", "dim3"), ("D4", "dim4")]


def acc(rows: list[dict]) -> str:
    return f"{100 * sum(r['choice'] == r['correct'] for r in rows) / len(rows):.1f}" if rows else "-"


def always_a(rows: list[dict]) -> str:
    return f"{100 * sum(r['correct'] == 'A' for r in rows) / len(rows):.1f}" if rows else "-"


def table(title: str, groups: list, results: list, per_match: bool) -> None:
    """One Markdown table: a row per (model, method), a column per group, then the overall accuracy
    and, with `per_match`, each match's accuracy and the unanswered count. `groups` pairs a column
    name with a test on a result row; `results` holds (model, method, each match's result rows)."""
    def row(label: str, method: str, cells: list[str]) -> None:
        print(f"| {label} | {method} | " + " | ".join(cells) + " |")

    names = [name for name, _ in groups] + ["**all**"] + (["MU", "LIV", "unanswered"] if per_match else [])
    print(f"\n{title}\n")
    row("model", "method", names)
    print("|---" * (len(names) + 2) + "|")
    # Every method answers the same questions, so any result gives the question counts and the
    # always-"A" baseline.
    matches = results[0][2]
    every = [r for rs in matches for r in rs]
    row("N", "", [str(sum(test(r) for r in every)) for _, test in groups] + [str(len(every))]
        + ([*(str(len(rs)) for rs in matches), ""] if per_match else []))
    for model, method, matches_m in results:
        rows = [r for rs in matches_m for r in rs]
        row(model, method, [acc([r for r in rows if test(r)]) for _, test in groups] + [f"**{acc(rows)}**"]
            + ([*(acc(rs) for rs in matches_m), str(sum(r["choice"] is None for r in rows))] if per_match else []))
    row("always A", "", [always_a([r for r in every if test(r)]) for _, test in groups] + [always_a(every)]
        + ([*(always_a(rs) for rs in matches), ""] if per_match else []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1001")
    ap.add_argument("--models", nargs="+", default=["gemini-3.8-flash", "gpt-6.1-sol", "claude-sonnet-5-5"])
    args = ap.parse_args()
    qa_dir = load_config()["paths"]["outputs"] / "qa"

    # (model, method, each match's result rows), in table order.
    results = []
    for model in args.models:
        for method in METHODS:
            per_match = []
            for match, run in MATCHES:
                path = qa_dir / match / ("no_consistency" if method == "events" else run) / f"{method}_{args.tag}_{model}.jsonl"
                if path.exists():
                    per_match.append([json.loads(line) for line in open(path)])
            if len(per_match) == len(MATCHES):
                results.append((model, method, per_match))
    if not results:
        print("no result files found")
        return

    table("By subcategory (accuracy, %)",
          [(name, lambda r, subs=subs: r["subcategory"] in subs) for name, subs in SUBCATEGORIES], results,
          per_match=True)
    table("By dimension (accuracy, %)",
          [(name, lambda r, cat=cat: r["category"] == cat) for name, cat in DIMENSIONS], results,
          per_match=False)


if __name__ == "__main__":
    main()
