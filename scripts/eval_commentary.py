"""Score the commentary of live runs against the ground truth: what it said as fact, what it
hedged, which players it named, and what it corrected (DESIGN.md §6.9).

    python scripts/eval_commentary.py [--runs full] [--match M ...]

Claims come from outputs/commentary/<match>/<run>/lines.jsonl (scripts/run_commentary.py). A line
about an event entry claims the event happened — as fact if the entry was confirmed when it was
said, hedged if pending — and names the players memory had identified on a confirmed entry. An
entry is true when it matches a true event (scripts/eval_memory.assign on the run's final memory);
a named player is checked where the labels name one (goals, cards, substitutions). A correction
is right when the entry it retracts is false, or when the player it names instead is the true one.

Next to it, the same claims from a status-blind commentator (v2-like): every event entry of an
announced type aired as fact the moment memory writes it, naming the players memory resolves
then, never corrected — read straight from the op log, no model call.

The generated text itself is checked for names the material did not give: a roster player named
in a line but in none of its facts is the generator adding a name.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from eval_memory import MATCHES, assign                 # scripts/eval_memory.py
from ground_truth import _mentions, ground_truth         # scripts/ground_truth.py
from matchlive.app.commentary import ROLE_WORDS
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, Op

IMPORTANT = ("goal", "red_card", "penalty", "yellow_card", "substitution")


def claims(lines, true_of, truth, events) -> Counter:
    c = Counter()
    for line in lines:
        eid = line["entry"]
        if eid is None or eid not in events:
            continue
        right = eid in true_of
        g = truth[true_of[eid]] if right else None
        if line["kind"] in ("news", "confirmation"):
            mode = "fact" if line["status"] == "confirmed" else "hedged"
            c[mode] += 1
            c[f"{mode}_false"] += not right
            if events[eid].type in IMPORTANT:
                c[f"{mode}_important"] += 1
                c[f"{mode}_important_false"] += not right
            _names(c, line["named"], g, events[eid])
        elif line["kind"] == "correction":
            if line["status"] == "retracted":
                c["corrected_retraction"] += 1
                c["corrected_retraction_right"] += not right
            else:
                c["corrected_player"] += 1
                c["corrected_player_right"] += bool(g) and all(g["players"].get(r) == p for r, p in line["named"].items())
    return c


def _names(c: Counter, named: dict, g: dict | None, e: Event) -> None:
    for role, pid in named.items():
        if g is not None and g["type"] == e.type and g["players"].get(role):
            c["named_checked"] += 1
            c["named_wrong"] += g["players"][role] != pid
        else:
            c["named_unchecked"] += 1


def blind_claims(ops, roster, cascade, announce, true_of, truth) -> Counter:
    """The status-blind commentator: every event entry aired as fact when written."""
    c = Counter()
    mem = MatchMemory(roster, cascade=cascade)
    for op in ops:
        mem._apply(op)                               # the same path a log replay takes
        if op.kind != "add_entry" or op.payload["entry"]["kind"] != "event":
            continue
        eid = op.payload["entry_id"]
        e = mem.entries[eid]
        if e.type not in announce:
            continue
        right = eid in true_of
        c["fact"] += 1
        c["fact_false"] += not right
        if e.type in IMPORTANT:
            c["fact_important"] += 1
            c["fact_important_false"] += not right
        named = {p.role: pid for p, pid in zip(e.participants, mem.event_players(eid)) if pid and p.role in ROLE_WORDS}
        _names(c, named, truth[true_of[eid]] if right else None, e)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["full"])
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    args = ap.parse_args()
    base = load_config()

    rows = {}
    for run in args.runs:
        ours, blind, text, lags = Counter(), Counter(), Counter(), []
        for name in args.match or MATCHES:
            run_dir = base["paths"]["outputs"] / "live" / name / run
            path = base["paths"]["outputs"] / "commentary" / name / run / "lines.jsonl"
            if not path.exists():
                print(f"(no commentary for {run} on {name})")
                continue
            config = load_config(overrides=run_overrides(run_dir))
            match = Match(config, name)
            truth = ground_truth(match)
            ops = [Op(d["op_id"], tuple(d["t_known"]), d["kind"], d["payload"]) for d in map(json.loads, open(run_dir / "ops.jsonl"))]
            mem = MatchMemory(match.prematch["players"], cascade=config["loop"]["cascade"])
            for op in ops:
                mem._apply(op)
            events = {eid: e for eid, e in mem.entries.items() if isinstance(e, Event) and e.type != "injury"}
            true_of = assign(mem, events, truth)
            lines = [json.loads(line) for line in open(path)]
            ours += claims(lines, true_of, truth, events)
            blind += blind_claims(ops, match.prematch["players"], config["loop"]["cascade"],
                                  config["commentary"]["announce"], true_of, truth)
            for line in lines:
                text["lines"] += 1
                extra = set(_mentions(line["text"], match.prematch["players"])) \
                    - set(_mentions("\n".join(line["facts"]), match.prematch["players"]))
                text["added_names"] += bool(extra)
                if line["entry"] in true_of and line["kind"] == "news":
                    g = truth[true_of[line["entry"]]]
                    lags.append(line["t_ready"][1] - g["time"][1] if line["t_ready"][0] == g["time"][0] else np.nan)
        rows[run] = (ours, blind, text, lags)

    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "    -"

    for run, (ours, blind, text, lags) in rows.items():
        print(f"\n== {run}: {text['lines']} lines; lines naming a player the material did not give: "
              f"{text['added_names']} ({pct(text['added_names'], text['lines']).strip()})")
        print(f"{'':28s} {'claims as fact':>15s} {'false':>7s} {'goals/cards/subs as fact':>25s} {'false':>7s} "
              f"{'hedged':>7s} {'false':>7s} {'names checked':>14s} {'wrong':>7s}")
        for label, c in (("status-aware (ours)", ours), ("status-blind (v2-like)", blind)):
            print(f"{label:28s} {c['fact']:15d} {pct(c['fact_false'], c['fact']):>7s} {c['fact_important']:25d} "
                  f"{pct(c['fact_important_false'], c['fact_important']):>7s} {c['hedged']:7d} "
                  f"{pct(c['hedged_false'], c['hedged']):>7s} {c['named_checked']:14d} {pct(c['named_wrong'], c['named_checked']):>7s}")
        print(f"corrections on air: {ours['corrected_retraction']} retractions ({ours['corrected_retraction_right']} of "
              f"a false entry), {ours['corrected_player']} player changes ({ours['corrected_player_right']} to the true player)")
        lags = [x for x in lags if not np.isnan(x)]
        if lags:
            print(f"true event -> news line ready (live): p50 {np.median(lags):.0f} s, p90 {np.percentile(lags, 90):.0f} s")


if __name__ == "__main__":
    main()
