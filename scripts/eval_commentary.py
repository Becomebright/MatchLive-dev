"""Score the commentary of live runs against the ground truth: what it said as fact, what it
hedged, which players it named, and what it corrected (DESIGN.md §6.9).

    python scripts/eval_commentary.py [--runs full ...] [--match M ...]

Claims come from outputs/commentary/<match>/<run>/lines.jsonl (scripts/run_commentary.py). A line
about an event entry claims the event happened — as fact if the entry was confirmed when it was
said, hedged if pending — and names the players memory had identified on a confirmed entry. An
entry is true when it matches a true event (scripts/eval_memory.assign on the run's final memory),
or a verdict replaced it with a fuller entry that does (the same event, with who took part);
a named player is checked where the labels name one (goals, cards, substitutions). A correction
is right when the entry it retracts is false, or when the player it names instead is the true one.

Next to it, the same claims from a status-blind commentator (v2-like): every event entry of an
announced type aired as fact the moment memory writes it, naming the players memory resolves
then, never corrected — read straight from the op log, no model call.

The generated text itself is checked for names the material did not give: a roster player named
in a line but in none of its facts (nor in who-is-who of the frames a play line is written over)
is the generator adding a name.

Pre-match background (DESIGN.md §6.8): how many lines were given a fact, and how many used it (a
word of the fact, such as a name or a number, that the rest of the line's material does not have).

Coverage: the visible true events some line spoke of (news or a confirmation). Style, next to the
broadcast's own commentary (scripts/transcribe_broadcast.py): the share of match time spoken, words
per minute of match, lines per minute, words per line, lexical variety (distinct word pairs) and
lines opening like one of the ten before them. A play line said as a burst of short phrases counts
as its phrases.

Timing, next to the broadcast's commentary too: the visible true shots and goals some line calls in
time — starting within [-1, +3] s of the labelled moment and saying something a shot call says (a
shot, a save, wide, blocked, ...) — how long after the frame it was written for a phrase starts,
the speaking rate (words per second of each voiced line, or each transcript segment), and the
energy (tone) lines are said with.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np

from eval_memory import MATCHES, assign                 # scripts/eval_memory.py
from ground_truth import _mentions, ground_truth         # scripts/ground_truth.py
from matchlive.app.commentary import ROLE_WORDS, replacement
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, Op

IMPORTANT = ("goal", "red_card", "penalty", "yellow_card", "substitution")
# What calling a shot as it happens sounds like (ours and the broadcast's, scored the same way).
SHOT_CALL = re.compile(r"\b(shot|shoots?|strikes?|struck|effort|saved?|saves|wide|blocked|blocks?|header|heads?|"
                       r"fires?|volley|drives?|curls?|goal|scores?|chance|keeper|post|crossbar|over the bar)\b", re.I)


def called_in_time(said: list[tuple[int, float, str]], when: tuple[int, float]) -> bool:
    """Whether one of `said` ((half, start, text)) calls the shot at `when` as it happens."""
    half, t = when
    return any(h == half and t - 1 <= start <= t + 3 and SHOT_CALL.search(text) for h, start, text in said)


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


def style(texts: list[str], speech: float, seconds: float) -> dict:
    """How much and how varied the talk is: share of match time spoken, words per minute of match,
    lines per minute, words per line, distinct word pairs / all word pairs, and the share of lines
    that open with the same two words as one of the ten before them."""
    words = [t.lower().replace(",", " ").replace(".", " ").replace("!", " ").replace("?", " ").split() for t in texts]
    pairs = [(a, b) for w in words for a, b in zip(w, w[1:])]
    openings = [tuple(w[:2]) for w in words]
    repeats = sum(o in openings[max(0, i - 10):i] for i, o in enumerate(openings))
    n = sum(len(w) for w in words)
    return {"speech": speech / seconds, "wpm": n / (seconds / 60), "lpm": len(texts) / (seconds / 60),
            "words_per_line": n / max(len(texts), 1), "distinct2": len(set(pairs)) / max(len(pairs), 1),
            "same_opening": repeats / max(len(texts), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["full"],
                    help="commentary folders under outputs/commentary/<match>/ (scripts/run_commentary.py --name)")
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    args = ap.parse_args()
    base = load_config()
    out = base["paths"]["outputs"]

    rows, styles, timing = {}, {}, {}
    human_texts, human_speech, total_seconds = [], 0.0, 0.0
    human_timing = {"shots": 0, "called": 0, "rate": []}
    for run in args.runs:
        ours, blind, text, lags, told = Counter(), Counter(), Counter(), [], set()
        texts, speech = [], 0.0
        seconds = 0.0
        timing[run] = {"shots": 0, "called": 0, "rate": [], "after_frame": [], "tone": Counter()}
        for name in args.match or MATCHES:
            folder = out / "commentary" / name / run
            if not (folder / "lines.jsonl").exists():
                print(f"(no commentary {run} on {name})")
                continue
            # The live run the commentary was made on (its memory is what is scored).
            live = json.loads((folder / "run.json").read_text())["run"] if (folder / "run.json").exists() else run
            run_dir = out / "live" / name / live
            config = load_config(overrides=run_overrides(run_dir))
            match = Match(config, name)
            truth = ground_truth(match)
            ops = [Op(d["op_id"], tuple(d["t_known"]), d["kind"], d["payload"]) for d in map(json.loads, open(run_dir / "ops.jsonl"))]
            mem = MatchMemory(match.prematch["players"], cascade=config["loop"]["cascade"])
            for op in ops:
                mem._apply(op)
            events = {eid: e for eid, e in mem.entries.items() if isinstance(e, Event) and e.type != "injury"}
            true_of = assign(mem, events, truth)
            # A line about an entry a verdict replaced with a fuller one (the same event, with who
            # took part) speaks of the event the new entry matches.
            true_of.update({eid: true_of[new] for eid in events if eid not in true_of
                            for new in [replacement(mem, eid)] if new in true_of})
            lines = [json.loads(line) for line in open(folder / "lines.jsonl")]
            ours += claims(lines, true_of, truth, events)
            blind += blind_claims(ops, match.prematch["players"], config["loop"]["cascade"],
                                  config["commentary"]["announce"], true_of, truth)
            # Coverage: the visible true events some line spoke of.
            spoken = {true_of[l["entry"]] for l in lines if l["entry"] in true_of and l["kind"] in ("news", "confirmation")}
            for i, g in enumerate(truth):
                if g["visible"] and g["type"] in config["commentary"]["announce"]:
                    key = "important" if g["type"] in IMPORTANT else "minor"
                    text[f"true_{key}"] += 1
                    text[f"spoken_{key}"] += i in spoken
            windows = [json.loads(l) for l in open(run_dir / "windows.jsonl")]
            # The stretch of video the commentary covers: whole halves, or a trial's (half, start, end).
            only = json.loads((folder / "run.json").read_text()).get("only") if (folder / "run.json").exists() else None
            ranges = [tuple(only)] if only else [(h, 0.0, max(w["window"][1] for w in windows if w["half"] == h)) for h in (1, 2)]
            match_seconds = sum(b - a for _, a, b in ranges)
            seconds += match_seconds
            # A play line said as a burst of phrases counts phrase by phrase (as the broadcast's
            # transcript is cut at its pauses).
            texts += [t for l in lines for t in (l.get("phrases") or [l["text"]])]
            speech += sum(l.get("duration") or 0.0 for l in lines)
            for line in lines:
                text["lines"] += 1
                extra = set(_mentions(line["text"], match.prematch["players"])) \
                    - set(_mentions("\n".join(line["facts"] + [line.get("pictures") or ""]), match.prematch["players"]))
                text["added_names"] += bool(extra)
                # The whole knowledge base goes with every stretch; a line marks the fact it tells.
                if line.get("fact"):
                    text["facts_told"] += 1
                    told.add((name, line["fact"]))
                if line["entry"] in true_of and line["kind"] == "news":
                    g = truth[true_of[line["entry"]]]
                    lags.append(line["t_ready"][1] - g["time"][1] if line["t_ready"][0] == g["time"][0] else np.nan)
            # Timing: shots and goals called as they happen, how long after its frame a phrase
            # starts, the speaking rate and the energy.
            tm = timing[run]
            shots = [g["time"] for g in truth if g["visible"] and g["type"] in ("shot", "goal")
                     and any(h == g["time"][0] and a <= g["time"][1] < b for h, a, b in ranges)]
            said = [(l["show"][0], l["show"][1], l["text"]) for l in lines]
            tm["shots"] += len(shots)
            tm["called"] += sum(called_in_time(said, g) for g in shots)
            tm["rate"] += [len(l["text"].split()) / l["duration"] for l in lines if l.get("audio") and l["duration"]]
            tm["after_frame"] += [l["show"][1] - l["at"][1] for l in lines if l.get("at")]
            tm["tone"].update(l["tone"] for l in lines)
            if run == args.runs[0]:
                for h, a, b in ranges:
                    ref = out / "reference" / name / f"half{h}.json"
                    if ref.exists():
                        segs = [x for x in json.loads(ref.read_text()) if a <= x["start"] < b]
                        human_texts += [x["text"] for x in segs]
                        human_speech += sum(x["end"] - x["start"] for x in segs)
                        human_timing["called"] += sum(called_in_time([(h, x["start"], x["text"]) for x in segs], g)
                                                      for g in shots if g[0] == h)
                        human_timing["shots"] += sum(g[0] == h for g in shots)
                        human_timing["rate"] += [len(x["text"].split()) / (x["end"] - x["start"])
                                                 for x in segs if x["end"] > x["start"]]
                total_seconds += match_seconds
        text["facts_different"] = len(told)
        rows[run] = (ours, blind, text, lags)
        if seconds:
            styles[run] = style(texts, speech, seconds)
    if human_texts:
        styles["human (broadcast)"] = style(human_texts, human_speech, total_seconds)

    def pct(a, b):
        return f"{100 * a / b:5.1f}%" if b else "    -"

    for run, (ours, blind, text, lags) in rows.items():
        print(f"\n== {run}: {text['lines']} lines; lines naming a player the material did not give: "
              f"{text['added_names']} ({pct(text['added_names'], text['lines']).strip()}); pre-match facts told: "
              f"{text['facts_told']} ({text['facts_different']} different)")
        print(f"{'':28s} {'claims as fact':>15s} {'false':>7s} {'goals/cards/subs as fact':>25s} {'false':>7s} "
              f"{'hedged':>7s} {'false':>7s} {'names checked':>14s} {'wrong':>7s}")
        for label, c in (("status-aware (ours)", ours), ("status-blind (v2-like)", blind)):
            print(f"{label:28s} {c['fact']:15d} {pct(c['fact_false'], c['fact']):>7s} {c['fact_important']:25d} "
                  f"{pct(c['fact_important_false'], c['fact_important']):>7s} {c['hedged']:7d} "
                  f"{pct(c['hedged_false'], c['hedged']):>7s} {c['named_checked']:14d} {pct(c['named_wrong'], c['named_checked']):>7s}")
        aired = ours["fact"] + ours["hedged"]
        print(f"false claims on air: {ours['fact_false'] + ours['hedged_false']} of {aired} "
              f"({pct(ours['fact_false'] + ours['hedged_false'], aired).strip()}); visible true events spoken of: "
              f"goals/cards/subs {text['spoken_important']}/{text['true_important']}, "
              f"others {text['spoken_minor']}/{text['true_minor']} ({pct(text['spoken_minor'], text['true_minor']).strip()})")
        print(f"corrections on air: {ours['corrected_retraction']} retractions ({ours['corrected_retraction_right']} of "
              f"a false entry), {ours['corrected_player']} player changes ({ours['corrected_player_right']} to the true player)")
        lags = [x for x in lags if not np.isnan(x)]
        if lags:
            print(f"true event -> news line ready (live): p50 {np.median(lags):.0f} s, p90 {np.percentile(lags, 90):.0f} s")

    print(f"\n{'timing':28s} {'shots called in time':>21s} {'start after its frame p50/p90':>30s} "
          f"{'words/s p50/p90':>16s}  energy")
    for run, tm in list(timing.items()) + [("human (broadcast)", human_timing)]:
        after = (f"{np.median(tm['after_frame']):.1f} / {np.percentile(tm['after_frame'], 90):.1f} s"
                 if tm.get("after_frame") else "-")
        rate = f"{np.median(tm['rate']):.2f} / {np.percentile(tm['rate'], 90):.2f}" if tm["rate"] else "-"
        tones = tm.get("tone") or Counter()
        energy = ", ".join(f"{k} {v / sum(tones.values()):.0%}" for k, v in tones.most_common())
        print(f"{run:28s} {tm['called']:>10d} / {tm['shots']:<3d} {pct(tm['called'], tm['shots']):>6s} {after:>30s} "
              f"{rate:>16s}  {energy}")

    print(f"\n{'style':28s} {'speech':>7s} {'words/min':>10s} {'lines/min':>10s} {'words/line':>11s} {'distinct-2':>11s} "
          f"{'same opening':>13s}")
    for run, s in styles.items():
        print(f"{run:28s} {s['speech']:7.1%} {s['wpm']:10.1f} {s['lpm']:10.2f} {s['words_per_line']:11.1f} "
              f"{s['distinct2']:11.2f} {s['same_opening']:13.1%}")


if __name__ == "__main__":
    main()
