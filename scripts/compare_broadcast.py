"""Our commentary next to the broadcast's own (human) commentary, on the same minutes: which match
events each says happen, and which players each names in the play, both scored against the labels.

    python scripts/compare_broadcast.py --match M --runs fn_ctx_low gemini_ctx_h2 [--only 2 0 900]

Both sides are read the same way. The human side is the speech-recognition transcript
(scripts/transcribe_broadcast.py --names), ours the lines of a run (outputs/commentary/<match>/<run>/
lines.jsonl, at the time each went on air). A model reads each minute of a side's text — with the
squads, so a misheard name is mapped to the player meant — and lists the events it says happen in
the live play (not replays or talk about earlier moments) and the players it names in the play.
Every reading is a cached model call.

Scored per side:
- Events: a claim is right when a label of its type (a shot claim may also be a goal), of its team
  if it gives one, lies within [-20, +5] s of when it was said; coverage is the share of the visible
  labels some claim is right about. The labels miss some events (blocked shots, for one), which
  costs both sides alike.
- Players named in an event the labels name someone for (goals, cards, substitutions): right or not.
- Players named in the play: how many a minute, how many were not on the pitch then (bench, or
  already substituted), and how many the other side also names within 8 s — the broadcast's
  commentators, who see the pictures, are the reference for who is on the ball; and, where the
  broadcast names someone within 5 s of one of ours, whether it is the same player.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from eval_memory import MATCHES                          # scripts/eval_memory.py
from ground_truth import ground_truth                    # scripts/ground_truth.py
from matchlive.common.api import APIClient
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.events.analyzer import clock, parse_json

TYPES = ("shot", "goal", "foul", "free_kick", "corner", "offside", "yellow_card", "red_card", "substitution", "penalty")

PROMPT = """Below is {who} of the football match {home} (home) v {away} (away), half {half}, each line with the
video time it was said ("mm:ss"). {asr}

Squads:
{squads}

1. List every match event the commentary says is happening in the live play at that moment: {types}. A shot
   includes an attempt that is saved, blocked, missed or headed at goal. Leave out what is only a replay or talk
   about an earlier moment, and what might happen.
2. List every player the commentary names as taking part in the play at that moment (on the ball, passing,
   shooting, tackling, saving, coming on). Leave out players only talked about (analysis, background, stats).

Name players as in the squads; map a misspelt or misheard name to the squad player meant, and leave it out if
unclear. Time each by the line it is said in.

Commentary:
{text}

Answer with JSON only: {{"events": [{{"time": "mm:ss", "type": "...", "team": "home" | "away" | null,
"player": "<squad name>" | null}}, ...], "players": [{{"time": "mm:ss", "name": "<squad name>"}}, ...]}}"""


def seconds(mmss) -> float | None:
    try:
        m, s = str(mmss).split(":")
        return int(m) * 60 + float(s)
    except ValueError:
        return None


def read(client, match, half, lines, who, asr, minute=60.0):
    """The events and players a side's lines (time, text) claim, minute by minute."""
    teams = match.prematch["teams"]
    squads = "\n".join(f"- {teams[side]} ({side}): " + ", ".join(
        f"{p['long_name']} (#{p['jersey_number']}{', sub' if not p['starting'] else ''})"
        for p in match.prematch["players"] if p["team_side"] == side) for side in ("home", "away"))
    hashes = {n: p["hash"] for p in match.prematch["players"] for n in (p["long_name"], p["short_name"])}
    events, players = [], []
    first, last = min((t for t, _ in lines), default=0.0), max((t for t, _ in lines), default=-1.0)
    chunks = [c for k in range(int((last - first) // minute) + 1)
              if (c := [(t, x) for t, x in lines if first + k * minute <= t < first + (k + 1) * minute])]

    def ask(chunk):
        return client.generate([PROMPT.format(who=who, home=teams["home"], away=teams["away"], half=half, asr=asr,
                                              squads=squads, types=", ".join(TYPES),
                                              text="\n".join(f"[{clock(t)}] {x}" for t, x in chunk))])

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(ask, chunks))
    for r in results:
        answer = parse_json(r.text)
        for e in answer.get("events") or []:
            t = seconds(e.get("time"))
            if t is not None and e.get("type") in TYPES:
                events.append({"t": t, "type": e["type"], "team": e.get("team") if e.get("team") in ("home", "away") else None,
                               "player": hashes.get(e.get("player"))})
        for p in answer.get("players") or []:
            t = seconds(p.get("time"))
            if t is not None and p.get("name") in hashes:
                players.append({"t": t, "player": hashes[p["name"]]})
    return events, players


def fits(claim, label):
    kinds = {"shot", "goal"} if claim["type"] == "shot" else {claim["type"]}
    return (label["type"] in kinds and (claim["team"] is None or claim["team"] == label["team"])
            and -20 <= label["time"][1] - claim["t"] <= 5)


def on_pitch(match, truth, half):
    """player hash -> the (start, end) seconds of this half he is on the pitch."""
    on = {p["hash"]: [0.0, float("inf")] for p in match.prematch["players"] if p["starting"]}
    for g in truth:
        if g["type"] == "substitution" and g["players"]:
            if g["time"][0] < half:
                on.pop(g["players"]["out"], None)
                on[g["players"]["in"]] = [0.0, float("inf")]
            elif g["time"][0] == half:
                if g["players"]["out"] in on:
                    on[g["players"]["out"]][1] = g["time"][1]
                on[g["players"]["in"]] = [g["time"][1], float("inf")]
    return on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default=MATCHES[1])
    ap.add_argument("--runs", nargs="+", required=True, help="commentary folders under outputs/commentary/<match>/")
    ap.add_argument("--only", nargs=3, type=float, metavar=("HALF", "START", "END"), default=(2, 0, 900))
    ap.add_argument("--model", default=None, help="the reading model (default: api.model)")
    args = ap.parse_args()
    config = load_config()
    out = config["paths"]["outputs"]
    client = APIClient(dict(config["api"], **({"model": args.model} if args.model else {})), config["paths"]["cache"])
    match = Match(config, args.match)
    half, a, b = int(args.only[0]), args.only[1], args.only[2]
    truth = ground_truth(match)
    labels = [g for g in truth if g["time"][0] == half and a <= g["time"][1] < b and g["visible"]]
    on = on_pitch(match, truth, half)
    short = {p["hash"]: p["short_name"] for p in match.prematch["players"]}

    sides = {"broadcast": [(s["start"], s["text"]) for s in json.loads(
        (out / "reference" / args.match / f"half{half}.names.json").read_text()) if a <= s["start"] < b]}
    for run in args.runs:
        lines = [json.loads(x) for x in open(out / "commentary" / args.match / run / "lines.jsonl")]
        sides[run] = [(l["show"][1], l["text"]) for l in lines if l["show"][0] == half and a <= l["show"][1] < b]

    read_out = {}
    for side, lines in sides.items():
        human = side == "broadcast"
        read_out[side] = read(client, match, half, lines,
                              "the TV broadcast's live commentary" if human else "a TV commentator's live commentary",
                              "It is a speech-recognition transcript: names may be misheard." if human else "")
    dump = out / "compare_broadcast" / args.match
    dump.mkdir(parents=True, exist_ok=True)
    (dump / f"h{half}_{int(a)}-{int(b)}.json").write_text(json.dumps(read_out, indent=1))

    minutes = (b - a) / 60
    print(f"{args.match}, half {half} {clock(a)}-{clock(b)}: {len(labels)} visible labels "
          f"({', '.join(f'{k} {n}' for k, n in Counter(g['type'] for g in labels).items())})\n")
    for side, (events, players) in read_out.items():
        right = [e for e in events if any(fits(e, g) for g in labels)]
        covered = [g for g in labels if any(fits(e, g) for e in events)]
        by_type = Counter(e["type"] for e in events)
        right_type = Counter(e["type"] for e in right)
        # Players of the labelled events that name one, where the claim names one too.
        named = [(e, g) for e in events for g in labels if fits(e, g) and g["players"] and e["player"]]
        named_ok = sum(e["player"] in g["players"].values() for e, g in named)
        off = [p for p in players if not (p["player"] in on and on[p["player"]][0] <= p["t"] <= on[p["player"]][1])]
        others = [x for x in read_out if x != side and (x == "broadcast" or side == "broadcast")]
        agree = {o: sum(any(q["player"] == p["player"] and abs(q["t"] - p["t"]) <= 8 for q in read_out[o][1])
                        for p in players) for o in others}
        print(f"== {side}")
        print(f"   events said {len(events)}, right {len(right)} ({len(right) / max(len(events), 1):.0%}); "
              f"labels covered {len(covered)}/{len(labels)}")
        print("   by type (right/said): " + ", ".join(f"{t} {right_type[t]}/{n}" for t, n in by_type.most_common()))
        print(f"   labelled players named: {named_ok}/{len(named)}")
        print(f"   players named in the play: {len(players)} ({len(players) / minutes:.1f}/min), "
              f"not on the pitch {len(off)}" + (f" ({', '.join(sorted({short[p['player']] for p in off}))})" if off else ""))
        for o, n in agree.items():
            print(f"   of them also named by {o} within 8 s: {n}/{len(players)} ({n / max(len(players), 1):.0%})")
        if side != "broadcast":
            # Where the broadcast names someone within 5 s, is one of them the same player?
            near = [p for p in players if any(abs(q["t"] - p["t"]) <= 5 for q in read_out["broadcast"][1])]
            same = sum(any(q["player"] == p["player"] and abs(q["t"] - p["t"]) <= 5 for q in read_out["broadcast"][1])
                       for p in near)
            print(f"   where the broadcast names someone within 5 s, the same player: {same}/{len(near)} "
                  f"({same / max(len(near), 1):.0%})")
        missed = [g for g in labels if g not in covered]
        if missed:
            print("   labels not covered: " + ", ".join(f"{g['type']}({g['team']})@{clock(g['time'][1])}" for g in missed))
        print()


if __name__ == "__main__":
    main()
