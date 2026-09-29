"""Error-injection experiment (DESIGN.md §10 phase 4): corrupt what the event analyzer says about
one true event, play the match live (runtime.runner.run_live) until `--horizon` s after it, and see
whether and how soon memory gets it right again.

    python scripts/inject_errors.py [--match M ...] [--kind drop_substitution wrong_card] [--horizon 900]
                                    [--workers 6] [--set loop.x=false ...] [--name inject]

One trial per true event (scripts/ground_truth.py), two kinds of error:
  drop_substitution  the analyzer never reports the substitution: every substitution candidate of
                     that team seen within ±90 s of it is removed (substitutions of one team less
                     than 90 s apart form one trial, since their sightings cannot be told apart).
                     Right again when memory has every player of the trial who came on on the
                     pitch and every player who went off off it; the way back is the off-pitch
                     player's number read on the pitch (loop.state_to_perception) and the arbiter.
  wrong_card         the analyzer's sightings of the yellow card name the wrong player: in every
                     yellow-card candidate of that team within ±60 s, the booked track's voted
                     player becomes a teammate on the pitch (the nearest shirt number). Right when
                     a card memory counts near that time names the true player and none names
                     the wrong one; "cleared" when at least no counted card names the wrong one.
Everything before the first corrupted window replays from the API cache (the uncorrupted run is
`full`), so a trial costs only the model calls after it. Each trial writes
outputs/live/<match>/<name>/<trial>/ like run_live; the summary prints, per trial, whether memory
recovered by the horizon and how long after the true event, next to the same measure on `full`.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ground_truth import ground_truth, lineups   # scripts/ground_truth.py
from matchlive.common.api import APIClient
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.memory.store import MatchMemory
from matchlive.runtime.runner import run_live

MATCHES = ("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool")
STEP = 10.0             # s between the knowledge times recovery is checked at


def trials(match: Match, kinds) -> list[dict]:
    truth = ground_truth(match)
    out = []
    if "drop_substitution" in kinds:
        subs = [g for g in truth if g["type"] == "substitution" and g["players"]]
        groups: list[list[dict]] = []
        for g in subs:
            last = groups[-1][-1] if groups else None
            if last and last["team"] == g["team"] and last["time"][0] == g["time"][0] and g["time"][1] - last["time"][1] < 90:
                groups[-1].append(g)
            else:
                groups.append([g])
        for group in groups:
            h, t = group[0]["time"]
            out.append({"id": f"drop_sub_{h}H{int(t)}", "kind": "drop_substitution", "team": group[0]["team"],
                        "time": (h, t), "end": group[-1]["time"][1],
                        "in": [g["players"]["in"] for g in group], "out": [g["players"]["out"] for g in group]})
    if "wrong_card" in kinds:
        players = match.prematch["players"]
        for g in truth:
            if g["type"] != "yellow_card" or not g["players"]:
                continue
            true = g["players"]["player"]
            number = int(next(p["jersey_number"] for p in players if p["hash"] == true))
            on = lineups(match, truth, g["time"])[g["team"]]
            mates = [p for p in players if p["hash"] in on and p["hash"] != true and p["position"] != "goalkeeper"]
            wrong = min(mates, key=lambda p: (abs(int(p["jersey_number"]) - number), p["hash"]))["hash"]
            h, t = g["time"]
            out.append({"id": f"wrong_card_{h}H{int(t)}", "kind": "wrong_card", "team": g["team"], "time": (h, t),
                        "end": t, "true": true, "wrong": wrong})
    return out


def corrupter(trial: dict, log: list):
    """The candidate filter run_live applies to every analyzer answer."""
    h, lo, hi = trial["time"][0], trial["time"][1], trial["end"]

    def corrupt(half, window, cands):
        out = []
        for c in cands:
            near = half == h and c["team"] in (trial["team"], None)
            if trial["kind"] == "drop_substitution" and near and c["type"] == "substitution" and lo - 90 <= c["second"] <= hi + 90:
                log.append({"window": window, "dropped": c})
                continue
            if trial["kind"] == "wrong_card" and near and c["type"] == "yellow_card" and lo - 60 <= c["second"] <= hi + 60:
                roles = [role for role, _ in c["participants"]]
                if "player" in roles:
                    players = list(c["players"])
                    players[roles.index("player")] = trial["wrong"]
                    log.append({"window": window, "renamed": c, "was": c["players"]})
                    c = {**c, "players": players}
            out.append(c)
        return out
    return corrupt


def run_trial(args):
    match_name, trial, horizon, overrides, name = args
    config = load_config(overrides=overrides)
    match = Match(config, match_name)
    out_dir = config["paths"]["outputs"] / "live" / match_name / name / trial["id"]
    log: list = []
    until = _later(_half_ends(config, match_name), trial["time"], horizon)
    run_live(config, match, out_dir, APIClient(config["api"], config["paths"]["cache"]),
             corrupt=corrupter(trial, log), until=until)
    (out_dir / "trial.json").write_text(json.dumps({"trial": trial, "until": until, "set": overrides,
                                                    "corrupted": log}, indent=1, default=str))
    return out_dir


def recovery(mem: MatchMemory, trial: dict, half_ends: dict, horizon: float) -> dict:
    """The first knowledge time (s after the true event) from which memory stays right up to the
    horizon, checked every STEP s; None if it is not right at the horizon. Also, for a card, when
    no counted card names the wrong player any more ("cleared")."""
    times = []
    t = trial["time"]
    while True:
        times.append(t)
        nxt = _later(half_ends, t, STEP)
        if _gap(half_ends, trial["time"], nxt) > horizon or nxt == t:
            break
        t = nxt
    right, cleared = [], []
    for t in times:
        view = mem.at(t)
        if trial["kind"] == "drop_substitution":
            on = set(view.on_field["home"] + view.on_field["away"])
            right.append(all(p in on for p in trial["in"]) and not any(p in on for p in trial["out"]))
            cleared.append(right[-1])
        else:
            near = [c for c in view.cards if c.team == trial["team"] and c.time[0] == trial["time"][0]
                    and abs(c.time[1] - trial["time"][1]) <= 120 and not c.derived]
            wrong = any(c.player_id == trial["wrong"] for c in near)
            right.append(any(c.player_id == trial["true"] for c in near) and not wrong)
            cleared.append(not wrong)

    def since(flags):
        if not flags or not flags[-1]:
            return None
        i = len(flags) - 1
        while i > 0 and flags[i - 1]:
            i -= 1
        return _gap(half_ends, trial["time"], times[i])
    return {"right_after": since(right), "cleared_after": since(cleared)}


def _half_ends(config, match_name) -> dict[int, float]:
    windows = [json.loads(line) for line in open(config["paths"]["outputs"] / "live" / match_name / "full" / "windows.jsonl")]
    return {h: max(w["window"][1] for w in windows if w["half"] == h) for h in (1, 2)}


def _later(half_ends, t, dt):
    """Match time `dt` s after `t`, running on into the second half; capped at the final whistle."""
    h, s = t
    s += dt
    if h == 1 and s > half_ends[1]:
        h, s = 2, s - half_ends[1]
    return (h, min(s, half_ends[2])) if h == 2 else (h, s)


def _gap(half_ends, a, b) -> float:
    return b[1] - a[1] if a[0] == b[0] else half_ends[1] - a[1] + b[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    ap.add_argument("--kind", nargs="+", default=["drop_substitution", "wrong_card"])
    ap.add_argument("--horizon", type=float, default=900.0, help="s of match time after the true event")
    ap.add_argument("--workers", type=int, default=6, help="trials played at once")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="config override for the trials (e.g. loop.arbiter=false)")
    ap.add_argument("--name", default="inject", help="folder of the trials under outputs/live/<match>/")
    ap.add_argument("--summary", action="store_true", help="only summarise the trials already played")
    args = ap.parse_args()

    base = load_config(overrides=args.set)
    jobs = [(name, trial, args.horizon, args.set, args.name)
            for name in args.match or MATCHES for trial in trials(Match(base, name), args.kind)]
    if not args.summary:
        # A trial already played (its trial.json written) is not played again.
        todo = [j for j in jobs if not (base["paths"]["outputs"] / "live" / j[0] / args.name / j[1]["id"] / "trial.json").exists()]
        with ProcessPoolExecutor(args.workers) as pool:
            for out_dir in pool.map(run_trial, todo):
                print("done", out_dir, flush=True)

    print(f"\n{'match':10s} {'trial':22s} {'corrupted':>9s} {'right after (s)':>16s} {'full: right after':>18s} "
          f"{'cleared after':>14s} {'full: cleared':>14s}")
    for name, trial, horizon, overrides, run_name in jobs:
        config = load_config(overrides=overrides)
        match = Match(config, name)
        half_ends = _half_ends(config, name)
        out_dir = config["paths"]["outputs"] / "live" / name / run_name / trial["id"]
        if not (out_dir / "trial.json").exists():
            continue
        corrupted = len(json.loads((out_dir / "trial.json").read_text())["corrupted"])
        mem = MatchMemory(match.prematch["players"], log_path=out_dir / "ops.jsonl", cascade=config["loop"]["cascade"])
        ref = MatchMemory(match.prematch["players"], log_path=config["paths"]["outputs"] / "live" / name / "full" / "ops.jsonl")
        got, base_got = recovery(mem, trial, half_ends, horizon), recovery(ref, trial, half_ends, horizon)
        fmt = lambda x: "-" if x is None else f"{x:.0f}"
        print(f"{name.split()[-1]:10s} {trial['id']:22s} {corrupted:9d} {fmt(got['right_after']):>16s} "
              f"{fmt(base_got['right_after']):>18s} {fmt(got['cleared_after']):>14s} {fmt(base_got['cleared_after']):>14s}")


if __name__ == "__main__":
    main()
