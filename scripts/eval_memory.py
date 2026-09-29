"""Score live runs' memory against the ground truth over knowledge time (DESIGN.md §10).

    python scripts/eval_memory.py [--runs full no_arbiter ...] [--match M ...]

Reads each run's op log (scripts/run_live.py, outputs/live/<match>/<run>/) and the match's true
events (scripts/ground_truth.py: SoccerNet Labels-v2 + captions). Memory is replayed op by op, so
every entry's status is known at every knowledge time; nothing is re-run. Per run, summed over
the matches (both benchmark matches by default):

  1. events at the final whistle — precision of the counted (confirmed) entries, recall of the
     visible true events among counted entries and among held ones (confirmed or pending);
  2. accuracy against knowledge time (§10): the share of true events memory counts / holds
     Δ s after they happened;
  3. revisions — status changes after an entry was written: false entries (matching no true
     event) that memory once counted and no longer counts at the end (corrections that reached
     the inferences), with the correction latency; true entries once counted and no longer
     (wrong corrections; §10's wrong-correction rate is their share of both); entries retracted
     after being written;
  4. derived state (what the rule inferences make of memory): the score, the lineups and the
     per-team counts memory gives for time t as known at t + L, against the truth at t;
  5. identity — the players memory names on goals, cards and substitutions; and, per sampled
     frame, how many tracks carry an identity (coverage) and how many of those name a player
     who is not on the pitch at that moment (a certain error: the open- vs closed-loop check of
     phase 4, which has no per-track ground truth on these matches);
  6. what made entries sufficient — the second justification of each entry that has one: a
     sighting from the overlap of two adjacent windows (the same frames seen twice), from
     another window, from a replay, the foul / free-kick pairing, or a verdict — and how many of
     the entries each one confirmed are true.

An entry matches a true event of its type (a shot entry may also match a goal), half and team
(an entry without a team fits either) whose time lies within the entry's interval ± TOL s.
Precision uses a one-to-one assignment (an entry standing at the end is preferred, then the
nearest); the knowledge-time curve asks only whether any matching entry is held at that moment.
Injuries are not scored (no such class in Labels-v2). Writes the raw counts per (match, run) to
outputs/eval_memory/<match>/<run>.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ground_truth import ground_truth, lineups, score   # scripts/ground_truth.py
from matchlive.common.config import load_config, run_overrides
from matchlive.common.match import Match
from matchlive.memory.store import MatchMemory
from matchlive.memory.types import Event, Op

MATCHES = ("2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United", "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool")
EVAL_TYPES = ("goal", "shot", "yellow_card", "red_card", "substitution", "corner", "foul", "offside",
              "free_kick", "penalty")
TOL = 5.0                           # s beyond an entry's interval
DELAYS = (30, 60, 120, 300, 600)    # s after a true event, for the knowledge-time curve
LAGS = (60, 120, 300)               # s of hindsight for the derived state
COUNTED_TYPES = ("shot", "corner", "free_kick", "foul")
END = (3, 0.0)                      # after the final whistle
RANK = {"confirmed": 0, "pending": 1, "retracted": 2}


def evaluate(config: dict[str, Any], match: Match, run_dir: Path) -> dict[str, Any]:
    truth = ground_truth(match)
    ops = [Op(d["op_id"], tuple(d["t_known"]), d["kind"], d["payload"]) for d in map(json.loads, open(run_dir / "ops.jsonl"))]
    roster, cascade = match.prematch["players"], config["loop"]["cascade"]
    windows = [json.loads(line) for line in open(run_dir / "windows.jsonl")]
    sources = {f"obs:{w['call_id']}#{i}": (w["window"], c) for w in windows for i, c in enumerate(w.get("candidates", []))}
    replay_obs = {obs for obs, (_, c) in sources.items() if c["replay"]}
    half_end = {h: max(w["window"][1] for w in windows if w["half"] == h) for h in (1, 2)}

    # Replay the log op by op: every entry's status after every op, and what changed it.
    mem = MatchMemory(roster, cascade=cascade)
    timeline: dict[str, list[tuple]] = {}           # entry id -> [(t_known, status, op)]
    for op in ops:
        mem._apply(op)                              # the same path a log replay takes
        for eid in mem.entries:
            s = mem.status(eid)
            if eid not in timeline or timeline[eid][-1][1] != s:
                timeline.setdefault(eid, []).append((op.t_known, s, op))
    events = {eid: e for eid, e in mem.entries.items() if isinstance(e, Event) and e.type in EVAL_TYPES}
    visible = [i for i, g in enumerate(truth) if g["visible"]]

    def status_at(eid, t):
        s = None
        for when, status, _ in timeline[eid]:
            if when > t:
                break
            s = status
        return s

    true_of = assign(mem, events, truth)

    def counted_by(eid, t):
        # What the rule inferences count (MatchMemory.counted_events): the confirmed entries —
        # or, with loop.cascade off, every entry ever confirmed (once counted, always counted).
        seen = [s for when, s, _ in timeline[eid] if when <= t]
        return ("confirmed" in seen) if not cascade else (seen[-1:] == ["confirmed"])

    out: dict[str, Any] = {}
    # 1. At the final whistle.
    final = {eid: mem.status(eid) for eid in events}
    counted = {eid for eid in events if counted_by(eid, END)}
    out["final"] = {
        "counted": len(counted),
        "counted_true": sum(eid in true_of for eid in counted),
        "held": sum(s != "retracted" for s in final.values()),
        "held_true": sum(s != "retracted" and eid in true_of for eid, s in final.items()),
        "visible": len(visible),
        "visible_counted": sum(any(true_of.get(eid) == i for eid in counted) for i in visible),
        "visible_held": sum(any(true_of.get(eid) == i and final[eid] != "retracted" for eid in events) for i in visible),
        "by_type": {ty: {"visible": sum(truth[i]["type"] == ty for i in visible),
                         "counted": sum(events[eid].type == ty for eid in counted),
                         "counted_true": sum(events[eid].type == ty and eid in true_of for eid in counted),
                         "visible_counted": sum(truth[i]["type"] == ty and any(true_of.get(eid) == i for eid in counted)
                                                for i in visible)}
                    for ty in EVAL_TYPES},
    }

    # 2. Accuracy against knowledge time: among the entries matching each true event, Δ s after it
    #    (known by then; an entry written later does not count yet), is one counted, or held pending?
    curve = {}
    for delay in (*DELAYS, "end"):
        c = Counter()
        for i in visible:
            g = truth[i]
            k = END if delay == "end" else (g["time"][0], g["time"][1] + delay)
            near = [eid for eid, e in events.items() if _compatible(e, g) and status_at(eid, k)]
            c["counted" if any(counted_by(eid, k) for eid in near)
              else "pending" if any(status_at(eid, k) == "pending" for eid in near) else "missing"] += 1
        curve[str(delay)] = dict(c)
    out["curve"] = curve

    # 3. Revisions after an entry was written: status changes (the beliefs), and entries once
    #    counted that the inferences no longer count at the end (the corrections that reached them).
    rev = Counter()
    latencies = []
    for eid in events:
        tl, right = timeline[eid], eid in true_of
        truth_tag = "true" if right else "false"
        rev[truth_tag] += 1
        for (_, a, _), (t, b, op) in zip(tl, tl[1:]):
            cause = op.kind + ("(replay)" if op.kind == "add_justification"
                               and op.payload["justification"]["obs_id"] in replay_obs else "")
            rev[f"{a}->{b}|{truth_tag}"] += 1
            rev[f"cause:{a}->{b}|{cause}"] += 1
        if final[eid] == "retracted" and len(tl) > 1:
            rev[f"{truth_tag}_retracted_later"] += 1
        first = next((t for t, s, _ in tl if s == "confirmed"), None)
        if first is None:
            continue
        rev[f"{truth_tag}_ever_counted"] += 1
        if eid not in counted:
            rev[f"{truth_tag}_uncounted"] += 1
            if not right:
                # From first counted to the last time it stopped being confirmed.
                dropped = [t for (_, a, _), (t, b, _) in zip(tl, tl[1:]) if a == "confirmed"][-1]
                latencies.append(_seconds_between(first, dropped, half_end))
    out["revisions"] = dict(rev)
    out["correction_latency"] = latencies

    # 4. Derived state for time t as known at t + L, every 30 s.
    derived = {}
    for lag in LAGS:
        c = Counter()
        err = defaultdict(list)
        for h in (1, 2):
            for t in np.arange(30, half_end[h], 30.0):
                view = mem_at(ops, roster, cascade, (h, float(t)), (h, float(t) + lag))
                true_on = lineups(match, truth, (h, float(t)))
                c["samples"] += 1
                c["score_right"] += view.score == score(truth, (h, float(t)))
                right_slots = sum(len(set(view.on_field[s]) & true_on[s]) for s in ("home", "away"))
                c["lineup_slots"] += 22
                c["lineup_slots_right"] += right_slots
                c["lineup_exact"] += right_slots == 22 and all(len(view.on_field[s]) == 11 for s in ("home", "away"))
                for ty in COUNTED_TYPES:
                    for side in ("home", "away"):
                        n_true = sum(g["type"] == ty and g["team"] == side and g["time"] <= (h, float(t)) for g in truth)
                        err[ty].append(view.team_stats[side][ty] - n_true)
        # Sums, so matches add up: mean |error| = abs / n, mean signed error = signed / n.
        derived[str(lag)] = dict(c) | {"count_error": {ty: {"abs": float(np.abs(v).sum()), "signed": float(np.sum(v)),
                                                            "n": len(v)} for ty, v in err.items()}}
    out["derived"] = derived

    # 5a. Players named on the true goals, cards and substitutions, at the end.
    who = Counter()
    track_players = mem.track_players()
    for eid, i in true_of.items():
        g = truth[i]
        if not g["players"] or final[eid] == "retracted" or events[eid].type != g["type"]:
            continue
        said = dict(zip((p.role for p in events[eid].participants), mem.event_players(eid, track_players)))
        for role, pid in g["players"].items():
            who["right" if said.get(role) == pid else "unnamed" if not said.get(role) else "wrong"] += 1
    out["event_players"] = dict(who)

    # 5b. Per sampled frame: coverage, and identities naming a player who is not on the pitch.
    frames = Counter()
    for h in (1, 2):
        path = run_dir / f"perceived_half{h}.pkl"
        if not path.exists():
            continue
        perceived = pickle.load(open(path, "rb"))
        for s in sorted(perceived):
            on = lineups(match, truth, (h, s))
            on = on["home"] | on["away"]
            for tr in perceived[s][0]:
                frames["tracks"] += 1
                if tr.player_id:
                    frames["identified"] += 1
                    frames["off_pitch"] += tr.player_id not in on
                    frames["entry"] += tr.identity_source == "entry"
    out["frames"] = dict(frames)

    # 6. The second justification that made each entry sufficient, and whether the entry is true.
    confirmers = Counter()
    for eid, e in events.items():
        js = [j for _, j in mem.justifications[eid]]
        second = next((j for j in js[1:] if j.sufficient), None)
        if second is None:
            continue
        first = sources.get(js[0].obs_id)
        this = sources.get(second.obs_id)
        if second.kind == "verdict":
            how = "verdict"
        elif first is None or this is None:
            how = "other"
        elif this[1]["type"] != e.type:
            how = "pairing"
        elif this[1]["replay"] or first[1]["replay"]:
            how = "replay"
        else:
            # Both sightings inside the stretch two adjacent windows share: the same frames.
            (a0, a1), (b0, b1) = first[0], this[0]
            lo, hi = max(a0, b0), min(a1, b1)
            how = "overlap" if lo < hi and lo <= first[1]["second"] <= hi and lo <= this[1]["second"] <= hi else "other window"
        confirmers[f"{how}|{'true' if eid in true_of else 'false'}"] += 1
    out["confirmers"] = dict(confirmers)
    return out


_views: dict = {}


def mem_at(ops: list[Op], roster, cascade: bool, t, known_by):
    """The view of time `t` as known by `known_by`, from a memory holding the first n ops."""
    n = sum(1 for op in ops if op.t_known <= known_by)
    key = (id(ops), n)
    if key not in _views:
        m = MatchMemory(roster, cascade=cascade)
        for op in ops[:n]:
            m._apply(op)
        _views.clear()                              # one prefix at a time; samples come in time order
        _views[key] = m
    return _views[key].at(t, known_by=known_by)


def assign(mem: MatchMemory, events: dict[str, Event], truth: list[dict]) -> dict[str, int]:
    """Entry id -> index of the true event it is (one-to-one): an entry standing at the end is
    preferred, then an exact type, then the nearest. An entry left out matches no true event."""
    pairs = sorted((RANK[mem.status(eid)], e.type != g["type"], abs(g["time"][1] - _center(e)), eid, i)
                   for eid, e in events.items() for i, g in enumerate(truth) if _compatible(e, g))
    true_of: dict[str, int] = {}
    taken: set[int] = set()
    for *_, eid, i in pairs:
        if eid not in true_of and i not in taken:
            true_of[eid] = i
            taken.add(i)
    return true_of


def _compatible(e: Event, g: dict) -> bool:
    (h, lo), (_, hi) = e.t_interval
    return ((g["type"] == e.type or (e.type == "shot" and g["type"] == "goal")) and g["time"][0] == h
            and (e.team is None or g["team"] is None or e.team == g["team"]) and lo - TOL <= g["time"][1] <= hi + TOL)


def _center(e: Event) -> float:
    return (e.t_interval[0][1] + e.t_interval[1][1]) / 2


def _seconds_between(a, b, half_end) -> float:
    """Match seconds from time a to time b (across half-time: the rest of the first half counts)."""
    if a[0] == b[0]:
        return b[1] - a[1]
    return half_end.get(a[0], a[1]) - a[1] + (b[1] if b[0] <= 2 else 0.0)


def _add(total: dict, part: dict) -> None:
    for k, v in part.items():
        if isinstance(v, dict):
            _add(total.setdefault(k, {}), v)
        elif isinstance(v, list):
            total.setdefault(k, []).extend(v)
        else:
            total[k] = total.get(k, 0) + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["full"])
    ap.add_argument("--match", action="append", default=None, help="default: both benchmark matches")
    args = ap.parse_args()
    base = load_config()

    rows = {}
    for run in args.runs:
        total: dict = {}
        for name in args.match or MATCHES:
            run_dir = base["paths"]["outputs"] / "live" / name / run
            if not (run_dir / "ops.jsonl").exists():
                print(f"(no run {run} for {name})")
                continue
            config = load_config(overrides=run_overrides(run_dir))
            result = evaluate(config, Match(config, name), run_dir)
            out = base["paths"]["outputs"] / "eval_memory" / name / f"{run}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1))
            _add(total, result)
        rows[run] = total

    def pct(a, b):
        return f"{100 * a / b:5.1f}" if b else "    -"

    print("\n1. Events at the final whistle (both matches): precision of counted entries, recall of visible true events")
    print(f"{'run':24s} {'counted':>8s} {'prec':>6s} {'held':>6s} {'prec':>6s} {'rec(counted)':>13s} {'rec(held)':>10s}")
    for run, r in rows.items():
        f = r["final"]
        print(f"{run:24s} {f['counted']:8d} {pct(f['counted_true'], f['counted']):>6s} {f['held']:6d} "
              f"{pct(f['held_true'], f['held']):>6s} {pct(f['visible_counted'], f['visible']):>13s} "
              f"{pct(f['visible_held'], f['visible']):>10s}")
    print("   per type, counted precision / recall:")
    for run, r in rows.items():
        bt = r["final"]["by_type"]
        print(f"   {run:21s} " + "  ".join(f"{ty} {pct(v['counted_true'], v['counted']).strip()}/{pct(v['visible_counted'], v['visible']).strip()}"
                                           for ty, v in bt.items() if v["visible"] or v["counted"]))

    print("\n2. Visible true events counted (held) Δ s after they happened")
    print(f"{'run':24s} " + " ".join(f"{str(d):>13s}" for d in (*DELAYS, "end")))
    for run, r in rows.items():
        cells = []
        for d in (*DELAYS, "end"):
            c = r["curve"][str(d)]
            n = sum(c.values())
            cells.append(f"{pct(c.get('counted', 0), n)} ({pct(c.get('counted', 0) + c.get('pending', 0), n).strip()})")
        print(f"{run:24s} " + " ".join(f"{x:>13s}" for x in cells))

    print("\n3. Revisions (false entry = matching no true event). Entries once counted that are no longer counted "
          "at the end: false ones (right corrections, with the latency from first counted) and true ones (wrong); "
          "entries retracted after being written")
    print(f"{'run':24s} {'true/false':>11s} {'false counted':>14s} {'uncounted':>10s} {'latency p50':>12s} "
          f"{'true uncounted':>15s} {'wrong-corr.':>12s} {'retracted later f/t':>20s}")
    for run, r in rows.items():
        v = r["revisions"]
        lat = r["correction_latency"]
        fixed, wrong = v.get("false_uncounted", 0), v.get("true_uncounted", 0)
        print(f"{run:24s} {v.get('true', 0):5d}/{v.get('false', 0):<5d} {v.get('false_ever_counted', 0):14d} "
              f"{fixed:10d} {np.median(lat) if lat else float('nan'):11.0f}s {wrong:15d} "
              f"{pct(wrong, fixed + wrong):>12s} {v.get('false_retracted_later', 0):>14d}/{v.get('true_retracted_later', 0)}")

    print("\n4. Derived state for time t as known at t + L (every 30 s): score right, lineup slots right, "
          "lineups exact; count error per team: mean |memory - truth| (mean signed)")
    for lag in LAGS:
        print(f"   L = {lag} s")
        for run, r in rows.items():
            d = r["derived"][str(lag)]
            ce = "  ".join(f"{ty} {v['abs'] / v['n']:.2f} ({v['signed'] / v['n']:+.2f})" for ty, v in d["count_error"].items())
            print(f"   {run:21s} score {pct(d['score_right'], d['samples'])}  lineup slots "
                  f"{pct(d['lineup_slots_right'], d['lineup_slots'])}  exact {pct(d['lineup_exact'], d['samples'])}  {ce}")

    print("\n5. Identity: players named on true goals / cards / substitutions (right, wrong, unnamed); "
          "per sampled frame: tracks identified, and identities of players not on the pitch")
    for run, r in rows.items():
        w, f = r["event_players"], r["frames"]
        n = sum(w.values())
        print(f"{run:24s} events right {w.get('right', 0)}/{n}, wrong {w.get('wrong', 0)}, unnamed {w.get('unnamed', 0)};"
              f"  frames identified {pct(f.get('identified', 0), f.get('tracks', 0))}%, off the pitch "
              f"{pct(f.get('off_pitch', 0), f.get('identified', 0))}% of identified ({f.get('off_pitch', 0)})")

    print("\n6. The second justification that made entries sufficient: entries confirmed that way (share true)")
    kinds = ("overlap", "other window", "replay", "pairing", "verdict", "other")
    print(f"{'run':24s} " + " ".join(f"{k:>16s}" for k in kinds))
    for run, r in rows.items():
        c = r.get("confirmers", {})
        cells = [f"{c.get(f'{k}|true', 0) + c.get(f'{k}|false', 0)} ({pct(c.get(f'{k}|true', 0), c.get(f'{k}|true', 0) + c.get(f'{k}|false', 0)).strip()})"
                 for k in kinds]
        print(f"{run:24s} " + " ".join(f"{x:>16s}" for x in cells))


if __name__ == "__main__":
    main()
