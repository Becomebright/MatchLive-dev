"""Score event candidates (scripts/run_events.py output) against ground-truth events.

    python scripts/eval_events.py --match "2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea" [--review]

Ground truth is the match's Labels-caption_with_gt.json (SoccerNet-Caption format), timed by
`gt_gameTime` (the corrected time; `gameTime` only where it is missing). Its labels map to
our types as agreed with the user:
    goal -> goal; saved by goal-keeper, shot off target -> shot; yellow card -> yellow_card;
    substitution -> substitution; corner -> corner; foul with no card -> foul;
    off side -> offside; lead to corner, injury -> not scored.

Matching (one-to-one, per type): a live candidate matches a true event within ±`tol` s; a
replay candidate matches one inside [its window's start - W, the moment it is seen]
(DESIGN.md §6.5). Candidates are first merged into proposed events the same way — the
overlap of two windows proposes one event twice — which is roughly what the consistency
check's matching (§6.4) will do.

Caption labels are not exhaustive (only what commentary mentioned), so recall is reliable
while precision is only reported for the types captions cover completely (goal, cards,
substitution); for the rest, the unmatched proposals are listed for manual review, and
--review writes a contact sheet for each (outputs/review_events/<match>/): 24 frames at 2 fps
from 6 s before to 6 s after the proposal (the flag, whistle or restart that decides an
offside or a foul usually comes after it), titled with the nearest same-type true event and
proposal, so each can be judged by eye (missing caption / duplicate / wrong type / no event).
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from matchlive.common.config import load_config
from matchlive.common.match import Match

GT_TYPE = {"goal": "goal", "saved by goal-keeper": "shot", "shot off target": "shot",
           "yellow card": "yellow_card", "red card": "red_card", "substitution": "substitution",
           "corner": "corner", "foul with no card": "foul", "off side": "offside"}


def clock_to_time(s: str) -> tuple[int, float]:
    half, clock = s.split(" - ")
    minutes, seconds = clock.split(":")
    return int(half), int(minutes) * 60 + float(seconds)


def ground_truth(match_dir) -> list[dict]:
    """The scored events of Labels-caption_with_gt.json: type, time (half, second), the
    involved players' hashes and the caption label."""
    with open(match_dir / "Labels-caption_with_gt.json") as f:
        anns = json.load(f)["annotations"]
    return [{"type": GT_TYPE[a["label"]], "time": clock_to_time(a["gt_gameTime"] or a["gameTime"]),
             "players": [p["hash"] for p in a["involved_players"]], "label": a["label"]}
            for a in anns if a["label"] in GT_TYPE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--tol", type=float, default=10.0, help="± seconds for a live candidate")
    ap.add_argument("--review", action="store_true", help="write a contact sheet per unmatched proposal")
    args = ap.parse_args()

    config = load_config()
    W = config["events"]["replay_lookback"]
    match = Match(config, args.match)
    truth = ground_truth(match.dir)

    # Proposals: candidates merged per type. A live one covers [t - tol, t + tol]; a replay
    # one covers [window start - W, t]. Candidates are merged when their spans overlap.
    proposals = []
    for half in (1, 2):
        path = config["paths"]["outputs"] / "events" / args.match / f"half{half}.jsonl"
        for line in open(path):
            r = json.loads(line)
            for c in r.get("candidates", []):
                lo, hi = ((r["window"][0] - W, c["second"]) if c["replay"]
                          else (c["second"] - args.tol, c["second"] + args.tol))
                actor = next((p for (role, _), p in zip(c["participants"], c["players"]) if role == "player"), None)
                for prop in proposals:
                    if (prop["type"] == c["type"] and prop["half"] == half
                            and lo <= prop["hi"] and prop["lo"] <= hi):
                        # A live sighting replaces a replay-only span (it is far tighter);
                        # otherwise the proposal covers the union of its sightings.
                        if prop["replay_only"] and not c["replay"]:
                            prop.update(lo=lo, hi=hi, replay_only=False)
                        elif prop["replay_only"] == c["replay"]:
                            prop.update(lo=min(prop["lo"], lo), hi=max(prop["hi"], hi))
                        prop["actors"].append(actor)
                        break
                else:
                    proposals.append({"type": c["type"], "team": c["team"], "half": half, "lo": lo, "hi": hi,
                                      "replay_only": c["replay"], "actors": [actor], "second": c["second"]})

    # One-to-one matching per type, closest first.
    pairs = sorted(((abs((p["lo"] + p["hi"]) / 2 - t["time"][1]), i, j)
                    for i, t in enumerate(truth) for j, p in enumerate(proposals)
                    if p["type"] == t["type"] and p["half"] == t["time"][0] and p["lo"] <= t["time"][1] <= p["hi"]))
    matched_t, matched_p = {}, {}
    for _, i, j in pairs:
        if i not in matched_t and j not in matched_p:
            matched_t[i], matched_p[j] = j, i
    # A shot proposed at a goal is the same moment seen as a shot, not a false alarm.
    for j, p in enumerate(proposals):
        if j not in matched_p and p["type"] == "shot" and any(
                t["type"] == "goal" and t["time"][0] == p["half"] and p["lo"] <= t["time"][1] <= p["hi"] for t in truth):
            matched_p[j] = None

    print(f"{len(truth)} true events, {len(proposals)} proposed events (tol ±{args.tol:.0f} s, W {W} s)\n")
    print(f"{'type':14s} {'true':>5s} {'found':>6s} {'recall':>7s} {'via replay only':>16s} "
          f"{'proposed':>9s} {'precision':>10s} {'actor right':>12s}")
    types = sorted({t["type"] for t in truth} | {p["type"] for p in proposals})
    for ty in types:
        ts = [i for i, t in enumerate(truth) if t["type"] == ty]
        ps = [j for j, p in enumerate(proposals) if p["type"] == ty]
        found = [i for i in ts if i in matched_t]
        replay_only = sum(proposals[matched_t[i]]["replay_only"] for i in found)
        tp = sum(1 for j in ps if j in matched_p)
        # Attribution: the proposal's actor (identified player of the "player" role) is
        # one of the players the caption names.
        judged = [(truth[i], proposals[matched_t[i]]) for i in found if truth[i]["players"]]
        right = sum(any(a in t["players"] for a in p["actors"] if a) for t, p in judged)
        named = sum(any(p["actors"]) for _, p in judged)
        complete = ty in ("goal", "yellow_card", "red_card", "substitution")
        print(f"{ty:14s} {len(ts):5d} {len(found):6d} {len(found) / max(len(ts), 1):7.2f} {replay_only:16d} "
              f"{len(ps):9d} {(f'{tp / max(len(ps), 1):.2f}' if complete else '(review)'):>10s} "
              f"{f'{right}/{named}' if named else '-':>12s}")
    n_t = len(truth)
    print(f"\n{'all':14s} {n_t:5d} {len(matched_t):6d} {len(matched_t) / n_t:7.2f}")

    print("\nMissed true events:")
    for i, t in enumerate(truth):
        if i not in matched_t:
            print(f"  {t['time'][0]} - {int(t['time'][1] // 60):02d}:{int(t['time'][1] % 60):02d}  {t['label']}")
    print("\nUnmatched proposals (for review):")
    unmatched = [p for j, p in enumerate(proposals) if j not in matched_p]
    for p in unmatched:
        s = p["second"]
        print(f"  {p['half']} - {int(s // 60):02d}:{int(s % 60):02d}  {p['type']}{'  (replay)' if p['replay_only'] else ''}")
    if args.review:
        write_review_sheets(config, match, unmatched, truth, proposals)


def write_review_sheets(config, match, unmatched, truth, proposals):
    out_dir = config["paths"]["outputs"] / "review_events" / match.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for half in (1, 2):
        cap = cv2.VideoCapture(str(match.video_path(half)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for p in (p for p in unmatched if p["half"] == half):
            s = p["second"]
            # Nearest same-type true event and other proposal, as signed seconds from this one.
            near_t = min((t["time"][1] - s for t in truth if t["type"] == p["type"] and t["time"][0] == half),
                         key=abs, default=None)
            near_p = min((q["second"] - s for q in proposals if q is not p and q["type"] == p["type"] and q["half"] == half),
                         key=abs, default=None)
            tiles = []
            for k in range(24):
                t = max(0.0, s - 6 + k * 0.5)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ok, frame = cap.read()
                tile = cv2.resize(frame, (480, 270)) if ok else np.zeros((270, 480, 3), np.uint8)
                cv2.putText(tile, f"{int(t // 60):02d}:{t % 60:04.1f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255) if abs(t - s) < 0.3 else (255, 255, 255), 2)
                tiles.append(tile)
            sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, 24, 6)])
            title = (f"{p['type']} team {p.get('team') or '?'}{' REPLAY' if p['replay_only'] else ''} at "
                     f"{half}-{int(s // 60):02d}:{s % 60:04.1f} | nearest true {p['type']}: "
                     f"{'none' if near_t is None else f'{near_t:+.0f}s'} | nearest other proposal: "
                     f"{'none' if near_p is None else f'{near_p:+.0f}s'}")
            bar = np.zeros((36, sheet.shape[1], 3), np.uint8)
            cv2.putText(bar, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            name = f"{half}_{int(s // 60):02d}{int(s % 60):02d}_{p['type']}.jpg"
            cv2.imwrite(str(out_dir / name), np.vstack([bar, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"\nreview sheets: {out_dir}")


if __name__ == "__main__":
    main()
