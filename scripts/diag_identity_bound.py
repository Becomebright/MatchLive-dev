"""Where does player-identification coverage go? Upper bounds on the SoccerNet-GSR test set.

Reads the per-sequence perception arrays that eval_identity_gsr.py caches (run it first with the
same config), matches our track boxes to the ground-truth boxes of outfield players with a
labelled number (Hungarian on IoU >= 0.5 per frame, as in eval_identity_gsr.py), and asks, for
the matched boxes, how much of them could be named by a reader of this quality under
progressively more generous rules. Every row is the share of those boxes whose number is
right; only the number is scored here (the team is right 0.96 of the time and is ignored).

  current      the real IdentityResolver (what eval_identity_gsr.py reports as player precision
               times coverage)
  vote         per-track majority vote, causal, reads with p >= 0.9, >= 2 votes, no
               one-player-per-track assignment: the resolver without its assignment step
  vote p>=0.3  the same with reads at p >= 0.3 and a single vote
  non-causal   the final vote of our track applied to all its frames (no streaming delay)
  merged       votes pooled over all our tracks of one ground-truth player (oracle track
               linking), causal / non-causal
  reader cap   every box of a ground-truth player on which at least one box has the number
               read right (p >= 0.9 / p >= 0.3): what a perfect fusion of this reader could get
  no read      ground-truth players on which no box is ever read right at p >= 0.3

Also: the same shares by the player's largest box height, vote accuracy, and a list of the
`no read` players (outputs/gsr_eval/diag/no_read.json) with `--montage`, which also writes
contact sheets of their largest boxes to look at: is the number visible there at all?

    python scripts/diag_identity_bound.py
    python scripts/diag_identity_bound.py --montage 60
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from trackers.eval import box_iou

from eval_identity_gsr import OPEN_ROSTER, SCALE        # sibling script (scripts/ is on sys.path)
from eval_tracking_gsr import LABEL_PATHS, raw_candidates_path
from matchlive.common.cache import iter_frames, load_perception, perception_key
from matchlive.common.config import load_config
from matchlive.perception.identity import IdentityResolver

UNREADABLE = 100
HEIGHT_BUCKETS = [0, 60, 80, 100, 1000]            # largest 720p box height of a ground-truth player


def gt_players(labels):
    """frame -> list of (gt track id, number, ltwh at 720p, team) for outfield players with a number."""
    frame_of_image = {im["image_id"]: int(im["file_name"][:-4]) for im in labels["images"]}
    gt = defaultdict(list)
    for a in labels["annotations"]:
        attr = a.get("attributes") or {}
        if a.get("supercategory") == "object" and a["category_id"] == 1 and attr.get("jersey"):
            b = a["bbox_image"]
            gt[frame_of_image[a["image_id"]]].append(
                (a["track_id"], attr["jersey"], (b["x"] * SCALE, b["y"] * SCALE, b["w"] * SCALE, b["h"] * SCALE),
                 {"left": "home", "right": "away"}.get(attr.get("team"))))
    return gt


def majority(votes: Counter, min_votes: int):
    """The most-voted number, or None while it has fewer than `min_votes` votes."""
    if not votes:
        return None
    number, n = votes.most_common(1)[0]
    return number if n >= min_votes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", type=int, default=0, metavar="N",
                    help="also draw contact sheets of N random `no read` players' largest boxes")
    args = ap.parse_args()

    config = load_config()
    out_dir = config["paths"]["outputs"] / "gsr_eval"
    diag_dir = out_dir / "diag"
    diag_dir.mkdir(exist_ok=True)
    params = config["perception"]["identity"]
    hard, soft = params["jersey_min_prob"], 0.3

    # One record per matched box, in (sequence, frame) order:
    # (seq, frame, our key, gt id, number, p of the true number, top1, p of top1, ltwh at 720p, resolver's number or None)
    boxes = []
    for label_path in LABEL_PATHS:
        labels = json.load(open(label_path))
        name = labels["info"]["name"]
        gt = gt_players(labels)
        key = perception_key(config, [label_path, raw_candidates_path(config, out_dir)])
        arrays, meta = load_perception(out_dir / "perception" / f"{name}_{key}.npz")
        kits = {kit: np.asarray(v, dtype=np.float32) for kit, v in meta["kits"].items()}
        resolver = IdentityResolver(params, kits)
        for frame_idx, tracks, jersey, color, _ in iter_frames(arrays):
            identities = resolver.update(frame_idx, tracks, jersey, color, OPEN_ROSTER, {})   # keep the resolver's state running
            truth = gt[frame_idx]
            if not (tracks and truth):
                continue
            sim = box_iou(np.array([t.bbox_ltwh for t in tracks]), np.array([t[2] for t in truth]), box_format="xywh")
            for r, c in zip(*linear_sum_assignment(-sim)):
                if sim[r, c] < 0.5 or tracks[r].role != "player":
                    continue
                track_id, number, ltwh, team = truth[c]
                p = jersey[r]
                top1 = int(p.argmax())
                said = identities[r].player_id
                boxes.append((name, int(frame_idx), (name, tracks[r].key), (name, track_id), int(number), float(p[int(number)]),
                              top1, float(p[top1]), ltwh,
                              int(said.split("#")[1]) if said and said.split("#")[0] == team else None))
        print(f"{name}: {len(boxes)} boxes so far", flush=True)

    n = len(boxes)
    print(f"\n{n} matched outfield-player boxes with a labelled number, "
          f"{len({b[3] for b in boxes})} ground-truth players, {len({b[2] for b in boxes})} of our tracks\n")

    def read_right(b, thr):          # the box's top-1 is the true number, with probability >= thr
        return b[6] == b[4] and b[7] >= thr

    # --- rules that vote over a track's boxes -------------------------------------------------
    def causal(group_of, thr, min_votes):
        votes, right, wrong = defaultdict(Counter), 0, 0
        for b in boxes:                              # (seq, frame) order = streaming order
            g = group_of(b)
            if read_right(b, thr) or (b[6] != UNREADABLE and b[7] >= thr):
                votes[g][b[6]] += 1
            said = majority(votes[g], min_votes)
            right += said == b[4]
            wrong += said is not None and said != b[4]
        return right / n, wrong / n

    def final(group_of, thr, min_votes):
        votes = defaultdict(Counter)
        for b in boxes:
            if b[6] != UNREADABLE and b[7] >= thr:
                votes[group_of(b)][b[6]] += 1
        said = [majority(votes[group_of(b)], min_votes) for b in boxes]
        return (sum(x == b[4] for x, b in zip(said, boxes)) / n,
                sum(x is not None and x != b[4] for x, b in zip(said, boxes)) / n)

    ours, gtid = (lambda b: b[2]), (lambda b: b[3])
    # --- reader cap: ground-truth players with at least one right read ------------------------
    right_at = {thr: Counter(b[3] for b in boxes if read_right(b, thr)) for thr in (hard, soft)}
    players = Counter(b[3] for b in boxes)
    rows = [("current (IdentityResolver)", (sum(b[9] == b[4] for b in boxes) / n, sum(b[9] is not None and b[9] != b[4] for b in boxes) / n)),
            (f"vote, causal, p>={hard}, >=2 votes", causal(ours, hard, 2)),
            (f"vote, causal, p>={soft}, 1 vote", causal(ours, soft, 1)),
            (f"vote, non-causal, p>={hard}, >=2 votes", final(ours, hard, 2)),
            (f"merged by GT player, causal, p>={hard}, >=2 votes", causal(gtid, hard, 2)),
            (f"merged by GT player, non-causal, p>={hard}, >=2 votes", final(gtid, hard, 2)),
            (f"merged by GT player, non-causal, p>={soft}, 1 vote", final(gtid, soft, 1)),
            (f"reader cap: GT players with >= 1 right read at p>={hard}", (sum(players[g] for g in right_at[hard]) / n, None)),
            (f"reader cap: GT players with >= 1 right read at p>={soft}", (sum(players[g] for g in right_at[soft]) / n, None)),
            (f"no read: GT players never read right at p>={soft}", (sum(c for g, c in players.items() if g not in right_at[soft]) / n, None))]
    print(f"{'rule':66s} {'right':>7s} {'wrong':>7s}   (share of boxes; wrong = a number was given but it is another one)")
    for label, (right, wrong) in rows:
        print(f"{label:66s} {right:7.3f} " + (f"{wrong:7.3f}" if wrong is not None else ""))

    # --- vote quality --------------------------------------------------------------------------
    confident = [b for b in boxes if b[6] != UNREADABLE and b[7] >= hard]
    print(f"\nboxes read at p>={hard}: {len(confident) / n:.3f} of all, {np.mean([b[6] == b[4] for b in confident]):.3f} right")
    wrong = Counter()
    for b in confident:
        wrong["right" if b[6] == b[4] else "wrong"] += 1
    print(f"  {dict(wrong)}")

    # --- by the player's largest box ----------------------------------------------------------
    height = defaultdict(float)
    for b in boxes:
        height[b[3]] = max(height[b[3]], b[8][3])
    merged = defaultdict(Counter)
    for b in boxes:
        if b[6] != UNREADABLE and b[7] >= hard:
            merged[b[3]][b[6]] += 1
    print(f"\n{'largest box (720p)':>18s} {'GT players':>11s} {'boxes':>7s} {'>=1 right read':>15s} {'merged>=2 votes right':>22s} {'resolver right':>15s}")
    for lo, hi in zip(HEIGHT_BUCKETS[:-1], HEIGHT_BUCKETS[1:]):
        ids = {g for g, h in height.items() if lo <= h < hi}
        sel = [b for b in boxes if b[3] in ids]
        if not sel:
            continue
        print(f"{f'{lo}-{hi}':>18s} {len(ids):11d} {len(sel):7d} "
              f"{len(ids & set(right_at[hard])) / len(ids):15.3f} "
              f"{np.mean([majority(merged[b[3]], 2) == b[4] for b in sel]):22.3f} "
              f"{np.mean([b[9] == b[4] for b in sel]):15.3f}")

    # --- by the number itself: is the reader weak on numbers its training data rarely has? -------
    number_of = {b[3]: b[4] for b in boxes}
    print(f"\n{'number':>8s} {'GT players':>11s} {'boxes':>7s} {'>=1 right read':>15s} {'resolver right':>15s}")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]:
        ids = {g for g, v in number_of.items() if lo <= v < hi}
        sel = [b for b in boxes if b[3] in ids]
        print(f"{f'{lo}-{hi - 1}':>8s} {len(ids):11d} {len(sel):7d} {len(ids & set(right_at[soft])) / len(ids):15.3f} "
              f"{np.mean([b[9] == b[4] for b in sel]):15.3f}")

    # --- the players never read right --------------------------------------------------------
    never = [g for g in players if g not in right_at[soft]]
    print(f"\nGT players never read right at p>={soft}: {len(never)} of {len(players)} "
          f"({sum(players[g] for g in never) / n:.3f} of boxes)")
    biggest = {g: sorted((b for b in boxes if b[3] == g), key=lambda b: -b[8][3])[:6] for g in never}
    (diag_dir / "no_read.json").write_text(json.dumps(
        [{"seq": g[0], "gt_track": g[1], "number": biggest[g][0][4], "boxes": players[g],
          "frames": [b[1] for b in biggest[g]], "top1_of_those": [b[6] for b in biggest[g]]} for g in never]))

    if args.montage:
        draw_montage(random.Random(0).sample(sorted(never), min(args.montage, len(never))), biggest, diag_dir)


def draw_montage(chosen, biggest, diag_dir, per_sheet=10, height=200):
    """Contact sheets: one row per player, its 6 largest boxes (full box plus a margin, cut from the
    original 1080p frame); the row is captioned with the true number and what the reader said."""
    gsr = diag_dir.parents[2] / "dataset/SoccerNetGS/test"
    for sheet, start in enumerate(range(0, len(chosen), per_sheet)):
        rows = []
        for seq, track_id in chosen[start:start + per_sheet]:
            tiles = []
            for b in biggest[(seq, track_id)]:
                frame = cv2.imread(str(gsr / seq / "img1" / f"{b[1]:06d}.jpg"))
                l, t, w, h = (v / SCALE for v in b[8])
                x0, y0 = max(int(l - 0.1 * w), 0), max(int(t - 0.05 * h), 0)
                crop = frame[y0:int(t + 1.05 * h), x0:int(l + 1.1 * w)]
                crop = cv2.resize(crop, (max(int(crop.shape[1] * height / crop.shape[0]), 1), height))
                canvas = np.full((height, 130, 3), 40, np.uint8)       # fixed-width tile
                canvas[:, :min(crop.shape[1], 130)] = crop[:, :130]
                tiles.append(canvas)
            row = np.hstack(tiles)
            caption = np.full((24, row.shape[1], 3), 255, np.uint8)
            cv2.putText(caption, f"{seq} player {track_id}  number {biggest[(seq, track_id)][0][4]}  ({sheet * per_sheet + len(rows) // 2})",
                        (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            rows += [caption, row]
        cv2.imwrite(str(diag_dir / f"no_read_{sheet}.jpg"), np.vstack(rows))
        print(f"wrote {diag_dir / f'no_read_{sheet}.jpg'}")


if __name__ == "__main__":
    main()
