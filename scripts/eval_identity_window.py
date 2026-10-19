"""Window-level (non-causal) identity resolution on the SoccerNet-GSR test set.

The live system sends a 30 s window of frames to the event analyzer only once the window is
complete (loop.window_seconds), so a track's identity can be decided from ALL of the window's
reads and applied to every frame in it, instead of frame by frame from the reads so far
(IdentityResolver). This script tries that on the cached perception of eval_identity_gsr.py
(each GSR sequence is one 30 s window) and scores it like scripts/diag_identity_bound.py: the
share of our track boxes matched to a ground-truth outfield player with a labelled number whose
identity (team + number) is right / wrong (a player is given but it is another one).

Methods
  causal   the real IdentityResolver (v2 voting: reads at p >= 0.9, >= 2 votes, per frame)
  vote     the same votes pooled over the whole window and applied to all of a track's frames
  soft     every read of the window is evidence: a track's evidence for number n is
           e(n) = sum over its frames of log(1 + p_frame(n) / eps), so a frame the reader cannot
           read adds nothing and a confident read adds log(1 + p / eps) (3.9 at p = 0.9). The best
           roster number wins when e >= tau and it leads the runner-up by tau.
Both window methods also enforce one player per track: two tracks in the same frame cannot be
the same player, so tracks are assigned best-evidence first and a player is taken away from every
track that shares a frame with a track already holding him.

Two rosters: `open` (every number 0-99 of both teams, as eval_identity_gsr.py) and `clip` (the
numbers of the players actually labelled in the sequence, about 22: what a real roster gives).

    python scripts/eval_identity_window.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackers.eval import box_iou

from diag_identity_bound import gt_players               # sibling scripts (scripts/ is on sys.path)
from eval_identity_gsr import OPEN_ROSTER
from eval_tracking_gsr import LABEL_PATHS, raw_candidates_path
from matchlive.common.cache import iter_frames, load_perception, perception_key
from matchlive.common.config import load_config
from matchlive.perception.identity import IdentityResolver

EPS = 0.02          # floor of a read's likelihood: a frame that reads another number cannot veto
TAUS = [2.0, 3.0, 4.0, 6.0, 8.0]


def assign(candidates, frames_of):
    """Greedy one-player-per-track assignment, best evidence first.

    candidates: list of (confidence, track key, player). frames_of: track key -> set of frames.
    A player held by a track is refused to every other track that shares a frame with it.
    """
    holders: dict[str, list] = defaultdict(list)       # player -> track keys holding him
    out: dict = {}
    for _, key, player in sorted(candidates, key=lambda c: -c[0]):
        if key in out or any(frames_of[key] & frames_of[h] for h in holders[player]):
            continue
        out[key] = player
        holders[player].append(key)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=EPS, help="floor of a read's likelihood in the soft evidence")
    args = ap.parse_args()
    eps = args.eps
    config = load_config()
    out_dir = config["paths"]["outputs"] / "gsr_eval"
    params = config["perception"]["identity"]
    hard, min_votes = params["jersey_min_prob"], params["min_number_votes"]

    # score[method][roster] = [right, wrong, boxes]
    score = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))

    def count(method, roster, pred, side, number):
        s = score[method][roster]
        s[2] += 1
        if pred is not None:
            s[0 if pred == f"{side}#{number}" else 1] += 1

    for label_path in LABEL_PATHS:
        labels = json.load(open(label_path))
        name = labels["info"]["name"]
        gt = gt_players(labels)
        key = perception_key(config, [label_path, raw_candidates_path(config, out_dir)])
        arrays, meta = load_perception(out_dir / "perception" / f"{name}_{key}.npz")
        kits = {kit: np.asarray(v, dtype=np.float32) for kit, v in meta["kits"].items()}

        # the clip's roster: the numbers of the labelled players of each side
        clip = {(side, number) for frame in gt.values() for _, number, _, side in frame if side}
        rosters = {"open": OPEN_ROSTER, "clip": {(side, number): f"{side}#{number}" for side, number in clip}}
        resolvers = {r: IdentityResolver(params, kits) for r in rosters}

        evidence = defaultdict(lambda: np.zeros(100))          # track key -> e(n)
        votes = defaultdict(Counter)                           # track key -> reads at p >= hard
        frames_of = defaultdict(set)
        matches = []                                           # (track key, side, number) per matched box
        for frame_idx, tracks, jersey, color, _ in iter_frames(arrays):
            said = {r: resolvers[r].update(frame_idx, tracks, jersey, color, rosters[r], {}) for r in rosters}
            for t, p in zip(tracks, jersey):
                frames_of[t.key].add(frame_idx)
                if not np.isnan(p[0]):
                    evidence[t.key] += np.log1p(p[:100] / eps)
                    top = int(p.argmax())
                    if top != 100 and p[top] >= hard:
                        votes[t.key][top] += 1
            truth = gt[frame_idx]
            if not (tracks and truth):
                continue
            sim = box_iou(np.array([t.bbox_ltwh for t in tracks]), np.array([t[2] for t in truth]), box_format="xywh")
            for r, c in zip(*linear_sum_assignment(-sim)):
                if sim[r, c] < 0.5 or tracks[r].role != "player" or not truth[c][3]:
                    continue
                _, number, _, side = truth[c]
                matches.append((tracks[r].key, side, number))
                for roster in rosters:
                    pid = said[roster][r].player_id
                    count("causal", roster, pid, side, number)

        # Each track's team after the whole window (the resolver pools the team votes).
        team = {k: resolvers["open"]._team(k) for k in frames_of}
        for roster_name, roster in rosters.items():
            # candidate (confidence, track, player) per method
            cands = {"vote": [], **{f"soft tau={tau}": [] for tau in TAUS}}
            for k in frames_of:
                sides = [team[k]] if team[k] else ["home", "away"]
                # hard votes pooled over the window
                for number, n in votes[k].items():
                    options = [s for s in sides if (s, str(number)) in roster]
                    if n >= min_votes and len(options) == 1:
                        cands["vote"].append((n, k, roster[(options[0], str(number))]))
                # soft evidence over the roster numbers of the track's team
                scored = sorted(((evidence[k][int(n)], s, n) for s in sides for (s2, n) in roster if s2 == s), reverse=True)
                if scored and scored[0][0] > 0:
                    best, runner = scored[0], (scored[1][0] if len(scored) > 1 else 0.0)
                    for tau in TAUS:
                        if best[0] >= tau and best[0] - runner >= tau:
                            cands[f"soft tau={tau}"].append((best[0] - runner, k, roster[(best[1], best[2])]))
            for method, c in cands.items():
                got = assign(c, frames_of)
                for k, side, number in matches:
                    count(method, roster_name, got.get(k), side, number)
        print(f"{name}: {len(matches)} matched boxes", flush=True)

    print(f"\n{'method':18s} " + " ".join(f"{r + ' right':>12s} {r + ' wrong':>12s}" for r in ("open", "clip")))
    for method, per in score.items():
        print(f"{method:18s} " + " ".join(f"{per[r][0] / per[r][2]:12.3f} {per[r][1] / per[r][2]:12.3f}" for r in ("open", "clip")))
    print(f"boxes: {score['causal']['open'][2]}")


if __name__ == "__main__":
    main()
