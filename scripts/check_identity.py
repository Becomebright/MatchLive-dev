"""Visual check of identity on a match video segment, with the match's real kit colours and roster.

Runs (or reuses) the raw perception cache for the segment, streams it through the
IdentityResolver, and writes outputs/check_identity/<name>.mp4: ellipses in team colour
with jersey number and name once identified. Prints how many track boxes got a team /
a player, and the off-roster reads kept as evidence.

    python scripts/check_identity.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1 --start 600 --seconds 120
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import cv2

from matchlive.common.cache import iter_frames, load_perception
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.perception.detection import to_detections
from matchlive.perception.identity import IdentityResolver, kit_bgr, kit_lab, roster_index
from matchlive.perception.tracking import select_ball
from matchlive.perception.visualization import draw_tracks
from matchlive.runtime.runner import MatchRunner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=600.0, help="start second within the half")
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args()

    config = load_config()
    match = Match(config, args.match)
    start_frame, n_frames = int(round(args.start * 25)), int(round(args.seconds * 25))
    arrays, meta = load_perception(MatchRunner(config).run_perception(match, args.half, start_frame, n_frames))
    decoded = {f: rest for f, *rest in iter_frames(arrays)}

    players = match.metadata["players"]
    # Phase 0 has no substitution tracking yet, so the whole squad counts as on the pitch.
    roster = roster_index(players)
    resolver = IdentityResolver(config["perception"]["identity"], kit_lab(match.metadata["jersey_colors"]))
    team_bgr = kit_bgr(match.metadata["jersey_colors"])
    labels = {p["hash"]: (str(p["jersey_number"]), p["short_name"]) for p in players}

    out_dir = config["paths"]["outputs"] / "check_identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.match[:10]}_h{args.half}_{int(args.start)}s"
    cap = cv2.VideoCapture(str(match.video_path(args.half)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], size)

    stats, identified, teams = Counter(), defaultdict(set), {}
    for frame_idx in range(start_frame, start_frame + meta["n_frames"]):
        ok, frame = cap.read()
        if not ok:
            break
        tracks, jersey, color, det = decoded.get(frame_idx, ([], [], [], arrays["det"][:0, 1:]))
        tracks = resolver.update(frame_idx, tracks, jersey, color, roster, confirmed={})
        for t in tracks:
            stats["boxes"] += 1
            if t.team:
                stats[f"team {t.team}"] += 1
            if t.player_id:
                stats["player"] += 1
                identified[t.team].add(labels[t.player_id][0])
            teams[t.key] = t.team
        dets = to_detections(det[:, :4], det[:, 4], det[:, 5], 0.0)
        ball = select_ball(dets, config["perception"]["ball"]["min_score"])
        writer.write(draw_tracks(frame, tracks, ball, team_bgr, labels, dets))
    writer.release()

    total = max(stats.pop("boxes", 0), 1)
    print(f"{out_dir / name}.mp4")
    print(f"track boxes: {total} | " + "  ".join(f"{k} {v / total:.3f}" for k, v in sorted(stats.items())))
    for team in ("home", "away"):
        print(f"{team} numbers identified: {sorted(identified[team], key=int)}")
    off = Counter((team, number) for _, team, number, _ in resolver.off_roster_reads(teams, roster))
    print("off-roster reads (team, number): tracks", dict(off.most_common(10)))


if __name__ == "__main__":
    main()
