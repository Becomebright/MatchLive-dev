"""Visual check of shot cuts and tracking on a match video segment.

Runs detection -> shot-cut detection -> player tracking + ball selection over `--seconds`
of one half and writes into outputs/check_tracking/:
  <name>.mp4       track overlay, with the shot id in the corner (red on a cut frame)
  <name>_cuts.jpg  every detected cut as a (frame before | cut frame) pair, to count
                   false and missed cuts by eye

    python scripts/check_tracking.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1 --start 600 --seconds 120
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.perception.detection import RFDETRDetector
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import PlayerTracker, select_ball
from matchlive.perception.visualization import draw_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--start", type=float, default=600.0, help="start second within the half")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--algorithm", choices=["botsort", "bytetrack"], help="override the config")
    args = ap.parse_args()

    config = load_config()
    paths, perception = config["paths"], config["perception"]
    if args.algorithm:
        perception["tracking"]["algorithm"] = args.algorithm
    match = Match(config, args.match)

    cap = cv2.VideoCapture(str(match.video_path(args.half)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(round(args.start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    detector = RFDETRDetector(perception["detection"], paths["checkpoints"], paths["cache"])
    shots = ShotDetector(perception["shots"]["cut_threshold"])
    tracker = PlayerTracker(perception["tracking"], frame_rate=fps)
    print("providers:", detector.providers, "| tracker:", perception["tracking"]["algorithm"])

    out_dir = paths["outputs"] / "check_tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    # e.g. "2016-09-16_h1_600s_botsort" (the match date is unique in our dataset)
    name = f"{args.match[:10]}_h{args.half}_{int(args.start)}s_{perception['tracking']['algorithm']}"
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    cut_pairs, prev_frame, prev_shot_id, frames = [], None, 0, 0
    timing = {"detect": 0.0, "shots": 0.0, "track": 0.0}
    while frames < int(round(args.seconds * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        detections = detector.predict(frame)
        t1 = time.perf_counter()
        shot_id = shots.update(frame)
        t2 = time.perf_counter()
        tracks = tracker.update(detections, frame, shot_id)
        ball = select_ball(detections, perception["ball"]["min_score"])
        t3 = time.perf_counter()
        timing["detect"] += t1 - t0
        timing["shots"] += t2 - t1
        timing["track"] += t3 - t2

        # The first frame is always shot 0, so it can never count as a cut.
        is_cut = shot_id != prev_shot_id
        prev_shot_id = shot_id
        if is_cut:
            pair = np.hstack([cv2.resize(prev_frame, (320, 180)), cv2.resize(frame, (320, 180))])
            second = (start_frame + frames) / fps
            cv2.putText(pair, f"{int(second // 60)}:{second % 60:05.2f} d={shots.last_distance:.2f}",
                        (4, 16), 0, 0.5, (0, 255, 255), 1)
            cut_pairs.append(pair)
        prev_frame = frame

        vis = draw_tracks(frame, tracks, ball, detections=detections)
        cv2.putText(vis, f"shot {shot_id}  d={shots.last_distance:.2f}", (20, 690), 0, 0.8,
                    (0, 0, 255) if is_cut else (255, 255, 255), 2)
        writer.write(vis)
        frames += 1
    writer.release()

    if cut_pairs:
        if len(cut_pairs) % 2:
            cut_pairs.append(np.zeros_like(cut_pairs[0]))
        grid = np.vstack([np.hstack(cut_pairs[j:j + 2]) for j in range(0, len(cut_pairs), 2)])
        cv2.imwrite(str(out_dir / f"{name}_cuts.jpg"), grid)
    print(f"{frames} frames, {shots.shot_id} cuts -> {out_dir / name}.mp4")
    print("ms/frame: " + "  ".join(f"{k} {1000 * v / frames:.1f}" for k, v in timing.items()))


if __name__ == "__main__":
    main()
