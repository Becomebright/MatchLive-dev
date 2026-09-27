"""Tracking and ball metrics on the SoccerNet-GSR test set.

Runs our perception front end — RF-DETR detections, shot-cut resets, the player tracker,
per-frame ball selection — over every GSR test sequence and scores it against the GSR
ground truth:
  people: player + goalkeeper tracks vs GT players and goalkeepers with the `trackers`
          library's TrackEval-compatible evaluator (HOTA / IDF1 / MOTA). Class-agnostic,
          so a keeper detected as a player still counts; GT referees and "other" are
          distractors — a track box on them is neither rewarded nor penalized, since we do
          not track referees at all.
  ball:   the one selected ball per frame vs the GT ball (not tracked, so no MOT
          metrics): a frame's ball is a hit when its centre is within 15 px of the GT
          centre; precision over frames with a selected ball, recall over frames with a
          GT ball.

Raw detector candidates are cached in outputs/gsr_eval/rfdetr-l_raw.npy (one row per query
with score > 0.01: image_id, l, t, w, h, score, detector class id) and computed with our
detector if missing; each frame's detections are rebuilt from them with the detector's own
filter (`to_detections`), so the cache serves any confidence threshold.

    python scripts/eval_tracking_gsr.py --algorithm bytetrack
    python scripts/eval_tracking_gsr.py --algorithm botsort --set track_activation_threshold=0.7 --name botsort_t07
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import cv2
import numpy as np
import yaml
from trackers.eval import evaluate_mot_sequences

from matchlive.common.config import REPO_ROOT, load_config
from matchlive.perception.detection import RFDETRDetector, to_detections
from matchlive.perception.shots import ShotDetector
from matchlive.perception.tracking import PlayerTracker, select_ball

GSR_DIR = REPO_ROOT / "dataset/SoccerNetGS/test"
LABEL_PATHS = sorted(GSR_DIR.glob("SNGS-*/Labels-GameState.json"))


def load_raw_candidates(config, out_dir):
    """Raw candidates of every GSR test frame: rows (image_id, l, t, w, h, score, class id)."""
    path = out_dir / "rfdetr-l_raw.npy"
    if path.exists():
        return np.load(path)
    paths = config["paths"]
    detector = RFDETRDetector(config["perception"]["detection"], paths["checkpoints"], paths["cache"])
    rows = []
    for label_path in LABEL_PATHS:
        for image in json.load(open(label_path))["images"]:
            ltwh, scores, class_ids = detector.predict_raw(cv2.imread(str(label_path.parent / "img1" / image["file_name"])))
            keep = scores > 0.01
            rows += [(int(image["image_id"]), *box, s, c) for box, s, c in zip(ltwh[keep], scores[keep], class_ids[keep])]
    raw = np.asarray(rows, dtype=np.float64)
    np.save(path, raw)
    return raw


def write_ground_truth(labels, gt_dir):
    """Write the people MOT ground truth; return the GT ball centre per frame."""
    # MOT GT rows: frame, id, l, t, w, h, consider-flag, class. Class 1 is scored,
    # class 8 is a distractor (see the module docstring).
    frame_of_image = {im["image_id"]: int(im["file_name"][:-4]) for im in labels["images"]}
    rows, ball_centre = [], {}
    for a in labels["annotations"]:
        if a.get("supercategory") != "object":
            continue
        b, cat = a["bbox_image"], a["category_id"]
        frame = frame_of_image[a["image_id"]]
        if cat == 4:
            ball_centre[frame] = (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
        else:
            rows.append(f"{frame},{a['track_id']},{b['x']},{b['y']},{b['w']},{b['h']},1,{1 if cat in (1, 2) else 8}")
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / f"{labels['info']['name']}.txt").write_text("\n".join(rows) + "\n")
    return ball_centre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", choices=["botsort", "bytetrack"], required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one tracker parameter of the chosen algorithm")
    ap.add_argument("--name", help="run name (default: the algorithm)")
    args = ap.parse_args()

    config = load_config()
    tracking = config["perception"]["tracking"]
    tracking["algorithm"] = args.algorithm
    for item in args.set:
        key, value = item.split("=", 1)
        tracking[args.algorithm][key] = yaml.safe_load(value)
    run_name = args.name or args.algorithm
    out_dir = config["paths"]["outputs"] / "gsr_eval"
    run_dir = out_dir / "tracking" / run_name
    gt_dir = out_dir / "gt" / "people"
    (run_dir / "people").mkdir(parents=True, exist_ok=True)
    min_ball_score = config["perception"]["ball"]["min_score"]

    raw = load_raw_candidates(config, out_dir)
    confidence = config["perception"]["detection"]["confidence"]
    raw = raw[raw[:, 5] > confidence]          # just a speed-up: to_detections cuts the same
    rows_of_image = defaultdict(list)
    for row in raw:
        rows_of_image[int(row[0])].append(row[1:])

    cuts, track_seconds, n_frames = {}, 0.0, 0
    ball_hits = ball_selected = ball_gt = 0
    for label_path in LABEL_PATHS:
        labels = json.load(open(label_path))
        name = labels["info"]["name"]
        gt_ball = write_ground_truth(labels, gt_dir)
        ball_gt += len(gt_ball)
        shots = ShotDetector(config["perception"]["shots"]["cut_threshold"])
        tracker = PlayerTracker(tracking, frame_rate=labels["info"]["frame_rate"])
        # Tracks are keyed (shot_id, track_id); the MOT file needs one integer id each.
        mot_id: dict[tuple, int] = {}
        lines = []
        for image in sorted(labels["images"], key=lambda im: im["file_name"]):
            frame_idx = int(image["file_name"][:-4])
            frame = cv2.imread(str(GSR_DIR / name / "img1" / image["file_name"]))
            shot_id = shots.update(frame)
            t0 = time.perf_counter()
            rows = np.asarray(rows_of_image[int(image["image_id"])]).reshape(-1, 6)
            detections = to_detections(rows[:, :4], rows[:, 4], rows[:, 5], confidence)
            tracks = tracker.update(detections, frame, shot_id)
            track_seconds += time.perf_counter() - t0
            n_frames += 1
            for tr in tracks:
                tid = mot_id.setdefault(tr.key, len(mot_id) + 1)
                l, t, w, h = tr.bbox_ltwh
                lines.append(f"{frame_idx},{tid},{l:.2f},{t:.2f},{w:.2f},{h:.2f},{tr.score:.4f},-1,-1,-1")
            ball = select_ball(detections, min_ball_score)
            if ball is not None:
                ball_selected += 1
                l, t, w, h = ball.bbox_ltwh
                gt = gt_ball.get(frame_idx)
                ball_hits += gt is not None and np.hypot(l + w / 2 - gt[0], t + h / 2 - gt[1]) <= 15
        cuts[name] = shots.shot_id
        (run_dir / "people" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{name}: {cuts[name]} cuts, {len(mot_id)} tracks", flush=True)

    summary = {
        "algorithm": args.algorithm,
        "params": tracking[args.algorithm],
        "cut_threshold": config["perception"]["shots"]["cut_threshold"],
        "detected_cuts": sum(cuts.values()),
        "sequences_with_cuts": {k: v for k, v in cuts.items() if v},
        "track_ms_per_frame": 1000 * track_seconds / n_frames,
        "ball": {"min_score": min_ball_score, "precision": ball_hits / max(ball_selected, 1),
                 "recall": ball_hits / max(ball_gt, 1)},
    }
    result = evaluate_mot_sequences(gt_dir, run_dir / "people", metrics=["CLEAR", "HOTA", "Identity"])
    agg = result.aggregate
    summary["people"] = {
        "HOTA": agg.HOTA.HOTA, "DetA": agg.HOTA.DetA, "AssA": agg.HOTA.AssA,
        "IDF1": agg.Identity.IDF1, "MOTA": agg.CLEAR.MOTA, "IDSW": agg.CLEAR.IDSW,
    }
    result.save(run_dir / "people_metrics.json")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[{run_name}] {args.algorithm} {tracking[args.algorithm]}")
    print(f"cuts detected: {summary['detected_cuts']} | tracker {summary['track_ms_per_frame']:.2f} ms/frame")
    m = summary["people"]
    print(f"people HOTA {100*m['HOTA']:5.1f}  DetA {100*m['DetA']:5.1f}  AssA {100*m['AssA']:5.1f}  "
          f"IDF1 {100*m['IDF1']:5.1f}  MOTA {100*m['MOTA']:5.1f}  IDSW {m['IDSW']}")
    b = summary["ball"]
    print(f"ball   score > {b['min_score']}: precision {b['precision']:.3f}  recall {b['recall']:.3f}")


if __name__ == "__main__":
    main()
