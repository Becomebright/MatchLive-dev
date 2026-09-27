"""Phase-0 acceptance check: v2 vs matchlive perception, side by side, on match segments.

Left: v2's chain as it was (RF-DETR-2XL, per-role ByteTrack without shot resets, v2
identity: kit colour vote + a jersey read every 15 frames per track, full-roster hard
filter). Right: ours, streamed from the half's raw perception cache through the
IdentityResolver. Both use the same jersey checkpoint (config perception.jersey), so the
comparison is of the pipelines, not the classifiers; v2's detector runs on TensorRT FP16
and its jersey reader on CUDA, only for speed.

Each segment starts both chains `--warmup` seconds early (identity votes start cold for
both) and renders only the segment. Writes into outputs/compare_v2/:
  <segment>.mp4          v2 | matchlive, half-size, side by side
  frames_<k>.jpg         the acceptance sample: 60 frames spread evenly over the segments,
                         5 side-by-side pairs per image
and prints per segment: track boxes per frame and the share given a jersey number.

    python scripts/compare_v2.py --match "2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool" --half 1
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

from export_jersey_onnx import remap_state_dict   # sibling script (scripts/ is on sys.path)
from matchlive.common.cache import iter_frames, load_perception
from matchlive.common.config import load_config
from matchlive.common.match import Match
from matchlive.perception.detection import session_providers, to_detections
from matchlive.perception.identity import IdentityResolver, kit_bgr, kit_lab, roster_index
from matchlive.perception.tracking import select_ball
from matchlive.perception.visualization import draw_tracks
from matchlive.runtime.runner import MatchRunner

# (name, start second, seconds): a wide shot with a fast pan, a stretch with cuts to
# close-ups and the bench, and close-ups with a player walking past the camera.
SEGMENTS = [("wide_pan", 685, 20), ("cuts", 660, 20), ("closeups", 695, 20)]


def load_v2(config, checkpoints):
    """v2's detector, tracker class, identity resolver and drawing, imported from its repo."""
    sys.path.insert(0, str(config["paths"]["v2"]))
    import onnxruntime as ort
    import perception.jersey as v2_jersey
    from perception.detection import RFDETRDetector
    from perception.identity import IdentityResolver as V2Resolver
    from perception.tracking import MultiRoleTracker as V2Tracker
    from perception.visualization import draw_tracks as v2_draw

    # v2's key remap prefixes every backbone key and no longer loads strictly under
    # transformers 5.x (CLAUDE.md porting notes); swap in the fixed one.
    v2_jersey._remap_state_dict = remap_state_dict
    detector = RFDETRDetector(str(checkpoints / "rfdetr-2xl.onnx"), confidence=0.15)
    detector.session = ort.InferenceSession(
        str(checkpoints / "rfdetr-2xl.onnx"),
        providers=session_providers({"providers": ["tensorrt", "cuda"], "fp16": True,
                                     "trt_cache": "trt/rfdetr-2xl_fp16"}, config["paths"]["cache"]))
    reader = v2_jersey.JerseyReader(str(checkpoints / "dinov3-convnext-tiny"),
                                    str(checkpoints / config["perception"]["jersey"]["checkpoint"]), device="cuda")
    return detector, V2Tracker, V2Resolver, reader, v2_draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=10.0)
    args = ap.parse_args()

    config = load_config()
    checkpoints = config["paths"]["checkpoints"]
    match = Match(config, args.match)
    metadata = match.metadata
    arrays, meta = load_perception(MatchRunner(config).run_perception(match, args.half))
    ours = {f: rest for f, *rest in iter_frames(arrays)}
    roster = roster_index(metadata["players"])
    team_bgr = kit_bgr(metadata["jersey_colors"])
    labels = {p["hash"]: (str(p["jersey_number"]), p["short_name"]) for p in metadata["players"]}
    fps = meta["fps"]

    detector, V2Tracker, V2Resolver, reader, v2_draw = load_v2(config, checkpoints)
    out_dir = config["paths"]["outputs"] / "compare_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for name, start, seconds in SEGMENTS:
        first, begin, end = (int(round(s * fps)) for s in (start - args.warmup, start, start + seconds))
        # v2 state and ours both start cold at `first`.
        v2_tracker, v2_resolver = V2Tracker(frame_rate=fps), V2Resolver(metadata, reader)
        resolver = IdentityResolver(config["perception"]["identity"], kit_lab(metadata["jersey_colors"]))
        cap = cv2.VideoCapture(str(match.video_path(args.half)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        writer = cv2.VideoWriter(str(out_dir / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 360))
        pick = set(np.linspace(begin, end - 1, 20).astype(int))
        stats = {"v2": [0, 0], "ours": [0, 0]}                  # track boxes, boxes with a number
        for frame_idx in range(first, end):
            ok, frame = cap.read()
            if not ok:
                break
            v2_tracks = v2_resolver.resolve(frame_idx, frame, v2_tracker.update(detector.predict(frame)))
            tracks, jersey, color, det = ours.get(frame_idx, ([], [], [], arrays["det"][:0, 1:]))
            tracks = resolver.update(frame_idx, tracks, jersey, color, roster, {})
            if frame_idx < begin:
                continue
            stats["v2"][0] += len(v2_tracks)
            stats["v2"][1] += sum(t.jersey_number is not None for t in v2_tracks)
            stats["ours"][0] += len(tracks)
            stats["ours"][1] += sum(t.player_id is not None for t in tracks)
            dets = to_detections(det[:, :4], det[:, 4], det[:, 5], 0.0)
            ball = select_ball(dets, config["perception"]["ball"]["min_score"])
            left = v2_draw(frame, v2_tracks, v2_resolver.team_bgr, v2_resolver.rosters)
            right = draw_tracks(frame, tracks, ball, team_bgr, labels, dets)
            pair = np.hstack([cv2.resize(left, (640, 360)), cv2.resize(right, (640, 360))])
            second = frame_idx / fps
            cv2.putText(pair, f"v2   {int(second // 60)}:{second % 60:05.2f}", (8, 24), 0, 0.7, (0, 255, 255), 2)
            cv2.putText(pair, f"matchlive   [{name}]", (648, 24), 0, 0.7, (0, 255, 255), 2)
            writer.write(pair)
            if frame_idx in pick:
                samples.append(pair)
        writer.release()
        print(f"{name}: " + "  ".join(
            f"{k} {n / ((end - begin) or 1):.1f} boxes/frame, {num / max(n, 1):.2f} with a number"
            for k, (n, num) in stats.items()), flush=True)

    for k in range(0, len(samples), 5):
        cv2.imwrite(str(out_dir / f"frames_{k // 5 + 1:02d}.jpg"), np.vstack(samples[k:k + 5]))
    print(f"{len(samples)} sampled frames -> {out_dir}")


if __name__ == "__main__":
    main()
