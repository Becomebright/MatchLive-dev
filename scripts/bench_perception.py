"""Per-frame timing of the whole perception chain on a video: decode, detection, shot cuts,
tracking, jersey reading and kit colour (MatchRunner.perceive, the timer behind results §8),
for one detector checkpoint. Nothing is cached; run it on an otherwise idle machine.

    python scripts/bench_perception.py --video <file.mp4> [--model rfdetr-2xl.onnx] [--frames 6000]

Prints the mean ms/frame of each stage, the detector's and the whole chain's frames per second,
and percentiles of each frame's total (real time at 25 fps: every frame within 40 ms).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from matchlive.common.config import load_config
from matchlive.runtime.runner import MatchRunner


def frames(video: str, n: int | None):
    """(frame_idx, BGR frame) for the first n frames of the video (all of them when n is None)."""
    cap = cv2.VideoCapture(video)
    i = 0
    while n is None or i < n:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", help="detector checkpoint under paths.checkpoints (default: the config's)")
    ap.add_argument("--frames", type=int, help="frames to time (default: the whole video)")
    ap.add_argument("--warmup", type=int, default=100, help="frames run first and not timed")
    args = ap.parse_args()

    overrides = []
    if args.model:
        # Each checkpoint gets its own TensorRT engine cache, built on the first run (minutes).
        overrides = [f"perception.detection.model={args.model}",
                     f"perception.detection.trt_cache=trt/{Path(args.model).stem}_fp16"]
    config = load_config(overrides=overrides)
    t = time.perf_counter()
    runner = MatchRunner(config)
    detector = runner.detector
    print(f"{config['perception']['detection']['model']}: {detector.providers[0]}, input "
          f"{detector.input_width}x{detector.input_height}, sessions ready in {time.perf_counter() - t:.0f}s")

    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    # The first frames pay for engine loading and first allocations.
    runner.perceive(frames(args.video, args.warmup), fps)
    arrays, timing = runner.perceive(frames(args.video, args.frames), fps)

    n = len(arrays["shot_id"])
    ms = timing["ms_per_frame"]
    total = sum(ms.values())
    print(json.dumps({"frames": n, "tracks_per_frame": round(len(arrays["track"]) / n, 1),
                      "ms_per_frame": {k: round(v, 2) for k, v in ms.items()}, "total_ms": round(total, 2),
                      "fps_detect": round(1000 / ms["detect"], 1), "fps_chain": round(1000 / total, 1),
                      "frame_total_ms": {k: round(v, 3) for k, v in timing["total_ms"].items()}}))


if __name__ == "__main__":
    main()
