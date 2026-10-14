"""Single-crop jersey-number accuracy on SoccerNet-GSR, by box height.

Crops ground-truth player boxes (goalkeepers carry no number label in GSR) from every
`--stride`-th frame of each test sequence, runs JerseyReader, and reports per box-height
bucket:
  coverage  = crops the model reads a number on (top label not "unreadable") / all crops
  precision = correct reads / reads
  recall    = correct reads / all crops
GSR labels a track's number even in frames where it cannot be seen, so recall is capped
by visibility; precision and coverage are what matter for vote-based identity.

Frames are first downscaled from 1080p to 720p so crop resolution matches our match
videos, then cropped exactly as the pipeline does (`crop_upper_body`); box heights are in
720p pixels. The last row repeats "all" counting only reads with probability >= 0.9.

    python scripts/eval_jersey_gsr.py --stride 10
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from matchlive.common.config import REPO_ROOT, load_config
from matchlive.perception.jersey import UNREADABLE, JerseyReader, crop_upper_body, label_to_number

GSR_DIR = REPO_ROOT / "dataset/SoccerNetGS/test"
HEIGHT_BUCKETS = [0, 40, 60, 80, 100, 150, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10, help="use every n-th frame")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a config setting, e.g. perception.jersey.model=/path/to/jersey.onnx")
    args = ap.parse_args()

    config = load_config(overrides=args.set)
    reader = JerseyReader(config["perception"]["jersey"], config["paths"]["checkpoints"], config["paths"]["cache"])
    print("providers:", reader.providers)

    crops, truth, heights = [], [], []
    for label_path in sorted(GSR_DIR.glob("SNGS-*/Labels-GameState.json")):
        labels = json.load(open(label_path))
        images = sorted(labels["images"], key=lambda im: im["file_name"])[::args.stride]
        wanted = {im["image_id"]: im["file_name"] for im in images}
        boxes: dict[str, list] = {}
        for a in labels["annotations"]:
            if a["image_id"] in wanted and a.get("category_id") == 1 and a["attributes"].get("jersey"):
                boxes.setdefault(a["image_id"], []).append((a["bbox_image"], a["attributes"]["jersey"]))
        for image_id, items in boxes.items():
            frame = cv2.imread(str(label_path.parent / "img1" / wanted[image_id]))
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            for b, jersey in items:
                ltwh = (b["x"] * 2 / 3, b["y"] * 2 / 3, b["w"] * 2 / 3, b["h"] * 2 / 3)   # 1080p -> 720p
                crop = crop_upper_body(frame, ltwh)
                if crop is not None:
                    crops.append(crop)
                    truth.append(jersey)
                    heights.append(ltwh[3])
    truth, heights = np.asarray(truth), np.asarray(heights)
    print(f"{len(truth)} labelled player crops (stride {args.stride})")

    t0 = time.perf_counter()
    probs = reader.predict_batch(crops)
    ms = 1000 * (time.perf_counter() - t0) / len(crops)
    pred = probs.argmax(1)
    read = pred != UNREADABLE
    correct = read & (np.asarray([label_to_number(int(l)) for l in pred]) == truth)
    print(f"{ms:.2f} ms/crop incl. preprocessing (batch {reader.max_batch})")
    print(f"{'height(720p)':>13s} {'crops':>7s} {'coverage':>9s} {'precision':>10s} {'recall':>7s}")
    for lo, hi in zip(HEIGHT_BUCKETS[:-1], HEIGHT_BUCKETS[1:]):
        m = (heights >= lo) & (heights < hi)
        if m.sum():
            print(f"{f'{lo}-{hi}':>13s} {m.sum():7d} {read[m].mean():9.3f} "
                  f"{correct[m].sum() / max(read[m].sum(), 1):10.3f} {correct[m].mean():7.3f}")
    print(f"{'all':>13s} {len(truth):7d} {read.mean():9.3f} {correct.sum() / max(read.sum(), 1):10.3f} {correct.mean():7.3f}")
    sure = read & (probs.max(1) >= 0.9)
    print(f"{'all, p>=0.9':>13s} {len(truth):7d} {sure.mean():9.3f} "
          f"{(correct & sure).sum() / max(sure.sum(), 1):10.3f} {(correct & sure).mean():7.3f}")


if __name__ == "__main__":
    main()
