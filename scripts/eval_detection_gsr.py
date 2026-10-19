"""Detection AP (COCO) of the configured detector on the SoccerNet-GSR test set (docs/results.md §1).

Scores the raw candidates that eval_tracking_gsr.py caches per detector model
(outputs/gsr_eval/<model>_raw.npy: every query with score > 0.01, 1080p pixel boxes) against
the GSR boxes: score > 0.01, at most 100 boxes per frame, four classes (player, goalkeeper,
referee, ball; GSR's "other" is not scored). Prints AP@[.5:.95], AP50, AP75 and the per-class AP.

    python scripts/eval_detection_gsr.py --config-set perception.detection.model=rfdetr-2xl-1019.onnx
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from eval_tracking_gsr import LABEL_PATHS, raw_candidates_path    # sibling script (scripts/ is on sys.path)
from matchlive.common.config import load_config

# GSR category id -> the detector's class id (ball, goalkeeper, referee, player).
GSR_TO_DETECTOR = {1: 3, 2: 1, 3: 2, 4: 0}
CLASS_NAMES = {3: "player", 1: "goalkeeper", 2: "referee", 0: "ball"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = ap.parse_args()
    config = load_config(overrides=args.config_set)
    raw = np.load(raw_candidates_path(config, config["paths"]["outputs"] / "gsr_eval"))

    # Ground truth in COCO format; the 49 sequences' image ids are unique across sequences.
    images, annotations = [], []
    for label_path in LABEL_PATHS:
        labels = json.load(open(label_path))
        images += [{"id": int(im["image_id"]), "width": im["width"], "height": im["height"]} for im in labels["images"]]
        for a in labels["annotations"]:
            if a.get("supercategory") == "object" and a["category_id"] in GSR_TO_DETECTOR:
                b = a["bbox_image"]
                annotations.append({"id": len(annotations) + 1, "image_id": int(a["image_id"]),
                                    "category_id": GSR_TO_DETECTOR[a["category_id"]],
                                    "bbox": [b["x"], b["y"], b["w"], b["h"]], "area": b["w"] * b["h"], "iscrowd": 0})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": annotations,
                  "categories": [{"id": i, "name": n} for i, n in CLASS_NAMES.items()]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    # Detections: the 100 best-scoring boxes of each frame.
    by_image = defaultdict(list)
    for row in raw:
        by_image[int(row[0])].append(row)
    results = []
    for image_id, rows in by_image.items():
        for r in sorted(rows, key=lambda r: -r[5])[:100]:
            results.append({"image_id": image_id, "category_id": int(r[6]), "bbox": [float(v) for v in r[1:5]], "score": float(r[5])})
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 100]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision[iou, recall, class, area, maxDets]; -1 marks classes without ground truth.
    def ap(iou=None, cls=None):
        p = ev.eval["precision"][:, :, :, 0, 2] if iou is None else ev.eval["precision"][iou:iou + 1, :, :, 0, 2]
        p = p if cls is None else p[:, :, [ev.params.catIds.index(cls)]]
        return 100 * float(np.mean(p[p > -1]))

    print(f"model {config['perception']['detection']['model']}: {len(raw)} candidates, {len(annotations)} GT boxes")
    print(f"AP {ap():.1f}  AP50 {ap(0):.1f}  AP75 {ap(5):.1f}  " + "  ".join(f"{n} {ap(cls=i):.1f}" for i, n in CLASS_NAMES.items()))


if __name__ == "__main__":
    main()
