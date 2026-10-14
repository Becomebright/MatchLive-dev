"""Per-box jersey-number label table from SoccerNet-GSR, for fine-tuning the jersey reader.

One row per ground-truth player / goalkeeper box. Nothing is dropped here: every row says
whether it is usable as a *positive* (box + number) and, if not, why, so the filtering
rules can be changed without redoing the pass over the labels.

What the GSR labels do and do not say (checked on train + valid):
  * the number is a property of the *track*: it never changes along a track, and a track is
    either labelled in every frame or in none. So an empty number means "annotators could
    not tell this player's number" — NOT "unreadable in this frame". Such boxes are marked
    `null_track` and must not be used as class-100 negatives without more evidence;
  * goalkeepers and referees carry no number at all (`goalkeeper`, `referee`);
  * the number is labelled in every frame of a labelled track, including frames where the
    player faces the camera or is occluded and the number cannot be seen. This table does
    NOT remove those: nothing here tells whether the number is visible in a given frame.

Columns: split, seq, game_id, frame (image file name), track_id, role, team, jersey (0-99,
empty when unlabelled), x, y, w, h (1080p box), h720 (box height after the 1080p -> 720p
downscale our pipeline works at), edge (box touches the image border), occ (share of the
box covered by a box closer to the camera — larger bottom edge — of a player, goalkeeper
or referee), reason ("ok" or the first rule that excludes the box as a positive).

Exclusion rules, in order:
  goalkeeper / referee  no number label exists
  null_track            the track has no number label
  small                 h720 < 40: the current reader covers 0.1% of such crops (docs/results.md §5)
  edge                  the box is cut by the image border
  occluded              occ > 0.5

    python scripts/prepare_jersey_labels.py            # train + valid -> dataset/SoccerNetGS/jersey_labels/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from matchlive.common.config import REPO_ROOT

GSR_DIR = REPO_ROOT / "dataset/SoccerNetGS"
OUT_DIR = GSR_DIR / "jersey_labels"
CATEGORIES = {1: "player", 2: "goalkeeper", 3: "referee"}
COLUMNS = ["split", "seq", "game_id", "frame", "track_id", "role", "team", "jersey",
           "x", "y", "w", "h", "h720", "edge", "occ", "reason"]


def occlusion(boxes: np.ndarray) -> np.ndarray:
    """Share of each (x, y, w, h) box covered by the single most-covering box that is closer
    to the camera (larger bottom edge). One box standing in front is the usual case; a union
    of several would need rasterising and changes little."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    iw = np.clip(np.minimum(x1[:, None], x1[None]) - np.maximum(x0[:, None], x0[None]), 0, None)
    ih = np.clip(np.minimum(y1[:, None], y1[None]) - np.maximum(y0[:, None], y0[None]), 0, None)
    covered = iw * ih / np.maximum(boxes[:, 2] * boxes[:, 3], 1e-6)[:, None]   # [i, j]: share of i covered by j
    in_front = y1[None] > y1[:, None]                                          # j is closer than i
    return np.where(in_front, covered, 0.0).max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "valid"],
                    help="test stays out of the default: it is the held-out benchmark")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    for split in args.splits:
        counts = Counter()
        with open(OUT_DIR / f"{split}.csv", "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(COLUMNS)
            for label_path in sorted((GSR_DIR / split).glob("SNGS-*/Labels-GameState.json")):
                labels = json.load(open(label_path))
                files = {im["image_id"]: im["file_name"] for im in labels["images"]}
                width = {im["image_id"]: (im["width"], im["height"]) for im in labels["images"]}
                by_image = defaultdict(list)
                for a in labels["annotations"]:
                    if a["category_id"] in CATEGORIES:
                        by_image[a["image_id"]].append(a)

                for image_id, anns in sorted(by_image.items()):
                    boxes = np.array([[a["bbox_image"][k] for k in "xywh"] for a in anns], dtype=float)
                    occ = occlusion(boxes)
                    W, H = width[image_id]
                    for a, (x, y, w, h), o in zip(anns, boxes, occ):
                        role = CATEGORIES[a["category_id"]]
                        jersey = a["attributes"].get("jersey")
                        edge = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
                        h720 = h * 2 / 3
                        if role != "player":
                            reason = role
                        elif jersey is None:
                            reason = "null_track"
                        elif h720 < 40:
                            reason = "small"
                        elif edge:
                            reason = "edge"
                        elif o > 0.5:
                            reason = "occluded"
                        else:
                            reason = "ok"
                        counts[reason] += 1
                        out.writerow([split, label_path.parent.name, labels["info"]["game_id"],
                                      files[image_id], a["track_id"], role, a["attributes"].get("team"),
                                      "" if jersey is None else int(jersey),
                                      int(x), int(y), int(w), int(h), round(h720, 1), int(edge), round(o, 3), reason])
        total = sum(counts.values())
        print(f"{split}: {total} boxes")
        for reason, n in counts.most_common():
            print(f"  {reason:12s} {n:8d}  {n / total:.1%}")


if __name__ == "__main__":
    main()
