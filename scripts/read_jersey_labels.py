"""Read every usable GSR box with the current jersey reader, to filter the labels for fine-tuning.

GSR labels a track's number in every frame, including frames where the player faces the
camera or is occluded and the number cannot be seen (see prepare_jersey_labels.py), so a
label alone does not tell which frames are good positives. Here the current reader
(`-20_matches`) scores each `ok` box of dataset/SoccerNetGS/jersey_labels/<split>.csv, and
a box counts as a readable positive when the reader names the track's number itself. That
keeps clean positives and drops frames the number is not visible in; it cannot find
readable frames the reader misses, so it mostly adapts the reader to GSR rather than
raising its coverage.

Crops are made as in the pipeline and in eval_jersey_gsr.py: frame and box scaled from
1080p to 720p, then `crop_upper_body`. Writes <split>_reads.csv next to the label table
(split, seq, frame, track_id, jersey, h720, top1, p_top1, p_label, p_unreadable), with the
probabilities kept so the agreement threshold can be chosen afterwards without re-reading.
Prints, per box-height bucket, how many boxes the reader reads as the track's number.

    python scripts/read_jersey_labels.py --stride 5
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from matchlive.common.config import REPO_ROOT, load_config
from matchlive.perception.jersey import UNREADABLE, JerseyReader, crop_upper_body

GSR_DIR = REPO_ROOT / "dataset/SoccerNetGS"
LABEL_DIR = GSR_DIR / "jersey_labels"
HEIGHT_BUCKETS = [40, 60, 80, 100, 150, 1000]
THRESHOLDS = [0.0, 0.3, 0.5, 0.9]       # p_label needed, besides top1 == label


def load_frame(path):
    frame = cv2.imread(str(path))
    return cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--stride", type=int, default=5, help="use every n-th frame (25 fps: neighbours look alike)")
    ap.add_argument("--reason", default="ok",
                    help="which boxes of the label table to read: `ok` (usable positives) or `null_track` "
                         "(tracks without a number; there the summary counts confident reads of any number)")
    args = ap.parse_args()

    config = load_config()
    reader = JerseyReader(config["perception"]["jersey"], config["paths"]["checkpoints"], config["paths"]["cache"])
    print("providers:", reader.providers)

    for split in args.splits:
        # usable boxes of the sampled frames, grouped by sequence and frame
        rows = defaultdict(lambda: defaultdict(list))
        with open(LABEL_DIR / f"{split}.csv") as f:
            for r in csv.DictReader(f):
                if r["reason"] == args.reason and (int(r["frame"][:-4]) - 1) % args.stride == 0:
                    rows[r["seq"]][r["frame"]].append(r)

        out_rows = []
        for seq, frames in sorted(rows.items()):
            names = sorted(frames)
            with ThreadPoolExecutor(16) as pool:       # decoding 1080p JPEGs is the slow part
                images = pool.map(lambda n: load_frame(GSR_DIR / split / seq / "img1" / n), names)
                crops, meta = [], []
                for name, frame in zip(names, images):
                    for r in frames[name]:
                        box = (float(r["x"]) * 2 / 3, float(r["y"]) * 2 / 3, float(r["w"]) * 2 / 3, float(r["h"]) * 2 / 3)
                        crop = crop_upper_body(frame, box)
                        if crop is not None:
                            crops.append(crop)
                            meta.append(r)
            probs = reader.predict_batch(crops)
            for r, p in zip(meta, probs):
                top = int(p.argmax())
                out_rows.append([split, seq, r["frame"], r["track_id"], r["jersey"], r["h720"], top,
                                 round(float(p[top]), 4), round(float(p[int(r["jersey"])]) if r["jersey"] else 0.0, 4), round(float(p[UNREADABLE]), 4)])
            print(f"{split} {seq}: {len(meta)} crops", flush=True)

        suffix = "" if args.reason == "ok" else f"_{args.reason}"
        with open(LABEL_DIR / f"{split}_reads{suffix}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "seq", "frame", "track_id", "jersey", "h720", "top1", "p_top1", "p_label", "p_unreadable"])
            w.writerows(out_rows)

        # how many boxes the reader reads as the track's number, by box height
        a = np.array([[float(r[5]), float(r[4] or -1), r[6], r[7] if args.reason != "ok" else r[8]] for r in out_rows])
        h, label, top1, p_label = a.T      # p_label: probability of the track's number (`ok`) / of the top read (other reasons)
        # `ok`: the reader names the track's number; otherwise (no label): it reads any number at all
        agree = (top1 == label) if args.reason == "ok" else (top1 != UNREADABLE)
        print(f"\n{split}: {len(a)} {args.reason} boxes read; share where the reader "
              + ("names the track's number (top1 == label):" if args.reason == "ok" else "reads some number (top1 != unreadable):"))
        print(f"{'height(720p)':>13s} {'boxes':>8s} " + " ".join(f"p>={t:<4}" for t in THRESHOLDS))
        for lo, hi in zip(HEIGHT_BUCKETS[:-1], HEIGHT_BUCKETS[1:]):
            m = (h >= lo) & (h < hi)
            print(f"{f'{lo}-{hi}':>13s} {m.sum():8d} " + " ".join(f"{(agree & (p_label >= t) & m).sum() / max(m.sum(), 1):7.3f}" for t in THRESHOLDS))
        print(f"{'all':>13s} {len(a):8d} " + " ".join(f"{(agree & (p_label >= t)).mean():7.3f}" for t in THRESHOLDS))


if __name__ == "__main__":
    main()
