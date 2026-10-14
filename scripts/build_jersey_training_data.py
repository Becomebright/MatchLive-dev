"""Crops + validation sets for training the jersey reader on 20_matches mixed with SoccerNet-GSR.

Training starts from the DINOv3 backbone (not from the `-20_matches` checkpoint), on the 20
EPL matches in training/jersey_train_pack/ plus the GSR crops built here. Output, all in
the schema of the 20-match parquets (image_id, image = PNG bytes, game_num, half,
source_path, jersey_number = string, null = unreadable):

  training/gsr_crops/train.parquet    GSR train split
  training/gsr_crops/val.parquet      5000 boxes sampled from the GSR valid split
  training/val_20_matches.parquet     5000 rows sampled (seed 42) from the 20-match data
  training/val_20_matches_ids.txt     their image_ids — to be left out of training

A GSR box becomes
  a positive  (jersey_number = the track's number) if the current reader (`-20_matches`)
              names that number on it with p >= --min-prob (<split>_reads.csv, from
              read_jersey_labels.py): GSR labels the number in every frame of a track,
              including frames where it cannot be seen, and the reader tells which;
  a negative  (null) if it belongs to a track without a number label (<split>_reads_null_track.csv):
              the reader reads a number on 0.3-0.5% of such boxes against 12-14% on labelled
              tracks, so those tracks are unreadable throughout.
Boxes of labelled tracks the reader misses are left out: unreadable, or merely hard — unknown.

Both GSR sets use the sampled frames of the reads files only (every 5th), and the GSR valid
split provides validation only, so no match is in both training and validation.
Since the positives and negatives are chosen with the `-20_matches` reader, GSR validation
accuracy measures agreement with that reader on clear cases, not the benefit of the new data;
the benefit is measured on GSR test (eval_jersey_gsr.py).

Crops follow the 20-match data: the whole detector-sized box on the 720p frame, not resized,
PNG. GSR frames are first scaled from 1080p to 720p.

    python scripts/build_jersey_training_data.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from matchlive.common.config import REPO_ROOT

GSR_DIR = REPO_ROOT / "dataset/SoccerNetGS"
LABEL_DIR = GSR_DIR / "jersey_labels"
PACK_DIR = REPO_ROOT / "training/jersey_train_pack/tmp/data_20_matches"
OUT_DIR = REPO_ROOT / "training"
COLUMNS = ["image_id", "image", "game_num", "half", "source_path", "jersey_number"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def gsr_boxes(split: str, min_prob: float) -> list[dict]:
    """Positives and negatives of one GSR split, each with its 1080p box and sequence."""
    reads = [r for r in read_csv(LABEL_DIR / f"{split}_reads.csv")
             if r["top1"] == r["jersey"] and float(r["p_label"]) >= min_prob]
    nulls = read_csv(LABEL_DIR / f"{split}_reads_null_track.csv")
    wanted = {(r["seq"], r["frame"], r["track_id"]): r["jersey"] or None for r in reads + nulls}
    out = []
    for r in read_csv(LABEL_DIR / f"{split}.csv"):
        key = (r["seq"], r["frame"], r["track_id"])
        if key in wanted:
            out.append({"split": split, "seq": r["seq"], "frame": r["frame"], "track_id": r["track_id"],
                        "game_id": int(r["game_id"]), "jersey": wanted[key],
                        "box": [float(r[k]) for k in "xywh"]})
    return out


def crop_png(frame720: np.ndarray, box1080: list[float]) -> bytes | None:
    """PNG of the whole box on the 720p frame (box scaled 2/3 from 1080p, clamped)."""
    h, w = frame720.shape[:2]
    x, y, bw, bh = (v * 2 / 3 for v in box1080)
    x1, y1, x2, y2 = max(0, int(x)), max(0, int(y)), min(w, int(np.ceil(x + bw))), min(h, int(np.ceil(y + bh)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.imencode(".png", frame720[y1:y2, x1:x2])[1].tobytes()


def gsr_rows(boxes: list[dict]) -> list[dict]:
    by_frame = defaultdict(list)
    for b in boxes:
        by_frame[(b["split"], b["seq"], b["frame"])].append(b)

    def load(key):
        split, seq, frame = key
        img = cv2.imread(str(GSR_DIR / split / seq / "img1" / frame))
        return cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

    rows, keys = [], sorted(by_frame)
    with ThreadPoolExecutor(32) as pool:       # decoding 1080p JPEGs and encoding PNGs dominate
        for key, frame in zip(keys, pool.map(load, keys)):
            for b in by_frame[key]:
                png = crop_png(frame, b["box"])
                if png is not None:
                    rows.append({
                        "image_id": f"gsr_{b['split']}_{b['seq']}_{b['frame'][:-4]}_t{b['track_id']}.png",
                        "image": png, "game_num": 100 + b["game_id"], "half": 0,
                        "source_path": f"dataset/SoccerNetGS/{b['split']}/{b['seq']}/img1/{b['frame']}",
                        "jersey_number": b["jersey"]})
    return rows


def write(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({c: [r[c] for r in rows] for c in COLUMNS},
                     schema=pa.schema([("image_id", pa.string()), ("image", pa.binary()), ("game_num", pa.int32()),
                                       ("half", pa.int32()), ("source_path", pa.string()), ("jersey_number", pa.string())]))
    pq.write_table(table, path, row_group_size=10000)
    labelled = sum(r["jersey_number"] is not None for r in rows)
    print(f"{path.relative_to(REPO_ROOT)}: {len(rows)} rows, {labelled} with a number, {len(rows) - labelled} unreadable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-prob", type=float, default=0.3, help="probability of the track's number for a GSR positive")
    ap.add_argument("--val-size", type=int, default=5000, help="rows of each validation set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # GSR training set: every positive / negative of the train split
    write(gsr_rows(gsr_boxes("train", args.min_prob)), OUT_DIR / "gsr_crops/train.parquet")

    # GSR validation set: a random sample of the valid split's positives and negatives
    pool = gsr_boxes("valid", args.min_prob)
    pick = rng.choice(len(pool), size=args.val_size, replace=False)
    write(gsr_rows([pool[i] for i in sorted(pick)]), OUT_DIR / "gsr_crops/val.parquet")

    # 20-match validation set: a fixed random sample of rows, left out of training by id
    files = sorted(PACK_DIR.glob("jersey_numbers*.parquet"))
    ids = [pq.read_table(f, columns=["image_id"]).column(0).to_pylist() for f in files]
    flat = [(i, j) for i, file_ids in enumerate(ids) for j in range(len(file_ids))]
    chosen = defaultdict(list)
    for k in rng.choice(len(flat), size=args.val_size, replace=False):
        chosen[flat[k][0]].append(flat[k][1])
    rows = []
    for i, js in sorted(chosen.items()):
        table = pq.read_table(files[i]).take(sorted(js))
        rows += table.to_pylist()
    write(rows, OUT_DIR / "val_20_matches.parquet")
    (OUT_DIR / "val_20_matches_ids.txt").write_text("\n".join(r["image_id"] for r in rows) + "\n")


if __name__ == "__main__":
    main()
