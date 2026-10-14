"""Accuracy of a jersey-reader ONNX model on the two validation sets used for training it.

  training/val_20_matches.parquet   5000 rows sampled from the 20 EPL matches (VLM labels)
  training/gsr_crops/val.parquet    5000 rows from the GSR valid split (GSR labels, selected with the
                                    `-20_matches` reader — see build_jersey_training_data.py, so
                                    this set favours that model; use it to compare training runs)

Crops are the stored whole-box PNGs, stretched to 224x224 exactly as in the pipeline.
Per set: accuracy over the 101 classes, and for the rows with a number: how many the model reads
correctly (recall) and how many of its reads are right (precision); for the unreadable rows: how
many the model wrongly reads a number on (false reads). Reads at p >= 0.9 are what identity
voting uses, so they are listed separately.

    python scripts/eval_jersey_val.py                           # the deployed checkpoint
    python scripts/eval_jersey_val.py --model path/to/new.onnx
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

from matchlive.common.config import REPO_ROOT, load_config
from matchlive.perception.jersey import UNREADABLE, JerseyReader

SETS = {"20 matches": REPO_ROOT / "training/val_20_matches.parquet",
        "GSR valid": REPO_ROOT / "training/gsr_crops/val.parquet"}


def label_of(value: str | None) -> int:
    # The same mapping as train.py: no number (null) -> unreadable.
    return int(value) if value is not None and value.isdigit() and 0 <= int(value) <= 99 else UNREADABLE


def predict(session, path, batch=256):
    labels, probs = [], []
    for rows in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=["image", "jersey_number"]):
        rows = rows.to_pylist()
        crops = [cv2.imdecode(np.frombuffer(r["image"], np.uint8), cv2.IMREAD_COLOR) for r in rows]
        probs.append(session.run(["probs"], {"pixel_values": JerseyReader._preprocess(crops)})[0])
        labels += [label_of(r["jersey_number"]) for r in rows]
    return np.asarray(labels), np.concatenate(probs)


def report(name, labels, probs):
    pred, conf = probs.argmax(1), probs.max(1)
    num = labels != UNREADABLE
    out = {"accuracy": (pred == labels).mean()}
    for tag, sure in (("", np.ones_like(conf, bool)), (" p>=0.9", conf >= 0.9)):
        read = (pred != UNREADABLE) & sure
        out[f"number recall{tag}"] = ((pred == labels) & read & num).sum() / num.sum()
        out[f"number precision{tag}"] = ((pred == labels) & read).sum() / max(read.sum(), 1)
        out[f"false reads{tag}"] = (read & ~num).sum() / (~num).sum()
    print(f"{name}: {len(labels)} rows, {num.sum()} with a number")
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="ONNX file; default: the deployed jersey checkpoint")
    args = ap.parse_args()
    config = load_config()
    model = args.model or str(config["paths"]["checkpoints"] / config["perception"]["jersey"]["model"])
    session = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(model, session.get_providers())
    for name, path in SETS.items():
        report(name, *predict(session, path))


if __name__ == "__main__":
    main()
