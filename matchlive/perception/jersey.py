"""Jersey-number classifier (DINOv3 ConvNeXt-tiny backbone + linear head), ported from v2.

Labels 0-99 are jersey numbers; label 100 means "unreadable".

Changed from v2: runs as an ONNX export (scripts/export_jersey_onnx.py) on ONNX Runtime
with TensorRT first, like the detector, so inference needs neither torch nor
transformers; preprocessing is done here in numpy; predict_batch returns the full
101-way probabilities instead of only the top label (DESIGN.md §6.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import onnxruntime as ort

from matchlive.perception.detection import session_providers

NUM_LABELS = 101
UNREADABLE = 100

# The backbone's ImageNet normalization, folded into one multiply-subtract per channel
# on 0-255 input: (x / 255 - mean) / std == x * scale - offset.
_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_SCALE = 1.0 / (255.0 * _STD)
_OFFSET = _MEAN / _STD


def label_to_number(label: int) -> str | None:
    # Labels 0-99 are the jersey numbers; label 100 ("unreadable") maps to None.
    return str(label) if 0 <= label <= 99 else None


def crop_upper_body(frame: np.ndarray, bbox_ltwh: tuple[float, float, float, float]) -> np.ndarray | None:
    # Take only the top 75% of the box (v2's crop): the number lives on the torso, while
    # the lower part is mostly shorts, legs and grass. On SoccerNet-GSR this reads about
    # as well as the full box (precision 0.747 vs 0.737 at the same coverage) and its
    # smaller crop preprocesses faster.
    # Clamp to frame bounds and reject crops too small to be meaningful.
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox_ltwh
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + bw)), min(h, int(y + bh * 0.75))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


class JerseyReader:
    def __init__(self, config: dict[str, Any], checkpoints_dir: Path, cache_dir: Path):
        # `config` is the perception.jersey section of the YAML config.
        self.max_batch = int(config["max_batch"])
        # The export has a dynamic batch dimension; TensorRT needs its range up front and
        # builds one engine for it, tuned for `max_batch`.
        one, most = "pixel_values:1x3x224x224", f"pixel_values:{self.max_batch}x3x224x224"
        self.session = ort.InferenceSession(
            str(checkpoints_dir / config["model"]),
            providers=session_providers(config, checkpoints_dir / config["model"], cache_dir, {
                "trt_profile_min_shapes": one,
                "trt_profile_opt_shapes": most,
                "trt_profile_max_shapes": most,
            }),
        )
        self.providers = self.session.get_providers()

    def predict_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Class probabilities (N, 101) for BGR crops, in chunks of at most `max_batch`."""
        if not crops:
            return np.zeros((0, NUM_LABELS), dtype=np.float32)
        out = []
        for i in range(0, len(crops), self.max_batch):
            batch = self._preprocess(crops[i:i + self.max_batch])
            out.append(self.session.run(["probs"], {"pixel_values": batch})[0])
        return np.concatenate(out)

    @staticmethod
    def _preprocess(crops: Sequence[np.ndarray]) -> np.ndarray:
        # Same as the backbone's image processor: stretch to 224x224 (no aspect-ratio
        # padding), bilinear, RGB, rescale and normalize — written straight into one
        # contiguous NCHW buffer.
        batch = np.empty((len(crops), 3, 224, 224), dtype=np.float32)
        for n, crop in enumerate(crops):
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            for c in range(3):
                np.multiply(resized[:, :, 2 - c], _SCALE[c], out=batch[n, c])
                batch[n, c] -= _OFFSET[c]
        return batch
