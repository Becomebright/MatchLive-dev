"""One-off: export the jersey classifier checkpoint to ONNX, then check it end to end.

  checkpoints/<perception.jersey.checkpoint>  (+ dinov3-convnext-tiny/ config)
      -> checkpoints/<perception.jersey.model>
         input  pixel_values (N, 3, 224, 224) float32, dynamic N
         output probs        (N, 101) softmax

Checks, on real player crops (SoccerNet-GSR ground-truth boxes):
  1. the checkpoint loads strictly into the HF model;
  2. JerseyReader's numpy preprocessing vs the backbone's HF image processor (v2's path);
  3. ONNX on ORT-CUDA FP32 vs torch FP32;
  4. JerseyReader (TensorRT FP16 by default config) vs torch FP32.

    python scripts/export_jersey_onnx.py
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.dinov3_convnext import DINOv3ConvNextModel

from matchlive.common.config import REPO_ROOT, load_config
from matchlive.perception.jersey import NUM_LABELS, JerseyReader


class Classifier(nn.Module):
    """Backbone + linear head, with the softmax folded into the exported graph."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_sizes[-1], NUM_LABELS)

    def forward(self, pixel_values):
        return self.classifier(self.backbone(pixel_values=pixel_values).pooler_output).softmax(dim=-1)


def remap_state_dict(model: nn.Module, state: dict) -> dict:
    # The checkpoint stores backbone weights as `backbone.*`. Under transformers 5.x the
    # HF model nests `stages.*` under `backbone.model.` while `layer_norm` stays at
    # `backbone.layer_norm`, so prefix only the keys the model actually nests (v2
    # prefixed every key and no longer loads strictly).
    keys = set(model.state_dict())
    out = {}
    for k, v in state.items():
        nested = "backbone.model." + k.removeprefix("backbone.")
        out[nested if k not in keys and nested in keys else k] = v
    return out


def gsr_crops(n: int) -> list[np.ndarray]:
    # Full ground-truth player boxes from the first frames of a few GSR test sequences.
    crops = []
    for label_path in sorted((REPO_ROOT / "dataset/SoccerNetGS/test").glob("SNGS-*/Labels-GameState.json"))[:8]:
        labels = json.load(open(label_path))
        first = labels["images"][0]
        frame = cv2.imread(str(label_path.parent / "img1" / first["file_name"]))
        for a in labels["annotations"]:
            if a["image_id"] == first["image_id"] and a.get("category_id") in (1, 2):
                b = a["bbox_image"]
                crops.append(frame[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]])
    return crops[:n]


def compare(name: str, ref: np.ndarray, test: np.ndarray) -> None:
    agree = (ref.argmax(1) == test.argmax(1)).mean()
    print(f"{name:34s} top-1 agreement {100 * agree:5.1f}%  max |dprob| {np.abs(ref - test).max():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="export this safetensors checkpoint (e.g. training/runs/<name>/jersey.safetensors) "
                                         "instead of the deployed one, to --out, and skip the checks below")
    ap.add_argument("--out", help="ONNX file to write with --checkpoint")
    args = ap.parse_args()

    config = load_config()
    ckpt_dir = config["paths"]["checkpoints"]
    backbone_dir = ckpt_dir / "dinov3-convnext-tiny"
    ckpt_path = ckpt_dir / config["perception"]["jersey"]["checkpoint"]
    onnx_path = ckpt_dir / config["perception"]["jersey"]["model"]
    if args.checkpoint:      # an experiment's weights: never overwrite the deployed ONNX
        ckpt_path, onnx_path = REPO_ROOT / args.checkpoint, REPO_ROOT / args.out
        backbone_dir = REPO_ROOT / "training/jersey_train_pack/tmp/dinov3-convnext-tiny"   # config only

    # 1. Strict load. The backbone is built from its config alone: the jersey checkpoint
    # carries every backbone tensor, so DINOv3's own pretrained weights are never read.
    model = Classifier(DINOv3ConvNextModel(AutoConfig.from_pretrained(backbone_dir)))
    # Accept either a safetensors or a PyTorch pickle checkpoint (same key layout).
    state = load_file(ckpt_path) if ckpt_path.suffix == ".safetensors" else torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(remap_state_dict(model, state), strict=True)
    model.eval()
    print("checkpoint loaded strictly:", ckpt_path.name, len(state), "tensors")

    torch.onnx.export(
        model, (torch.zeros(1, 3, 224, 224),), str(onnx_path),
        input_names=["pixel_values"], output_names=["probs"],
        dynamic_axes={"pixel_values": {0: "batch"}, "probs": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print("exported:", onnx_path)
    if args.checkpoint:
        return

    crops = gsr_crops(256)
    model = model.cuda()

    def torch_probs(pixel_values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(pixel_values).cuda()).float().cpu().numpy()

    # 2. Preprocessing: HF image processor on RGB crops (v2) vs our numpy version.
    processor = AutoImageProcessor.from_pretrained(backbone_dir)
    hf_input = processor([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops], return_tensors="pt")["pixel_values"].numpy()
    our_input = JerseyReader._preprocess(crops)
    ref = torch_probs(hf_input)
    compare("preprocess: numpy vs HF processor", ref, torch_probs(our_input))

    # 3. Export parity on identical input.
    ort.preload_dlls()
    cuda = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
    compare("ONNX CUDA FP32 vs torch FP32", torch_probs(our_input), cuda.run(["probs"], {"pixel_values": our_input})[0])

    # 4. The deployed reader (TensorRT FP16 with the default config).
    reader = JerseyReader(config["perception"]["jersey"], ckpt_dir, config["paths"]["cache"])
    print("reader providers:", reader.providers)
    compare("JerseyReader vs torch FP32 (HF)", ref, reader.predict_batch(crops))


if __name__ == "__main__":
    main()
