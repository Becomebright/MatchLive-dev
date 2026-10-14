"""Train the jersey-number reader from the DINOv3-ConvNeXt-tiny backbone on 20_matches + SoccerNet-GSR.

Same model and recipe as the `-20_matches` checkpoint (training/jersey_train_pack/tmp/train.py:
backbone + one linear layer over 101 classes, 0-99 = number, 100 = unreadable; cross entropy;
AdamW, lr 5e-5 linearly decayed, 5 epochs; crops stretched to 224x224; no augmentation), but

  * data = the 20 EPL matches (VLM labels) without the 5000 rows held out for validation
           + the GSR train crops, repeated `--gsr-repeat` times (build_jersey_training_data.py);
  * validation = those 5000 held-out 20-match rows + 5000 rows from the GSR valid split,
    evaluated every epoch; the checkpoint with the best mean accuracy of the two is kept.

Launch on several GPUs; this machine's NCCL_NET=gIB plugin does not initialise, and
single-node training needs no network plugin, so override it:

    NCCL_NET=Socket torchrun --nproc_per_node=4 scripts/train_jersey.py --name fp32_gsr2

Writes training/runs/<name>/ with Trainer checkpoints, tensorboard logs and `jersey.safetensors`
(the best checkpoint's weights, keys backbone.* / classifier.*).
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

from matchlive.common.config import REPO_ROOT

# Arrow copies of the parquet files go under the (git-ignored) cache/, not the home directory.
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "cache/hf_datasets"))

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn
from torchvision import transforms
from transformers import AutoImageProcessor, Trainer, TrainingArguments
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.dinov3_convnext import DINOv3ConvNextModel

TRAINING = REPO_ROOT / "training"
UNREADABLE = 100


class JerseyClassifier(nn.Module):
    def __init__(self, backbone_path: Path):
        super().__init__()
        self.backbone = DINOv3ConvNextModel.from_pretrained(backbone_path)
        self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], 101)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values, labels=None):
        logits = self.classifier(self.backbone(pixel_values=pixel_values).pooler_output)
        return ImageClassifierOutput(loss=self.loss_fn(logits, labels) if labels is not None else None, logits=logits)


def label_of(value: str | None) -> int:
    # A number 0-99 is its own label; anything else (null = no number readable) is 100.
    return int(value) if value is not None and value.isdigit() and 0 <= int(value) <= 99 else UNREADABLE


def with_pixels(dataset, processor, augment=False, size=(224, 224), pad_square=False):
    # Decode the PNG bytes and build the labels on the fly, so the 9 GB of images are never
    # rewritten into a second Arrow file.
    # Augmentation (training only): colour jitter and a small shift / scale / rotation. No flip:
    # a mirrored digit is a different digit.
    jitter = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ]) if augment else None

    def transform(batch):
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch["image"]]
        if jitter is not None:
            images = [jitter(img) for img in images]
        if pad_square:      # keep the aspect ratio: pad the crop to a square with the ImageNet mean colour
            padded = []
            for img in images:
                side = max(img.size)
                canvas = Image.new("RGB", (side, side), (124, 116, 104))
                canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
                padded.append(canvas)
            images = padded
        inputs = processor(images, size={"height": size[0], "width": size[1]}, return_tensors="pt")
        inputs["labels"] = torch.tensor([label_of(v) for v in batch["jersey_number"]], dtype=torch.long)
        return inputs

    return dataset.select_columns(["image", "jersey_number"]).with_transform(transform)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output directory name under training/runs/")
    ap.add_argument("--gsr-repeat", type=int, default=2, help="times the GSR train crops are repeated per epoch")
    ap.add_argument("--no-gsr", action="store_true", help="20 matches only (control run)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128, help="per GPU")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=8, help="data-loading processes per GPU")
    ap.add_argument("--augment", action="store_true", help="colour jitter + small affine on the training crops")
    ap.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"),
                    help="model input size; the default stretches every crop to a square like the deployed reader")
    ap.add_argument("--pad-square", action="store_true", help="pad crops to a square (keeps the aspect ratio) before resizing")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    backbone = TRAINING / "jersey_train_pack/tmp/dinov3-convnext-tiny"
    processor = AutoImageProcessor.from_pretrained(backbone)

    # 20 matches, without the validation rows
    held_out = set((TRAINING / "val_20_matches_ids.txt").read_text().split())
    old = load_dataset("parquet", split="train",
                       data_files=[str(p) for p in sorted((TRAINING / "jersey_train_pack/tmp/data_20_matches").glob("jersey_numbers*.parquet"))])
    old = old.filter(lambda ids: [i not in held_out for i in ids], batched=True, input_columns=["image_id"])
    parts = [old]
    if not args.no_gsr:
        gsr = load_dataset("parquet", split="train", data_files=str(TRAINING / "gsr_crops/train.parquet"))
        parts += [gsr] * args.gsr_repeat
    train = concatenate_datasets(parts)

    # both validation sets in one dataset (20 matches first), split again when scoring
    val_20m = load_dataset("parquet", split="train", data_files=str(TRAINING / "val_20_matches.parquet"))
    val_gsr = load_dataset("parquet", split="train", data_files=str(TRAINING / "gsr_crops/val.parquet"))
    n_20m = len(val_20m)

    def compute_metrics(pred):
        logits, labels = pred
        hit = np.argmax(logits, axis=-1) == labels
        acc_20m, acc_gsr = hit[:n_20m].mean(), hit[n_20m:].mean()
        return {"acc_20m": float(acc_20m), "acc_gsr": float(acc_gsr), "acc_mean": float((acc_20m + acc_gsr) / 2)}

    run_dir = TRAINING / "runs" / args.name
    trainer = Trainer(
        model=JerseyClassifier(backbone),
        args=TrainingArguments(
            output_dir=str(run_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr, bf16=args.bf16, seed=args.seed,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            metric_for_best_model="acc_mean", greater_is_better=True,
            logging_steps=100, report_to="tensorboard",
            remove_unused_columns=False, dataloader_num_workers=args.workers),
        train_dataset=with_pixels(train, processor, args.augment, args.size, args.pad_square),
        eval_dataset=with_pixels(concatenate_datasets([val_20m, val_gsr]), processor, size=args.size, pad_square=args.pad_square),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    if trainer.is_world_process_zero():
        # Keep the weights of the epoch with the best validation accuracy (not the last one).
        best = Path(trainer.state.best_model_checkpoint or sorted(run_dir.glob("checkpoint-*"))[-1])
        weights = best / "model.safetensors"
        state = load_file(weights) if weights.exists() else torch.load(best / "pytorch_model.bin", map_location="cpu")
        save_file({k: v.contiguous() for k, v in state.items()}, run_dir / "jersey.safetensors")
        print(f"best checkpoint {best.name}, metric {trainer.state.best_metric}; weights -> {run_dir / 'jersey.safetensors'}")


if __name__ == "__main__":
    main()
