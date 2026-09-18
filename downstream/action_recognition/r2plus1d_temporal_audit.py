#!/usr/bin/env python
"""Audit temporal sensitivity of the locked five-frame R(2+1)D evaluator."""

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.ucf101_train_val_test import build_train_val_dataloaders
from downstream.action_recognition.models.r2plus1d_recognizer import (
    R2Plus1DRecognizer, load_r2plus1d_checkpoint, normalization_tensors,
)

EXPECTED_MANIFEST_SHA256 = "ea27a8557ef1e8f658f63c8f86f33721d6f94a1e58c657fa0a02c93e94befb7c"
CONDITIONS = ("original", "reversed", "shuffled", "centre_repeated")
SHUFFLE_ORDER = (2, 0, 4, 1, 3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="downstream/action_recognition/weights/r2plus1d_ucf101_layer4_locked_split_best.pt")
    p.add_argument("--frames_root", default="datasets/UCF101Frames")
    p.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    p.add_argument("--output_dir", default="downstream/action_recognition/results/r2plus1d_temporal_audit")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def transform(gops, condition):
    if condition == "original":
        return gops
    if condition == "reversed":
        return gops.flip(1)
    if condition == "shuffled":
        return gops[:, SHUFFLE_ORDER]
    if condition == "centre_repeated":
        return gops[:, 2:3].expand(-1, 5, -1, -1, -1)
    raise ValueError(condition)


def wilson_interval(correct, total, z=1.96):
    p = correct / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [100 * (centre - radius), 100 * (centre + radius)]


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=args.frames_root, annotation_path=args.annotation_path,
        image_size=128, batch_size=args.batch_size, num_workers=args.num_workers,
        gop_size=5, gops_per_clip=1, val_fraction=0.1, seed=42,
        enforce_locked_counts=True,
    )
    if len(val_loader.dataset) != 1128 or manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Locked validation mismatch: n={len(val_loader.dataset)}, hash={manifest_hash}")

    model = R2Plus1DRecognizer(pretrained=False, num_classes=101)
    metadata = load_r2plus1d_checkpoint(model, checkpoint)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.freeze_batchnorm()
    mean, std = normalization_tensors(device)
    criterion = nn.CrossEntropyLoss(reduction="none")
    rows = []
    totals = {name: {"correct1": 0, "correct5": 0, "loss_sum": 0.0} for name in CONDITIONS}
    offset = 0

    with torch.inference_mode():
        for gops, labels in tqdm(val_loader, desc="temporal audit"):
            gops = gops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_outputs = {}
            for condition in CONDITIONS:
                clips = transform(gops, condition)
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = model((clips - mean) / std)
                losses = criterion(logits.float(), labels)
                top5 = logits.topk(5, dim=1).indices
                pred = top5[:, 0]
                correct1 = pred.eq(labels)
                correct5 = top5.eq(labels[:, None]).any(dim=1)
                totals[condition]["correct1"] += int(correct1.sum())
                totals[condition]["correct5"] += int(correct5.sum())
                totals[condition]["loss_sum"] += float(losses.sum())
                batch_outputs[condition] = (pred.cpu(), correct1.cpu(), correct5.cpu())
            for i in range(labels.size(0)):
                record = {"sample_index": offset + i, "label": int(labels[i])}
                for condition in CONDITIONS:
                    pred, c1, c5 = batch_outputs[condition]
                    record[f"{condition}_prediction"] = int(pred[i])
                    record[f"{condition}_top1_correct"] = int(c1[i])
                    record[f"{condition}_top5_correct"] = int(c5[i])
                rows.append(record)
            offset += labels.size(0)

    total = len(rows)
    metrics = {}
    for condition in CONDITIONS:
        item = totals[condition]
        metrics[condition] = {
            "loss": item["loss_sum"] / total,
            "top1_correct": item["correct1"],
            "top1_percent": 100 * item["correct1"] / total,
            "top1_wilson_95_percent": wilson_interval(item["correct1"], total),
            "top5_correct": item["correct5"],
            "top5_percent": 100 * item["correct5"] / total,
            "top1_drop_from_original_pp": 100 * (totals["original"]["correct1"] - item["correct1"]) / total,
        }

    fieldnames = list(rows[0])
    with (output_dir / "per_clip_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed", "architecture": "torchvision_r2plus1d_18",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": metadata.get("epoch"), "validation_samples": total,
        "split_manifest_sha256": manifest_hash, "official_test_used": False,
        "parameter_updates": 0, "batchnorm_frozen": True,
        "input_frames": 5, "image_size": 128,
        "shuffle_order_zero_based": list(SHUFFLE_ORDER), "metrics": metrics,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
