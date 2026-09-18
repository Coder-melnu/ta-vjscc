#!/usr/bin/env python
"""Fine-tune Kinetics-400 TSM-ResNet-50 on five-frame UCF101 GoPs.

Purpose
-------
This script evaluates whether an explicit temporal-interaction model is a
stronger action-recognition teacher/evaluator for TA-VJSCC than the current
ImageNet-pretrained, frame-independent ResNet-50 consensus model.  It adapts
the official MIT Han Lab TSM-ResNet-50 checkpoint, originally trained on
Kinetics-400 with eight sampled segments, to the existing TA-VJSCC setting of
five consecutive frames at 128 x 128 resolution.

Why an eight-frame checkpoint can be used with five frames
-----------------------------------------------------------
TSM exchanges information between neighbouring frames by shifting feature
channels inside the ResNet residual blocks.  This temporal shift is
parameter-free: the checkpoint contains learned ResNet weights, but no
learned parameter whose shape depends on the number of segments.  Therefore,
the Kinetics checkpoint can be instantiated with ``num_segments=5`` and then
adapted to five-frame UCF101 GoPs.  This changes the temporal sampling context,
so UCF101 fine-tuning and validation are still required.

Training protocol
-----------------
* Dataset: UCF101 official Split 1.
* The 9,537 official training clips are divided, at clip level, into a
  deterministic class-stratified internal training/validation split.  The
  default is 90% training and 10% validation with random seed 42.
* The 3,783 official test clips remain separate and are not evaluated during
  epoch-by-epoch model selection.
* By default, ResNet ``layer4`` and the new ``Linear(2048, 101)`` classifier
  are fine-tuned.  Earlier layers remain frozen and BatchNorm running
  statistics are kept fixed for stability with small video batches.
* Cross-entropy loss and SGD are used.  The learning rate is reduced twice by
  ``MultiStepLR``.  CUDA automatic mixed precision is enabled unless
  ``--no_amp`` is passed.
* The best checkpoint is selected only by internal-validation Top-1 accuracy.
  After training finishes, that checkpoint is loaded and evaluated exactly
  once on the official test set.

Reported outputs
----------------
* Per-epoch training loss and training Top-1 accuracy.
* Per-epoch internal-validation loss, Top-1, and Top-5 accuracy.
* Final official-test loss, Top-1, and Top-5 accuracy.
* A full best-model checkpoint (``.pth``), an epoch log (``.jsonl``), and a
  final test record (``.test.json``).

Important interpretation
------------------------
This experiment determines whether the complete Kinetics-pretrained TSM
pipeline is a stronger five-frame action-recognition model.  By itself, a gain
over the previous ImageNet-only recognizer does not isolate the effect of
temporal shifting, because pretraining and fine-tuning also differ.  A matched
Kinetics-pretrained TSN-versus-TSM comparison and spatial-only/shuffled-time
importance controls are still required to support a causal temporal-importance
claim.

Example
-------
Run from the TA-VJSCC repository root::

    python downstream/action_recognition/finetune_tsm.py \
        --device cuda:0 \
        --epochs 25 \
        --batch_size 8 \
        --num_workers 4 \
        --image_size 128 \
        --num_segments 5 \
        --finetune layer4 \
        --val_fraction 0.1 \
        --seed 42 \
        --lr 0.001 \
        --save_path downstream/action_recognition/weights/tsm_ucf101_5f_split_best.pth
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.ucf101_dataloader import (
    UCF101GoPDataset,
    load_class_index,
    parse_testlist,
    parse_trainlist,
)
from downstream.action_recognition.models.tsm_recognizer import (
    TSMResNet50,
    configure_trainable_layers,
    download_kinetics_checkpoint,
    freeze_batch_norm_statistics,
    load_kinetics_backbone,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_root", default="datasets/UCF101Frames")
    parser.add_argument(
        "--annotation_path",
        default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist",
    )
    parser.add_argument(
        "--kinetics_checkpoint",
        default=(
            "downstream/action_recognition/weights/"
            "TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth"
        ),
    )
    parser.add_argument(
        "--save_path",
        default="downstream/action_recognition/weights/tsm_ucf101_5f_best.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_segments", type=int, default=5)
    parser.add_argument("--gops_per_clip", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--finetune", choices=("head", "layer4", "all"), default="layer4"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(gops: torch.Tensor) -> torch.Tensor:
    mean = gops.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = gops.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (gops - mean) / std


def limit_reached(batch_index: int, maximum: int) -> bool:
    return maximum > 0 and batch_index >= maximum


def stratified_train_val_split(samples, val_fraction: float, seed: int):
    """Split official training clips by class, with no clip overlap."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1.")

    by_class = {}
    for frame_dir, label in samples:
        by_class.setdefault(label, []).append((frame_dir, label))

    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    for label in sorted(by_class):
        class_samples = list(by_class[label])
        rng.shuffle(class_samples)
        val_count = max(1, round(len(class_samples) * val_fraction))
        # Every class must retain at least one training clip.
        val_count = min(val_count, len(class_samples) - 1)
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_paths = {path for path, _ in train_samples}
    val_paths = {path for path, _ in val_samples}
    overlap = train_paths & val_paths
    if overlap:
        raise RuntimeError(f"Train/validation clip overlap detected: {len(overlap)}")

    return train_samples, val_samples


def build_protocol_loaders(args):
    """Build internal train/validation loaders and untouched official test loader."""

    official_train = parse_trainlist(
        args.annotation_path, args.frames_root, split=1
    )
    class_to_idx = load_class_index(args.annotation_path)
    official_test = parse_testlist(
        args.annotation_path, args.frames_root, class_to_idx, split=1
    )
    train_samples, val_samples = stratified_train_val_split(
        official_train, args.val_fraction, args.seed
    )

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = UCF101GoPDataset(
        train_samples,
        transform,
        gop_size=args.num_segments,
        gops_per_clip=args.gops_per_clip,
        seed=args.seed,
    )
    val_dataset = UCF101GoPDataset(
        val_samples,
        transform,
        gop_size=args.num_segments,
        gops_per_clip=args.gops_per_clip,
        seed=args.seed,
    )
    test_dataset = UCF101GoPDataset(
        official_test,
        transform,
        gop_size=args.num_segments,
        gops_per_clip=args.gops_per_clip,
        seed=args.seed,
    )

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def validate(model, loader, device, criterion, max_batches: int):
    model.eval()
    total = top1_correct = top5_correct = 0
    loss_sum = 0.0
    for batch_index, (gops, labels) in enumerate(
        tqdm(loader, desc="validation", leave=False)
    ):
        if limit_reached(batch_index, max_batches):
            break
        gops = normalize(gops.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        logits = model(gops)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        top5_predictions = logits.topk(5, dim=1).indices
        top1_correct += (top5_predictions[:, 0] == labels).sum().item()
        top5_correct += (
            (top5_predictions == labels.unsqueeze(1)).any(dim=1).sum().item()
        )
        total += labels.size(0)
    return (
        loss_sum / max(total, 1),
        100.0 * top1_correct / max(total, 1),
        100.0 * top5_correct / max(total, 1),
        total,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    checkpoint_path = download_kinetics_checkpoint(args.kinetics_checkpoint)
    model = TSMResNet50(
        num_classes=101,
        num_segments=args.num_segments,
        dropout=0.8,
        shift_div=8,
    )
    load_kinetics_backbone(model, checkpoint_path)
    trainable = configure_trainable_layers(model, args.finetune)
    model.to(device)

    train_loader, val_loader, test_loader = build_protocol_loaders(args)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        parameters,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    milestones = sorted({max(1, args.epochs // 2), max(2, 4 * args.epochs // 5)})
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda" and not args.no_amp
    # torch.cuda.amp is compatible with the server's PyTorch 2.1 installation.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = save_path.with_suffix(".jsonl")
    # Do not mix records from a smoke test and a later full run.
    log_path.write_text("", encoding="utf-8")

    print("\nTSM five-frame diagnostic")
    print(f"  device:       {device}")
    print(f"  train clips:  {len(train_loader.dataset)}")
    print(f"  internal val: {len(val_loader.dataset)}")
    print(f"  test clips:   {len(test_loader.dataset)} (untouched until training ends)")
    print(f"  segments:     {args.num_segments}")
    print(f"  image size:   {args.image_size}")
    print(f"  fine-tuning:  {args.finetune}")
    print(f"  trainable:    {trainable:,} parameters")
    print(f"  checkpoint:   {checkpoint_path}\n")

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_batch_norm_statistics(model)
        total = correct = 0
        loss_sum = 0.0

        progress = tqdm(train_loader, desc=f"epoch {epoch:02d}/{args.epochs}")
        for batch_index, (gops, labels) in enumerate(progress):
            if limit_reached(batch_index, args.max_train_batches):
                break
            gops = normalize(gops.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(gops)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            progress.set_postfix(
                loss=f"{loss_sum / max(total, 1):.4f}",
                acc=f"{100.0 * correct / max(total, 1):.2f}%",
            )

        val_loss, val_top1, val_top5, val_count = validate(
            model, val_loader, device, criterion, args.max_val_batches
        )
        train_loss = loss_sum / max(total, 1)
        train_accuracy = 100.0 * correct / max(total, 1)
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_samples": total,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top5": val_top5,
            "val_samples": val_count,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"epoch {epoch:02d}: train loss={train_loss:.4f}, "
            f"train acc={train_accuracy:.2f}%, val loss={val_loss:.4f}, "
            f"val Top-1={val_top1:.2f}%, val Top-5={val_top5:.2f}%"
        )

        if val_top1 > best_accuracy:
            best_accuracy = val_top1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "num_segments": args.num_segments,
                    "image_size": args.image_size,
                    "finetune": args.finetune,
                    "class_count": 101,
                },
                save_path,
            )
            print(f"  saved best checkpoint -> {save_path}")

        scheduler.step()

    print(f"\nBest internal-validation Top-1: {best_accuracy:.2f}%")
    print(f"Loading selected checkpoint for one final official-test evaluation ...")
    best_checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_top1, test_top5, test_count = validate(
        model, test_loader, device, criterion, max_batches=0
    )
    test_record = {
        "selected_epoch": best_checkpoint["epoch"],
        "selection_val_top1": best_checkpoint["val_top1"],
        "selection_val_top5": best_checkpoint["val_top5"],
        "test_loss": test_loss,
        "test_top1": test_top1,
        "test_top5": test_top5,
        "test_samples": test_count,
    }
    test_path = save_path.with_suffix(".test.json")
    test_path.write_text(json.dumps(test_record, indent=2) + "\n", encoding="utf-8")

    print(f"Official test Top-1: {test_top1:.2f}%")
    print(f"Official test Top-5: {test_top5:.2f}%")
    print(f"Checkpoint:          {save_path}")
    print(f"Training log:        {log_path}")
    print(f"Final test record:   {test_path}")


if __name__ == "__main__":
    main()
