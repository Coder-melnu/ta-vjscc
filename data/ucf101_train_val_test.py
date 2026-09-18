"""Leakage-safe internal UCF101 train/validation loaders.

The official Split-1 test set is deliberately not parsed or constructed here.
"""

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .ucf101_dataloader import UCF101GoPDataset, parse_trainlist

EXPECTED_TRAIN = 8409
EXPECTED_VAL = 1128
EXPECTED_TRAIN_GROUPS = 1606
EXPECTED_VAL_GROUPS = 212


def video_group(frame_dir: str) -> str:
    match = re.search(r"_g(\d+)_", Path(frame_dir).name)
    if not match:
        raise ValueError(f"Invalid UCF101 clip name: {Path(frame_dir).name}")
    return f"g{int(match.group(1)):02d}"


def stratified_group_train_val_split(samples, val_fraction=0.1, seed=42):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    by_class = defaultdict(lambda: defaultdict(list))
    for frame_dir, label in samples:
        if not 0 <= int(label) < 101:
            raise ValueError(f"Invalid UCF101 label {label}: {frame_dir}")
        by_class[int(label)][video_group(frame_dir)].append((frame_dir, int(label)))

    train, val = [], []
    for label in sorted(by_class):
        groups = by_class[label]
        names = sorted(groups)
        if len(names) < 2:
            raise ValueError(f"Class {label} has fewer than two groups")
        random.Random(seed + label).shuffle(names)
        target = max(1, round(sum(map(len, groups.values())) * val_fraction))
        chosen, count = [], 0
        for name in names[:-1]:
            if count >= target:
                break
            chosen.append(name)
            count += len(groups[name])
        chosen = set(chosen)
        for name, group_samples in groups.items():
            (val if name in chosen else train).extend(group_samples)

    train.sort()
    val.sort()
    train_paths = {p for p, _ in train}
    val_paths = {p for p, _ in val}
    train_groups = {(y, video_group(p)) for p, y in train}
    val_groups = {(y, video_group(p)) for p, y in val}
    if train_paths & val_paths:
        raise RuntimeError("Clip leakage detected")
    if train_groups & val_groups:
        raise RuntimeError("Recording-group leakage detected")
    if {y for _, y in train} != set(range(101)) or {y for _, y in val} != set(range(101)):
        raise RuntimeError("Both internal subsets must contain all 101 classes")
    return train, val, train_groups, val_groups


def _worker_init(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)


def make_manifest(train, val, train_dataset, val_dataset, seed, val_fraction, gop_size, gops_per_clip):
    def rows(items):
        return [
            {"frame_dir": p, "label": y, "group": video_group(p)}
            for p, y in items
        ]

    def gop_rows(dataset):
        # Record the exact fixed GoP selected from every clip.  The split alone
        # is insufficient to reproduce reconstruction metrics or snapshots.
        return [
            {
                "frame_dir": frame_dir,
                "label": label,
                "start_frame_index": start,
                "frame_files": [Path(frame).name for frame in frame_paths],
            }
            for frame_paths, label, frame_dir, start in dataset.index
        ]
    manifest = {
        "protocol": "UCF101 official split 1; group-aware internal validation",
        "seed": seed,
        "val_fraction_requested": val_fraction,
        "gop_size": gop_size,
        "gops_per_clip": gops_per_clip,
        "train": rows(train),
        "validation": rows(val),
        "fixed_train_gops": gop_rows(train_dataset),
        "fixed_validation_gops": gop_rows(val_dataset),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return manifest, hashlib.sha256(canonical.encode()).hexdigest()


def build_train_val_dataloaders(
    frames_root, annotation_path, image_size=128, batch_size=8,
    num_workers=4, split=1, gop_size=5, gops_per_clip=1,
    val_fraction=0.1, seed=42, enforce_locked_counts=True,
):
    official_train = parse_trainlist(annotation_path, frames_root, split)
    train, val, train_groups, val_groups = stratified_group_train_val_split(
        official_train, val_fraction, seed
    )
    observed = (len(train), len(val), len(train_groups), len(val_groups))
    expected = (EXPECTED_TRAIN, EXPECTED_VAL, EXPECTED_TRAIN_GROUPS, EXPECTED_VAL_GROUPS)
    if enforce_locked_counts and observed != expected:
        raise RuntimeError(f"Locked split counts differ: observed={observed}, expected={expected}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor()
    ])
    train_ds = UCF101GoPDataset(
        train, transform, gop_size=gop_size, gops_per_clip=gops_per_clip, seed=seed
    )
    val_ds = UCF101GoPDataset(
        val, transform, gop_size=gop_size, gops_per_clip=gops_per_clip, seed=seed + 1
    )
    if len(train_ds) != len(train) * gops_per_clip or len(val_ds) != len(val) * gops_per_clip:
        raise RuntimeError("A selected clip is missing frames or shorter than the GoP")

    generator = torch.Generator().manual_seed(seed)
    common = dict(num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=generator, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    manifest, manifest_hash = make_manifest(
        train, val, train_ds, val_ds, seed, val_fraction, gop_size, gops_per_clip
    )
    return train_loader, val_loader, manifest, manifest_hash
