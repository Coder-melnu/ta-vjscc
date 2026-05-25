# -*- coding: utf-8 -*-
"""
UCF101 DataLoader for TA-VJSCC experiments.

Supports two modes:
  - 'frame' : returns individual frames (B, C, H, W)   — Week 2
  - 'gop'   : returns N consecutive frames (B, N, C, H, W) — Week 3

Expected folder structure after frame extraction:
    datasets/
    ├── UCF101Frames/
    │   ├── train/
    │   │   ├── ApplyEyeMakeup/
    │   │   │   ├── v_ApplyEyeMakeup_g08_c01/
    │   │   │   │   ├── 0001.png
    │   │   │   │   ├── 0002.png
    │   │   │   │   └── ...
    │   │   │   └── ...
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── UCF101TrainTestSplits-RecognitionTask/
        └── ucfTrainTestlist/
            ├── classInd.txt
            ├── trainlist01.txt
            └── testlist01.txt

Annotation file formats:
    trainlist01.txt : "ClassName/v_xxx.avi <label>"   (1-indexed label)
    testlist01.txt  : "ClassName/v_xxx.avi"            (no label)
    classInd.txt    : "<label> ClassName"              (1-indexed)

Usage:
    from data.ucf101_dataloader import build_dataloaders

    # Week 2 — individual frames
    train_loader, test_loader = build_dataloaders(
        frames_root='datasets/UCF101Frames',
        annotation_path='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist',
        mode='frame',
        image_size=256,
        batch_size=32,
    )

    # Week 3 — GoPs of 5 frames
    train_loader, test_loader = build_dataloaders(
        frames_root='datasets/UCF101Frames',
        annotation_path='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist',
        mode='gop',
        gop_size=5,
        gops_per_clip=1,
        image_size=256,
        batch_size=32,
    )
"""

import os
import random
from pathlib import Path
from typing import Literal, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Annotation Parsing
# ---------------------------------------------------------------------------

def load_class_index(annotation_path: str) -> dict:
    """
    Parse classInd.txt → {class_name: int_label} (0-indexed).

    classInd.txt format: "<1-indexed-label> ClassName"
    """
    class_file = Path(annotation_path) / 'classInd.txt'
    mapping = {}
    with open(class_file) as f:
        for line in f:
            idx, name = line.strip().split()
            mapping[name] = int(idx) - 1    # convert to 0-indexed
    return mapping


def parse_trainlist(annotation_path: str, frames_root: str,
                    split: int = 1) -> list:
    """
    Parse trainlist0{split}.txt.
    Returns list of (frame_folder_path, label) tuples.

    trainlist format: "ClassName/v_xxx.avi <label>"
    Maps to        : frames_root/train/ClassName/v_xxx/
    """
    train_file = Path(annotation_path) / f'trainlist0{split}.txt'
    samples = []
    with open(train_file) as f:
        for line in f:
            parts = line.strip().split()
            rel_path = parts[0]                     # e.g. ApplyEyeMakeup/v_xxx.avi
            label    = int(parts[1]) - 1            # 0-indexed

            # Strip .avi → frame folder name
            clip_name  = Path(rel_path).stem        # v_ApplyEyeMakeup_g08_c01
            class_name = Path(rel_path).parent.name # ApplyEyeMakeup

            frame_dir = Path(frames_root) / 'train' / class_name / clip_name

            if frame_dir.exists():
                samples.append((str(frame_dir), label))

    print(f"  Train split {split}: {len(samples)} clips found")
    return samples


def parse_testlist(annotation_path: str, frames_root: str,
                   class_to_idx: dict, split: int = 1) -> list:
    """
    Parse testlist0{split}.txt.
    Returns list of (frame_folder_path, label) tuples.

    testlist format: "ClassName/v_xxx.avi"  (no label column)
    Label is inferred from class name via classInd.txt.
    Maps to        : frames_root/test/ClassName/v_xxx/
    """
    test_file = Path(annotation_path) / f'testlist0{split}.txt'
    samples = []
    with open(test_file) as f:
        for line in f:
            rel_path   = line.strip()               # e.g. ApplyEyeMakeup/v_xxx.avi
            clip_name  = Path(rel_path).stem        # v_ApplyEyeMakeup_g01_c01
            class_name = Path(rel_path).parent.name # ApplyEyeMakeup
            label      = class_to_idx.get(class_name, -1)

            frame_dir = Path(frames_root) / 'test' / class_name / clip_name

            if frame_dir.exists():
                samples.append((str(frame_dir), label))

    print(f"  Test  split {split}: {len(samples)} clips found")
    return samples


def get_sorted_frames(frame_dir: str) -> list:
    """Return sorted list of PNG frame paths in a clip folder."""
    exts = {'.png', '.jpg', '.jpeg'}
    frames = sorted([
        str(f) for f in Path(frame_dir).iterdir()
        if f.suffix.lower() in exts
    ])
    return frames


# ---------------------------------------------------------------------------
# Frame Dataset (Week 2)
# ---------------------------------------------------------------------------

class UCF101FrameDataset(Dataset):
    """
    Each item is a single frame (C, H, W) tensor + class label.
    Samples frames_per_clip frames uniformly from each clip.

    Args:
        samples         : list of (frame_dir, label) from parse_trainlist/testlist
        transform       : torchvision transform applied to each frame
        frames_per_clip : how many frames to sample per clip
        seed            : random seed for reproducible sampling
    """

    def __init__(self, samples: list, transform,
                 frames_per_clip: int = 10, seed: int = 42):
        self.transform       = transform
        self.frames_per_clip = frames_per_clip
        self.rng             = random.Random(seed)

        # Build flat index: list of (frame_path, label)
        print(f"  Indexing {len(samples)} clips...")
        self.index = []
        missing = 0
        for frame_dir, label in samples:
            frames = get_sorted_frames(frame_dir)
            if not frames:
                missing += 1
                continue
            n = min(frames_per_clip, len(frames))
            chosen = sorted(self.rng.sample(frames, n))
            for fp in chosen:
                self.index.append((fp, label))

        print(f"  → {len(self.index)} frames indexed "
              f"({missing} clips skipped — frames not yet extracted)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_path, label = self.index[idx]
        img = Image.open(frame_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# GoP Dataset (Week 3)
# ---------------------------------------------------------------------------

class UCF101GoPDataset(Dataset):
    """
    Each item is a GoP: (N, C, H, W) tensor of N consecutive frames + label.
    Samples gops_per_clip GoPs with random start positions per clip.

    Args:
        samples       : list of (frame_dir, label)
        transform     : torchvision transform applied to each frame
        gop_size      : number of consecutive frames per GoP (N)
        gops_per_clip : number of GoPs to randomly sample per clip
        seed          : random seed for reproducibility
    """

    def __init__(self, samples: list, transform,
                 gop_size: int = 5, gops_per_clip: int = 1, seed: int = 42):
        self.transform = transform
        self.gop_size  = gop_size
        self.rng       = random.Random(seed)

        print(f"  Indexing {len(samples)} clips for GoPs "
              f"(N={gop_size}, gops_per_clip={gops_per_clip})...")
        self.index = []
        missing = 0
        for frame_dir, label in samples:
            frames = get_sorted_frames(frame_dir)
            if len(frames) < gop_size:
                missing += 1
                continue
            max_start = len(frames) - gop_size
            for _ in range(gops_per_clip):
                start      = self.rng.randint(0, max_start)
                gop_frames = frames[start:start + gop_size]
                self.index.append((gop_frames, label))

        print(f"  → {len(self.index)} GoPs indexed "
              f"({missing} clips skipped — too short or not extracted)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        frame_paths, label = self.index[idx]
        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Pad if any frame failed to load
        while len(frames) < self.gop_size:
            frames.append(frames[-1] if frames else torch.zeros(3, 256, 256))

        gop = torch.stack(frames, dim=0)    # (N, C, H, W)
        return gop, label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    frames_root: str,
    annotation_path: str,
    mode: Literal['frame', 'gop'] = 'frame',
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    split: int = 1,
    frames_per_clip: int = 1,
    gop_size: int = 5,
    gops_per_clip: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for UCF101.

    Args:
        frames_root      : path to UCF101Frames/ (contains train/ and test/)
        annotation_path  : path to ucfTrainTestlist/ folder
        mode             : 'frame' (Week 2) or 'gop' (Week 3)
        image_size       : spatial size for resizing frames (square)
        batch_size       : samples per batch
        num_workers      : DataLoader worker processes
        split            : official split number (1, 2, or 3)
        frames_per_clip  : (frame mode) frames sampled per clip
        gop_size         : (gop mode) consecutive frames per GoP
        gops_per_clip    : (gop mode) GoPs randomly sampled per clip
        seed             : random seed for reproducibility

    Returns:
        (train_loader, test_loader)
    """
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1] — matches Deep-JSCC-PyTorch
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Parse annotations
    class_to_idx  = load_class_index(annotation_path)
    train_samples = parse_trainlist(annotation_path, frames_root, split)
    test_samples  = parse_testlist(annotation_path, frames_root, class_to_idx, split)

    # Build datasets
    if mode == 'frame':
        train_ds = UCF101FrameDataset(train_samples, train_transform,
                                      frames_per_clip=frames_per_clip, seed=seed)
        test_ds  = UCF101FrameDataset(test_samples,  test_transform,
                                      frames_per_clip=frames_per_clip, seed=seed)
    elif mode == 'gop':
        train_ds = UCF101GoPDataset(train_samples, train_transform,
                                    gop_size=gop_size, gops_per_clip=gops_per_clip,
                                    seed=seed)
        test_ds  = UCF101GoPDataset(test_samples,  test_transform,
                                    gop_size=gop_size, gops_per_clip=gops_per_clip,
                                    seed=seed)
    else:
        raise ValueError(f"mode must be 'frame' or 'gop', got '{mode}'")

    # Build loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=False)

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UCF101 DataLoader sanity check')
    parser.add_argument('--frames_root',
                        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
                        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--mode',            default='frame', choices=['frame', 'gop'])
    parser.add_argument('--gop_size',        type=int, default=5)
    parser.add_argument('--gops_per_clip',   type=int, default=1)
    parser.add_argument('--image_size',      type=int, default=256)
    parser.add_argument('--batch_size',      type=int, default=4)
    parser.add_argument('--frames_per_clip', type=int, default=3)
    args = parser.parse_args()

    print(f"\nBuilding DataLoaders (mode={args.mode})...")
    train_loader, test_loader = build_dataloaders(
        frames_root=args.frames_root,
        annotation_path=args.annotation_path,
        mode=args.mode,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
        frames_per_clip=args.frames_per_clip,
        gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip,
    )

    print(f"\n--- Train loader ---")
    frames, labels = next(iter(train_loader))
    print(f"  frames shape : {frames.shape}")
    print(f"  labels       : {labels.tolist()}")
    print(f"  frames range : [{frames.min():.3f}, {frames.max():.3f}]")

    print(f"\n--- Test loader ---")
    frames, labels = next(iter(test_loader))
    print(f"  frames shape : {frames.shape}")
    print(f"  labels       : {labels.tolist()}")

    print("\nSanity check passed.")