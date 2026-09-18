"""Minimal UCF101 annotation and fixed-GoP dataset utilities."""

import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def _natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", Path(path).name)]


def get_sorted_frames(frame_dir):
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        (str(p) for p in Path(frame_dir).iterdir() if p.suffix.lower() in exts),
        key=_natural_key,
    )


def parse_trainlist(annotation_path, frames_root, split=1):
    annotation_file = Path(annotation_path) / f"trainlist0{split}.txt"
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)
    samples, missing = [], []
    for line in annotation_file.read_text().splitlines():
        rel_path, label_text = line.split()
        rel = Path(rel_path)
        frame_dir = Path(frames_root) / "train" / rel.parent.name / rel.stem
        if frame_dir.is_dir():
            samples.append((str(frame_dir.resolve()), int(label_text) - 1))
        else:
            missing.append(str(frame_dir))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} official-train frame directories missing:\n{preview}")
    return samples


class UCF101GoPDataset(Dataset):
    """One or more fixed, reproducibly sampled consecutive GoPs per clip."""

    def __init__(self, samples, transform, gop_size=5, gops_per_clip=1, seed=42):
        if gop_size < 1 or gops_per_clip < 1:
            raise ValueError("gop_size and gops_per_clip must be positive")
        self.transform = transform
        self.gop_size = gop_size
        rng = random.Random(seed)
        self.index, self.rejected = [], []
        for frame_dir, label in samples:
            frames = get_sorted_frames(frame_dir)
            if len(frames) < gop_size:
                self.rejected.append((frame_dir, len(frames)))
                continue
            max_start = len(frames) - gop_size
            starts = [rng.randint(0, max_start) for _ in range(gops_per_clip)]
            for start in starts:
                self.index.append((frames[start:start + gop_size], label, frame_dir, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        frame_paths, label, _, _ = self.index[index]
        frames = []
        for path in frame_paths:
            with Image.open(path) as image:
                image = image.convert("RGB")
                frames.append(self.transform(image) if self.transform else image)
        return torch.stack(frames), label
