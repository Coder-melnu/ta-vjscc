# -*- coding: utf-8 -*-
"""
Video utilities — shared across all downstream tasks.
Frame loading, GoP↔numpy conversion, video writing.
"""

import os
import sys
import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

UCF_FRAMES_ROOT = os.path.join(PROJECT_ROOT, 'datasets/UCF101Frames')
UCF_CLASSIND    = os.path.join(
    PROJECT_ROOT,
    'datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt'
)


# ---------------------------------------------------------------------------
# UCF101 video listing
# ---------------------------------------------------------------------------

def list_ucf101_videos(
    frames_root:   str,
    classind_path: str,
    split_file:    str,
    max_videos:    Optional[int] = None,
) -> List[Tuple[str, int, str]]:
    """
    List UCF101 videos from a split file.

    Returns:
        list of (frame_dir, class_idx_0based, class_name)
    """
    label2idx = {}
    with open(classind_path) as f:
        for line in f:
            idx, name = line.strip().split()
            label2idx[name] = int(idx) - 1

    videos = []
    with open(split_file) as f:
        for line in f:
            entry = line.strip()
            if not entry:
                continue
            rel_path   = entry.split()[0]     # strip label if present
            if '/' in rel_path:
                class_name, video_file = rel_path.split('/', 1)
            else:
                video_file = rel_path
                class_name = rel_path.split('_')[1] if '_' in rel_path else 'Unknown'

            video_name = os.path.splitext(video_file)[0]
            frame_dir  = os.path.join(frames_root, class_name, video_name)

            if not os.path.isdir(frame_dir):
                continue

            class_idx = label2idx.get(class_name, -1)
            if class_idx == -1:
                continue

            videos.append((frame_dir, class_idx, class_name))

            if max_videos and len(videos) >= max_videos:
                break

    return videos


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def load_frames_from_dir(
    frame_dir:  str,
    n_frames:   int = 5,
    image_size: int = 128,
    start_idx:  int = 0,
) -> Optional[torch.Tensor]:
    """
    Load N consecutive frames from a UCF101 frame directory.

    Returns:
        (N, 3, H, W) float [0,1]  or  None if not enough frames
    """
    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(frame_files) < n_frames:
        return None

    if len(frame_files) < start_idx + n_frames:
        indices    = np.linspace(0, len(frame_files) - 1, n_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    else:
        frame_files = frame_files[start_idx: start_idx + n_frames]

    frames = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(frame_dir, fname))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size, image_size))
        frames.append(img)

    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)   # (N, 3, H, W)


# ---------------------------------------------------------------------------
# Tensor ↔ numpy conversion
# ---------------------------------------------------------------------------

def gop_tensor_to_numpy(gop: torch.Tensor) -> np.ndarray:
    """
    (N, 3, H, W) float [0,1]  →  (N, H, W, 3) uint8
    Also accepts (B, N, 3, H, W) — takes first item in batch.
    """
    if gop.dim() == 5:
        gop = gop[0]
    gop = gop.detach().cpu().permute(0, 2, 3, 1)
    return (gop * 255).clamp(0, 255).numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def write_frames_to_video(
    frames:      np.ndarray,
    output_path: str,
    fps:         int = 25,
) -> str:
    """
    Write (N, H, W, 3) uint8 RGB frames to .mp4.

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return output_path


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.path.isdir(UCF_FRAMES_ROOT):
        classes = os.listdir(UCF_FRAMES_ROOT)
        cls     = classes[0]
        vids    = os.listdir(os.path.join(UCF_FRAMES_ROOT, cls))
        if vids:
            frame_dir = os.path.join(UCF_FRAMES_ROOT, cls, vids[0])
            gop = load_frames_from_dir(frame_dir, n_frames=5, image_size=128)
            if gop is not None:
                print(f"[video_utils] GoP: {gop.shape}, min={gop.min():.3f}, max={gop.max():.3f}")
                np_frames = gop_tensor_to_numpy(gop)
                print(f"[video_utils] numpy: {np_frames.shape}, dtype={np_frames.dtype}")
    else:
        print(f"[video_utils] {UCF_FRAMES_ROOT} not found")