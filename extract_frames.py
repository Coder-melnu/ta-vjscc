# -*- coding: utf-8 -*-
"""
Extract frames from UCF101 .avi videos to PNG images.

Usage:
    # Use defaults
    python extract_frames.py

    # Override arguments
    python extract_frames.py --videos_root datasets/UCF101Videos --frames_root datasets/UCF101Frames
    python extract_frames.py --image_size 128 --splits train
    python extract_frames.py --splits test   # extract only test set

Output structure:
    datasets/UCF101Frames/
    ├── train/
    │   ├── ApplyEyeMakeup/
    │   │   ├── v_ApplyEyeMakeup_g08_c01/
    │   │   │   ├── 0001.png
    │   │   │   ├── 0002.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── test/
        └── ...
"""

import os
import subprocess
import argparse
from pathlib import Path


def extract_clip(video_path: str, output_dir: str, image_size: int = 256) -> int:
    """
    Extract all frames from a single .avi video file.

    Args:
        video_path : path to input .avi file
        output_dir : folder to save extracted PNG frames
        image_size : resize frames to (image_size x image_size)

    Returns:
        number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'scale={image_size}:{image_size}',
        os.path.join(output_dir, '%04d.png'),
        '-hide_banner', '-loglevel', 'error'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: FFmpeg failed for {video_path}\n  {result.stderr}")
        return 0

    n_frames = len(list(Path(output_dir).glob('*.png')))
    return n_frames


def extract_split(videos_root: str, frames_root: str, split: str,
                  image_size: int = 256, skip_existing: bool = True) -> dict:
    """
    Extract all frames for one split (train or test).

    Args:
        videos_root    : root folder containing train/ and test/ video folders
        frames_root    : root folder to save extracted frames
        split          : 'train' or 'test'
        image_size     : resize frames to square
        skip_existing  : skip clips that already have extracted frames

    Returns:
        dict with stats: n_clips, n_frames, n_skipped, n_failed
    """
    split_dir = Path(videos_root) / split
    if not split_dir.exists():
        print(f"  WARNING: {split_dir} does not exist — skipping")
        return {}

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"\n[{split.upper()}] Found {len(class_dirs)} classes")

    stats = {'n_clips': 0, 'n_frames': 0, 'n_skipped': 0, 'n_failed': 0}

    for class_dir in class_dirs:
        class_name = class_dir.name
        videos = sorted(class_dir.glob('*.avi'))

        for video in videos:
            clip_name  = video.stem
            output_dir = str(Path(frames_root) / split / class_name / clip_name)

            # Skip if already extracted
            if skip_existing and Path(output_dir).exists():
                existing = len(list(Path(output_dir).glob('*.png')))
                if existing > 0:
                    stats['n_skipped'] += 1
                    stats['n_frames']  += existing
                    continue

            # Extract
            n = extract_clip(str(video), output_dir, image_size)
            stats['n_clips'] += 1

            if n == 0:
                stats['n_failed'] += 1
            else:
                stats['n_frames'] += n

        print(f"  Done: {split}/{class_name} ({len(videos)} clips)")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract UCF101 frames from .avi videos')

    # ---- Edit defaults here ----
    parser.add_argument('--videos_root',   default='datasets/UCF101Videos',
                        help='Root folder containing train/ and test/ video folders')
    parser.add_argument('--frames_root',   default='datasets/UCF101Frames',
                        help='Root folder to save extracted PNG frames')
    parser.add_argument('--image_size',    type=int, default=256,
                        help='Resize frames to this square size')
    parser.add_argument('--splits',        nargs='+', default=['train', 'test'],
                        choices=['train', 'test'],
                        help='Which splits to extract')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip clips that already have extracted frames')
    # ----------------------------

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"UCF101 Frame Extraction")
    print(f"Videos root : {args.videos_root}")
    print(f"Frames root : {args.frames_root}")
    print(f"Image size  : {args.image_size}x{args.image_size}")
    print(f"Splits      : {args.splits}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"{'='*60}")

    total_frames = 0
    total_clips  = 0

    for split in args.splits:
        stats = extract_split(
            videos_root=args.videos_root,
            frames_root=args.frames_root,
            split=split,
            image_size=args.image_size,
            skip_existing=args.skip_existing,
        )
        total_frames += stats.get('n_frames', 0)
        total_clips  += stats.get('n_clips', 0) + stats.get('n_skipped', 0)

        print(f"\n[{split.upper()}] Summary:")
        print(f"  Extracted : {stats.get('n_clips', 0)} clips")
        print(f"  Skipped   : {stats.get('n_skipped', 0)} clips (already done)")
        print(f"  Failed    : {stats.get('n_failed', 0)} clips")
        print(f"  Frames    : {stats.get('n_frames', 0)}")

    print(f"\n{'='*60}")
    print(f"DONE — {total_clips} total clips, {total_frames} total frames")
    print(f"Frames saved to: {args.frames_root}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()