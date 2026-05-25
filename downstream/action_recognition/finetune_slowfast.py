#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone SlowFast head fine-tuning script.
Run overnight on the RTX 3090 Ti (cuda:1).

Usage:
    nohup python downstream/action_recognition/finetune_slowfast.py \
        --device cuda:1 \
        --epochs 10 \
        --batch_size 4 \
        > downstream/action_recognition/logs/finetune_slowfast.log 2>&1 &

    tail -f downstream/action_recognition/logs/finetune_slowfast.log

Notes:
    - Backbone FROZEN — only Linear(2304, 101) trains
    - Expected Top-1 on UCF101: ~60–75% (head-only, no full fine-tune)
    - VRAM: ~6–8 GB with batch_size=4 on 3090 Ti
    - Time: ~4–8 hours for 10 epochs on 3090 Ti
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune SlowFast head on UCF101')
    parser.add_argument('--frames_root',
        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--save_path',
        default='downstream/action_recognition/weights/slowfast_ucf101_head.pth')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=4,
        help='4 for 3090 Ti (cuda:1), reduce to 2 if OOM')
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--device',      default='cuda:1',
        help='Use cuda:1 (3090 Ti) — leaves cuda:0 (3060) free for main training')
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--image_size',  type=int,   default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.makedirs('downstream/action_recognition/logs',    exist_ok=True)
    os.makedirs('downstream/action_recognition/weights', exist_ok=True)

    print("=" * 60)
    print("  SlowFast Head Fine-tuning on UCF101")
    print("=" * 60)
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Save path  : {args.save_path}")
    print(f"  Backbone   : FROZEN")
    print("=" * 60)

    from downstream.action_recognition.models.slowfast_recognizer import finetune_slowfast_head

    finetune_slowfast_head(
        frames_root=args.frames_root,
        annotation_path=args.annotation_path,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )