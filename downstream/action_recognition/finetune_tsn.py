#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone TSN head fine-tuning script.

Usage:
    conda activate action_eval
    cd ~/Nu/ta-vjscc
    python downstream/action_recognition/finetune_tsn.py \
        --device cuda:0 \
        --epochs 10 \
        --batch_size 16

Expected:
    - Runtime : ~3-5 min/epoch → ~30-50 min total
    - Val acc : ~60-70% (frame-level averaging, no temporal modeling)
    - Saved to: downstream/action_recognition/weights/tsn_ucf101_head.pth
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune TSN head on UCF101')
    parser.add_argument('--frames_root',
        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--save_path',
        default='downstream/action_recognition/weights/tsn_ucf101_head.pth')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--image_size',  type=int,   default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    os.makedirs('downstream/action_recognition/weights', exist_ok=True)
    os.makedirs('downstream/action_recognition/logs',    exist_ok=True)

    print("=" * 60)
    print("  TSN Head Fine-tuning on UCF101")
    print("=" * 60)
    print(f"  Device     : {args.device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"  Image size : {args.image_size}")
    print(f"  Save path  : {args.save_path}")
    print(f"  Backbone   : ResNet-50 FROZEN")
    print(f"  Head       : Linear(2048, 101) trainable")
    print("=" * 60)

    from downstream.action_recognition.models.tsn_recognizer import finetune_tsn_head

    finetune_tsn_head(
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