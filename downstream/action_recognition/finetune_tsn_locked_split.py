#!/usr/bin/env python
"""Fine-tune the corrected TSN head on the locked VideoJSCC split.

The official UCF101 test set is not constructed. The ResNet-50 backbone and
all BatchNorm state remain frozen. Best selection uses internal-validation
Top-1, with validation loss as the tie-breaker.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from data.ucf101_train_val_test import build_train_val_dataloaders
from downstream.action_recognition.models.tsn_recognizer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TSNModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames_root', default='datasets/UCF101Frames')
    parser.add_argument(
        '--annotation_path',
        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist',
    )
    parser.add_argument(
        '--save_path',
        default='downstream/action_recognition/weights/tsn_ucf101_head_locked_split_best.pt',
    )
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--gop_size', type=int, default=5)
    parser.add_argument('--gops_per_clip', type=int, default=1)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, mean, std, criterion):
    model.eval()
    loss_sum = correct1 = correct5 = total = 0
    for gops, labels in loader:
        gops = gops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model((gops - mean) / std)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        top5 = logits.topk(5, dim=1).indices
        correct1 += top5[:, 0].eq(labels).sum().item()
        correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
        total += labels.size(0)
    if total == 0:
        raise RuntimeError('Empty validation loader')
    return {
        'loss': loss_sum / total,
        'top1': 100.0 * correct1 / total,
        'top5': 100.0 * correct5 / total,
        'samples': total,
    }


def checkpoint_payload(model, optimizer, scheduler, epoch, metrics, args, manifest_hash):
    return {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'val_loss': metrics['loss'],
        'val_top1': metrics['top1'],
        'val_top5': metrics['top5'],
        'finetune': 'head',
        'class_count': 101,
        'image_size': args.image_size,
        'gop_size': args.gop_size,
        'gops_per_clip': args.gops_per_clip,
        'backbone': 'ResNet50_Weights.IMAGENET1K_V2',
        'backbone_batchnorm_frozen': True,
        'split_manifest_sha256': manifest_hash,
        'config': vars(args),
    }


def main():
    args = parse_args()
    if args.gop_size != 5:
        raise ValueError('The locked TSN protocol requires gop_size=5')
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {save_path}')

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=args.frames_root,
        annotation_path=args.annotation_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip,
        val_fraction=args.val_fraction,
        seed=args.seed,
        enforce_locked_counts=True,
    )
    if len(train_loader.dataset) != 8409 or len(val_loader.dataset) != 1128:
        raise RuntimeError('Locked TSN split must contain 8409 train and 1128 validation clips')

    manifest_path = save_path.with_suffix('.split_manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    save_path.with_suffix('.split_manifest.sha256').write_text(manifest_hash + '\n')

    model = TSNModel(pretrained=True, num_classes=101).to(device)
    trainable = model.configure_finetuning('head')
    optimizer = torch.optim.Adam(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )
    criterion = nn.CrossEntropyLoss()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    print(f'Locked split: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}')
    print(f'Split manifest SHA-256: {manifest_hash}')
    print(f'Trainable parameters: {trainable:,} (head only)')
    print('Official test constructed: false')

    history = []
    best_top1 = -1.0
    best_loss = float('inf')
    best_epoch = None
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        loss_sum = correct1 = correct5 = total = 0
        progress = tqdm(train_loader, desc=f'epoch {epoch:02d}/{args.epochs}')
        for gops, labels in progress:
            gops = gops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model((gops - mean) / std)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            top5 = logits.topk(5, dim=1).indices
            batch = labels.size(0)
            loss_sum += loss.item() * batch
            correct1 += top5[:, 0].eq(labels).sum().item()
            correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
            total += batch
            progress.set_postfix(loss=f'{loss_sum/total:.4f}', top1=f'{100*correct1/total:.2f}')

        val = evaluate(model, val_loader, device, mean, std, criterion)
        record = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': loss_sum / total,
            'train_top1': 100.0 * correct1 / total,
            'train_top5': 100.0 * correct5 / total,
            'val_loss': val['loss'],
            'val_top1': val['top1'],
            'val_top5': val['top5'],
            'seconds': time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record))

        better = val['top1'] > best_top1 or (
            val['top1'] == best_top1 and val['loss'] < best_loss
        )
        if better:
            best_top1, best_loss, best_epoch = val['top1'], val['loss'], epoch
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, epoch, val, args, manifest_hash
                ),
                save_path,
            )
            print(f'Best checkpoint saved: epoch={epoch}, val_top1={best_top1:.4f}%')
        scheduler.step()

    last_path = save_path.with_name(save_path.stem.replace('_best', '_last') + save_path.suffix)
    final_val = evaluate(model, val_loader, device, mean, std, criterion)
    torch.save(
        checkpoint_payload(
            model, optimizer, scheduler, args.epochs, final_val, args, manifest_hash
        ),
        last_path,
    )

    selected = torch.load(save_path, map_location='cpu', weights_only=False)
    if selected.get('split_manifest_sha256') != manifest_hash:
        raise RuntimeError('Saved best checkpoint has the wrong split-manifest hash')
    model.load_state_dict(selected['model_state_dict'], strict=True)
    model.to(device).eval()
    best_reloaded = evaluate(model, val_loader, device, mean, std, criterion)

    history_path = save_path.with_suffix('.history.csv')
    with history_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        'status': 'completed',
        'best_epoch': best_epoch,
        'best_internal_val_top1': best_top1,
        'best_internal_val_loss': best_loss,
        'best_reloaded': best_reloaded,
        'train_clips': len(train_loader.dataset),
        'validation_clips': len(val_loader.dataset),
        'split_manifest_sha256': manifest_hash,
        'best_checkpoint_sha256': hashlib.sha256(save_path.read_bytes()).hexdigest(),
        'backbone_batchnorm_frozen': True,
        'finetune': 'head',
        'official_test_used': False,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'history': history,
    }
    save_path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
