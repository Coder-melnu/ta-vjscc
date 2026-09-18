#!/usr/bin/env python
"""Fine-tune R(2+1)D-18 on the locked, leakage-safe UCF101 split.

The official test set is never constructed. Selection uses internal-validation
Top-1, with validation loss as the tie-breaker. BatchNorm running statistics are
kept frozen, including while layer4 is fine-tuned.
"""

import argparse
import csv
import hashlib
import json
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
from downstream.action_recognition.models.r2plus1d_recognizer import (
    KINETICS_MEAN, KINETICS_STD, R2Plus1DRecognizer, normalization_tensors,
)

EXPECTED_MANIFEST_SHA256 = 'ea27a8557ef1e8f658f63c8f86f33721d6f94a1e58c657fa0a02c93e94befb7c'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames_root', default='datasets/UCF101Frames')
    p.add_argument('--annotation_path', default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    p.add_argument('--save_path', default='downstream/action_recognition/weights/r2plus1d_ucf101_layer4_locked_split_best.pt')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--step_size', type=int, default=8)
    p.add_argument('--gamma', type=float, default=0.1)
    p.add_argument('--finetune', choices=('head', 'layer4', 'full'), default='layer4')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--gop_size', type=int, default=5)
    p.add_argument('--gops_per_clip', type=int, default=1)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--smoke_batches', type=int, default=0, help='0 uses the full loaders; positive values are diagnostic only')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def topk_counts(logits, labels):
    top5 = logits.topk(5, dim=1).indices
    return top5[:, 0].eq(labels).sum().item(), top5.eq(labels[:, None]).any(dim=1).sum().item()


@torch.no_grad()
def evaluate(model, loader, device, mean, std, criterion, limit=0):
    model.eval(); loss_sum = correct1 = correct5 = total = 0
    for batch_index, (gops, labels) in enumerate(loader):
        if limit and batch_index >= limit: break
        gops, labels = gops.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model((gops - mean) / std)
        loss = criterion(logits, labels)
        c1, c5 = topk_counts(logits, labels); batch = labels.size(0)
        loss_sum += loss.item() * batch; correct1 += c1; correct5 += c5; total += batch
    if not total: raise RuntimeError('Empty validation loader')
    return {'loss': loss_sum/total, 'top1': 100*correct1/total, 'top5': 100*correct5/total, 'samples': total}


def payload(model, optimizer, scheduler, epoch, metrics, args, manifest_hash):
    return {
        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch,
        'val_loss': metrics['loss'], 'val_top1': metrics['top1'], 'val_top5': metrics['top5'],
        'architecture': 'torchvision_r2plus1d_18', 'pretraining': 'R2Plus1D_18_Weights.KINETICS400_V1',
        'finetune': args.finetune, 'class_count': 101, 'image_size': args.image_size,
        'gop_size': args.gop_size, 'gops_per_clip': args.gops_per_clip,
        'input_layout_external': 'B,T,C,H,W', 'input_layout_model': 'B,C,T,H,W',
        'normalization_mean': KINETICS_MEAN, 'normalization_std': KINETICS_STD,
        'batchnorm_frozen': True, 'split_manifest_sha256': manifest_hash,
        'official_test_used': False, 'config': vars(args),
    }


def main():
    args = parse_args()
    if (args.image_size, args.gop_size, args.gops_per_clip, args.seed, args.val_fraction) != (128, 5, 1, 42, 0.1):
        raise ValueError('Locked protocol requires image_size=128, gop_size=5, gops_per_clip=1, seed=42, val_fraction=0.1')
    save_path = Path(args.save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists(): raise FileExistsError(f'Refusing to overwrite {save_path}')
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=args.frames_root, annotation_path=args.annotation_path, image_size=args.image_size,
        batch_size=args.batch_size, num_workers=args.num_workers, gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip, val_fraction=args.val_fraction, seed=args.seed,
        enforce_locked_counts=True,
    )
    if len(train_loader.dataset) != 8409 or len(val_loader.dataset) != 1128: raise RuntimeError('Wrong locked split counts')
    if manifest_hash != EXPECTED_MANIFEST_SHA256: raise RuntimeError(f'Wrong locked manifest: {manifest_hash}')
    save_path.with_suffix('.split_manifest.json').write_text(json.dumps(manifest, indent=2))
    save_path.with_suffix('.split_manifest.sha256').write_text(manifest_hash + '\n')

    model = R2Plus1DRecognizer(pretrained=True, num_classes=101).to(device)
    model.configure_finetuning(args.finetune); model.freeze_batchnorm()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss(); mean, std = normalization_tensors(device)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    print(f'Locked split: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}')
    print(f'Split manifest SHA-256: {manifest_hash}')
    print(f'Trainable parameters: {trainable:,} ({args.finetune})')
    print('Official test constructed: false')

    history=[]; best_top1=-1.0; best_loss=float('inf'); best_epoch=None
    for epoch in range(1, args.epochs + 1):
        started=time.time(); model.train(); model.freeze_batchnorm()
        loss_sum=correct1=correct5=total=0
        progress=tqdm(train_loader, desc=f'epoch {epoch:02d}/{args.epochs}')
        for batch_index, (gops, labels) in enumerate(progress):
            if args.smoke_batches and batch_index >= args.smoke_batches: break
            gops, labels=gops.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == 'cuda'):
                logits=model((gops-mean)/std); loss=criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            c1,c5=topk_counts(logits.detach(),labels); batch=labels.size(0)
            loss_sum+=loss.item()*batch; correct1+=c1; correct5+=c5; total+=batch
            progress.set_postfix(loss=f'{loss_sum/total:.4f}', top1=f'{100*correct1/total:.2f}')
        val=evaluate(model,val_loader,device,mean,std,criterion,args.smoke_batches)
        record={'epoch':epoch,'lr':optimizer.param_groups[0]['lr'],'train_loss':loss_sum/total,
                'train_top1':100*correct1/total,'train_top5':100*correct5/total,
                'val_loss':val['loss'],'val_top1':val['top1'],'val_top5':val['top5'],
                'val_samples':val['samples'],'seconds':time.time()-started}
        history.append(record); print(json.dumps(record))
        better=val['top1']>best_top1 or (val['top1']==best_top1 and val['loss']<best_loss)
        if better:
            best_top1,best_loss,best_epoch=val['top1'],val['loss'],epoch
            torch.save(payload(model,optimizer,scheduler,epoch,val,args,manifest_hash),save_path)
            print(f'Best checkpoint saved: epoch={epoch}, val_top1={best_top1:.4f}%')
        scheduler.step()

    selected=torch.load(save_path,map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model_state_dict'],strict=True); model.to(device).eval()
    reloaded=evaluate(model,val_loader,device,mean,std,criterion,args.smoke_batches)
    history_path=save_path.with_suffix('.history.csv')
    with history_path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    summary={'status':'smoke_test' if args.smoke_batches else 'completed','best_epoch':best_epoch,
             'best_internal_val_top1':best_top1,'best_internal_val_loss':best_loss,'best_reloaded':reloaded,
             'train_clips':len(train_loader.dataset),'validation_clips':len(val_loader.dataset),
             'split_manifest_sha256':manifest_hash,'best_checkpoint_sha256':hashlib.sha256(save_path.read_bytes()).hexdigest(),
             'architecture':'torchvision_r2plus1d_18','finetune':args.finetune,'batchnorm_frozen':True,
             'official_test_used':False,'created_utc':datetime.now(timezone.utc).isoformat(),'history':history}
    save_path.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
