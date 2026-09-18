# -*- coding: utf-8 -*-
"""
TSN (Temporal Segment Networks) Action Recognizer.
Uses ResNet-50 backbone with frame-level averaging.

Why TSN instead of SlowFast:
    SlowFast requires 32 frames minimum. Our GoP size is N=5.
    TSN is designed for sparse frame sampling — naturally handles any N.
    Each frame is passed through ResNet-50 independently, then predictions
    are averaged across all N frames. This is the correct approach for
    5-frame GoP evaluation.

Architecture:
    - Backbone: ResNet-50 pretrained on ImageNet (frozen)
    - Head    : Linear(2048, 101) fine-tuned on UCF101
    - Inference: average frame-level logits across N=5 frames, then softmax

Usage:
    # Fine-tune head on UCF101 (~30-60 min on RTX 3060):
    python downstream/action_recognition/models/tsn_recognizer.py --mode finetune

    # Test:
    python downstream/action_recognition/models/tsn_recognizer.py --mode test
"""

import os
import sys
import json
import random
import re
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Tuple
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

WEIGHTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'weights')
UCF_CLASSIND = os.path.join(
    PROJECT_ROOT,
    'datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt'
)

NUM_CLASSES_UCF = 101

# ImageNet normalization (ResNet-50 pretrained stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def load_ucf101_labels(classind_path: str) -> dict:
    idx2label = {}
    if os.path.exists(classind_path):
        with open(classind_path) as f:
            for line in f:
                idx, name = line.strip().split()
                idx2label[int(idx) - 1] = name
    return idx2label


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

def build_frame_transform(image_size: int = 128):
    """Build per-frame preprocessing transform."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# TSN Model
# ---------------------------------------------------------------------------

class TSNModel(nn.Module):
    """
    TSN with ResNet-50 backbone (frozen) + Linear head for UCF101.
    Frame-level predictions averaged across N frames.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES_UCF):
        super().__init__()

        # Load ResNet-50 backbone
        print("[TSN] Loading ResNet-50 backbone...")
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Remove final FC layer — keep feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = 2048

        # Freeze backbone
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Trainable head
        self.head = nn.Linear(feature_dim, num_classes)
        self.finetune_mode = 'head'
        print(f"[TSN] Head: Linear({feature_dim}, {num_classes}) — trainable")
        print(f"[TSN] Backbone: frozen")

    def forward_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single frame.
        Args:
            frame: (B, 3, H, W) normalized
        Returns:
            logits: (B, num_classes)
        """
        features = self.feature_extractor(frame)    # (B, 2048, 1, 1)
        features = features.flatten(1)              # (B, 2048)
        return self.head(features)                  # (B, num_classes)

    def train(self, mode: bool = True):
        """Train only the UCF101 head; keep frozen ResNet BatchNorm fixed.

        ``requires_grad=False`` does not stop BatchNorm running-stat updates.
        The old implementation called ``model.train()`` and consequently
        trained the head on UCF101-adapted BN statistics, then saved only the
        head and evaluated it with fresh ImageNet BN statistics.  This override
        removes that training/inference mismatch.
        """
        super().train(mode)
        self.feature_extractor.eval()
        self.head.train(mode)
        return self

    def configure_finetuning(self, mode: str = 'layer4') -> int:
        """Enable either head-only or ResNet layer4-plus-head adaptation."""
        if mode not in {'head', 'layer4'}:
            raise ValueError("finetune mode must be 'head' or 'layer4'")
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.head.parameters():
            parameter.requires_grad = True
        if mode == 'layer4':
            for parameter in self.feature_extractor[7].parameters():
                parameter.requires_grad = True
        self.finetune_mode = mode
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        TSN forward: average predictions across N frames.
        Args:
            frames: (B, N, 3, H, W) normalized
        Returns:
            logits: (B, num_classes) — averaged across frames
        """
        B, N, C, H, W = frames.shape
        # Process all frames at once: reshape to (B*N, 3, H, W)
        frames_flat = frames.reshape(B * N, C, H, W)
        logits_flat = self.forward_single_frame(frames_flat)  # (B*N, num_classes)
        logits = logits_flat.reshape(B, N, -1)                # (B, N, num_classes)
        return logits.mean(dim=1)                             # (B, num_classes) — TSN averaging


def load_tsn_checkpoint(model: TSNModel, checkpoint_path: str) -> dict:
    """Load a full corrected checkpoint or a legacy head-only state dict."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"TSN checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False
    )
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if int(checkpoint.get('class_count', 101)) != 101:
            raise ValueError('TSN checkpoint class_count is not 101')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return checkpoint

    # Backward compatibility: historical Linear(2048,101) head only.
    if isinstance(checkpoint, dict) and {'weight', 'bias'} <= set(checkpoint):
        model.head.load_state_dict(checkpoint, strict=True)
        return {'format': 'legacy_head_only', 'finetune': 'head'}
    raise KeyError(
        "Expected a full checkpoint containing 'model_state_dict' or a "
        "legacy head-only state dictionary"
    )


# ---------------------------------------------------------------------------
# TSN Recognizer
# ---------------------------------------------------------------------------

class TSNRecognizer:
    """
    TSN action recognizer for UCF101 — designed for 5-frame GoPs.

    Args:
        head_ckpt   : path to fine-tuned head .pth
        device      : 'cuda:0' or 'cpu'
        ucf_classind: path to classInd.txt
        image_size  : frame spatial size (must match training)
    """

    def __init__(
        self,
        head_ckpt:    str  = None,
        device:       str  = 'cuda:0',
        ucf_classind: str  = UCF_CLASSIND,
        image_size:   int  = 128,
    ):
        self.device     = device
        self.image_size = image_size
        self.transform  = build_frame_transform(image_size)
        self.idx2label  = load_ucf101_labels(ucf_classind)

        self.model = TSNModel(pretrained=True)

        metadata = load_tsn_checkpoint(self.model, head_ckpt)
        print(
            f"[TSN] Loaded {head_ckpt} | "
            f"finetune={metadata.get('finetune', 'unknown')}"
        )

        self.model = self.model.to(device)
        self.model.eval()

    def _preprocess_gop(self, gop: torch.Tensor) -> torch.Tensor:
        """
        Preprocess GoP tensor for TSN inference.

        Args:
            gop: (N, 3, H, W) float [0,1]  or  (B, N, 3, H, W)

        Returns:
            (B, N, 3, H, W) normalized tensor on device
        """
        if gop.dim() == 4:
            gop = gop.unsqueeze(0)       # (1, N, 3, H, W)

        B, N, C, H, W = gop.shape
        gop = gop.clamp(0, 1)

        # Apply ImageNet normalization per frame
        mean = torch.tensor(IMAGENET_MEAN, device=gop.device).view(1, 1, 3, 1, 1)
        std  = torch.tensor(IMAGENET_STD,  device=gop.device).view(1, 1, 3, 1, 1)
        gop_norm = (gop - mean) / std

        return gop_norm.to(self.device)

    def predict_from_gop(
        self,
        gop: torch.Tensor,
    ) -> Tuple[int, str, float]:
        """
        Predict action from a GoP tensor.

        Args:
            gop: (N, 3, H, W) float [0,1] — single GoP, no batch dim
                 OR (B, N, 3, H, W)

        Returns:
            (class_idx, class_label, confidence)
        """
        gop_norm = self._preprocess_gop(gop)

        with torch.no_grad():
            logits = self.model(gop_norm)        # (B, 101)
        scores    = logits[0].softmax(0).cpu()
        class_idx = int(scores.argmax().item())
        conf      = float(scores[class_idx].item())
        label     = self.idx2label.get(class_idx, f"class_{class_idx}")
        return class_idx, label, conf

    def predict_from_frames(
        self,
        frames: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[int, str, float]:
        """
        Args:
            frames: (N, H, W, 3) uint8 numpy  OR  (N, 3, H, W) float [0,1]
        """
        if isinstance(frames, np.ndarray):
            t = torch.from_numpy(frames).float() / 255.0
            if t.shape[-1] == 3:
                t = t.permute(0, 3, 1, 2)    # (N, 3, H, W)
        else:
            t = frames.float()
            if t.max() > 1.0:
                t = t / 255.0

        return self.predict_from_gop(t)


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def _group_aware_split_loader(full_train_loader, val_fraction: float, seed: int):
    """Split official training data without leaking related UCF clip groups."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be between 0 and 1')
    dataset = full_train_loader.dataset
    grouped = {}
    for index, (frame_paths, label) in enumerate(dataset.index):
        clip = os.path.basename(os.path.dirname(frame_paths[0]))
        group_name = re.sub(r'_c\d+$', '', clip)
        grouped.setdefault((int(label), group_name), []).append(index)

    by_class = {}
    for group_key in grouped:
        by_class.setdefault(group_key[0], []).append(group_key)

    rng = random.Random(seed)
    validation_groups = set()
    for group_keys in by_class.values():
        rng.shuffle(group_keys)
        count = max(1, round(len(group_keys) * val_fraction))
        count = min(count, len(group_keys) - 1)
        validation_groups.update(group_keys[:count])

    train_indices, val_indices = [], []
    for group_key, indices in grouped.items():
        target = val_indices if group_key in validation_groups else train_indices
        target.extend(indices)

    common = dict(
        batch_size=full_train_loader.batch_size,
        num_workers=full_train_loader.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, drop_last=False, **common
    )
    print(
        f"  Internal group-aware split: train={len(train_indices)}, "
        f"val={len(val_indices)}"
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate_tsn(model, loader, device, mean, std, criterion):
    model.eval()
    loss_sum = 0.0
    correct1 = correct5 = total = 0
    for gops, labels in loader:
        gops = gops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model((gops - mean) / std)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()
        total += labels.size(0)
    return {
        'loss': loss_sum / max(total, 1),
        'top1': 100.0 * correct1 / max(total, 1),
        'top5': 100.0 * correct5 / max(total, 1),
        'samples': total,
    }

def finetune_tsn_head(
    frames_root:     str,
    annotation_path: str,
    save_path:       str   = None,
    epochs:          int   = 10,
    batch_size:      int   = 16,
    lr:              float = 1e-3,
    layer4_lr:       float = 1e-4,
    finetune:        str   = 'layer4',
    device:          str   = 'cuda:0',
    num_workers:     int   = 4,
    image_size:      int   = 128,
    val_fraction:    float = 0.1,
    seed:            int   = 42,
):
    """
    Adapt TSN on UCF101 using either the head alone or layer4 plus head.
    Frozen stages and all BatchNorm running statistics remain fixed.

    Expected runtime: ~3-5 min/epoch on RTX 3060 → ~30-50 min for 10 epochs
    """
    import torch.optim as optim
    from tqdm import tqdm
    from data.ucf101_dataloader import build_dataloaders

    if save_path is None:
        save_path = os.path.join(WEIGHTS_DIR, 'tsn_ucf101_head.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[TSN Finetune] device={device}, epochs={epochs}, "
          f"batch={batch_size}, lr={lr}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    full_train_loader, test_loader = build_dataloaders(
        frames_root=frames_root,
        annotation_path=annotation_path,
        mode='gop',
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        gop_size=5,
        gops_per_clip=1,
        seed=seed,
    )
    train_loader, val_loader = _group_aware_split_loader(
        full_train_loader, val_fraction, seed
    )

    model = TSNModel(pretrained=True).to(device)
    trainable = model.configure_finetuning(finetune)
    parameter_groups = [{'params': model.head.parameters(), 'lr': lr}]
    if finetune == 'layer4':
        parameter_groups.insert(0, {
            'params': model.feature_extractor[7].parameters(),
            'lr': layer4_lr,
        })
    optimizer = optim.Adam(parameter_groups)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 1, 3, 1, 1)

    print(f"  Fine-tuning: {finetune} | trainable parameters: {trainable:,}")
    print(f"  Learning rates: layer4={layer4_lr:g}, head={lr:g}")
    best_acc = -1.0
    history = []

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total, correct1, correct5, loss_sum = 0, 0, 0, 0.0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1:02d}/{epochs} [train]", leave=False)
        for gops, labels in pbar:
            gops   = gops.to(device)            # (B, N, 3, H, W) float [0,1]
            labels = labels.to(device)

            # Normalize
            gops_norm = (gops - mean) / std

            optimizer.zero_grad()
            logits = model(gops_norm)           # (B, 101)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()
            total    += gops.shape[0]
            loss_sum += loss.item() * labels.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'Top-1': f"{correct1/total*100:.1f}%",
                'Top-5': f"{correct5/total*100:.1f}%",
            })

        train_loss = loss_sum / max(total, 1)
        train_top1 = 100.0 * correct1 / max(total, 1)
        train_top5 = 100.0 * correct5 / max(total, 1)
        val = _evaluate_tsn(model, val_loader, device, mean, std, criterion)
        record = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_loss,
            'train_top1': train_top1,
            'train_top5': train_top5,
            'val_loss': val['loss'],
            'val_top1': val['top1'],
            'val_top5': val['top5'],
        }
        history.append(record)
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"train loss={train_loss:.4f}, Top-1={train_top1:.2f}%, "
              f"Top-5={train_top5:.2f}% | val loss={val['loss']:.4f}, "
              f"Top-1={val['top1']:.2f}%, Top-5={val['top5']:.2f}%")

        if val['top1'] > best_acc:
            best_acc = val['top1']
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_top1': val['top1'],
                'val_top5': val['top5'],
                'finetune': finetune,
                'class_count': 101,
                'image_size': image_size,
                'gop_size': 5,
                'backbone': 'ResNet50_Weights.IMAGENET1K_V2',
                'backbone_batchnorm_frozen': True,
            }, save_path)
            print(f"  ✓ Best saved (val Top-1={best_acc:.2f}%) → {save_path}")

        scheduler.step()

    selected = load_tsn_checkpoint(model, save_path)
    test = _evaluate_tsn(model, test_loader, device, mean, std, criterion)
    metadata = {
        'best_internal_val_top1': best_acc,
        'official_test': test,
        'val_fraction': val_fraction,
        'seed': seed,
        'image_size': image_size,
        'gop_size': 5,
        'backbone': 'ResNet50_Weights.IMAGENET1K_V2',
        'backbone_batchnorm_frozen': True,
        'finetune': finetune,
        'layer4_lr': layer4_lr,
        'head_lr': lr,
        'selected_epoch': selected['epoch'],
        'history': history,
    }
    metadata_path = os.path.splitext(save_path)[0] + '.json'
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\n[TSN Fine-tune] Best internal-val Top-1: {best_acc:.2f}%")
    print(f"Official test Top-1: {test['top1']:.2f}%")
    print(f"Official test Top-5: {test['top5']:.2f}%")
    print(f"Head checkpoint: {save_path}")
    print(f"Metadata/log:   {metadata_path}")
    return save_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['finetune', 'test'], default='test')
    parser.add_argument('--frames_root',
        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--save_path',
        default='downstream/action_recognition/weights/tsn_ucf101_head.pth')
    parser.add_argument('--head_ckpt',
        default='downstream/action_recognition/weights/tsn_ucf101_head.pth')
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--layer4_lr',  type=float, default=1e-4)
    parser.add_argument('--finetune', choices=['head', 'layer4'], default='layer4')
    parser.add_argument('--device',     default='cuda:0')
    parser.add_argument('--num_workers',type=int,   default=4)
    parser.add_argument('--image_size', type=int,   default=128)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.mode == 'finetune':
        finetune_tsn_head(
            frames_root=args.frames_root,
            annotation_path=args.annotation_path,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            layer4_lr=args.layer4_lr,
            finetune=args.finetune,
            device=args.device,
            num_workers=args.num_workers,
            image_size=args.image_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    else:
        rec   = TSNRecognizer(head_ckpt=args.head_ckpt, device=args.device)
        dummy = torch.rand(5, 3, 128, 128)   # 5-frame GoP
        idx, label, conf = rec.predict_from_gop(dummy)
        print(f"[TSN] idx={idx}, label={label}, conf={conf:.3f}")
