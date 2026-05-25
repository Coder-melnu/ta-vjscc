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
    - Inference: average softmax scores across N=5 frames

Usage:
    # Fine-tune head on UCF101 (~30-60 min on RTX 3060):
    python downstream/action_recognition/models/tsn_recognizer.py --mode finetune

    # Test:
    python downstream/action_recognition/models/tsn_recognizer.py --mode test
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Tuple
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

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

        if head_ckpt and os.path.exists(head_ckpt):
            print(f"[TSN] Loading UCF101 head from {head_ckpt}")
            self.model.head.load_state_dict(
                torch.load(head_ckpt, map_location=device)
            )
        else:
            print("[TSN] No head_ckpt — using random head (fine-tune first!)")

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

def finetune_tsn_head(
    frames_root:     str,
    annotation_path: str,
    save_path:       str   = None,
    epochs:          int   = 10,
    batch_size:      int   = 16,
    lr:              float = 1e-3,
    device:          str   = 'cuda:0',
    num_workers:     int   = 4,
    image_size:      int   = 128,
):
    """
    Fine-tune TSN head on UCF101. Backbone fully frozen.
    Only trains Linear(2048, 101).

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

    train_loader, val_loader = build_dataloaders(
        frames_root=frames_root,
        annotation_path=annotation_path,
        mode='gop',
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        gop_size=5,
        gops_per_clip=1,
        seed=42,
    )

    model     = TSNModel(pretrained=True).to(device)
    optimizer = optim.Adam(model.head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # ImageNet normalization
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 1, 3, 1, 1)

    best_acc = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

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

            correct  += (logits.argmax(1) == labels).sum().item()
            total    += gops.shape[0]
            loss_sum += loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc' : f"{correct/total*100:.1f}%",
            })

        train_acc = correct / total * 100

        # --- Validate ---
        model.eval()
        val_total, val_correct = 0, 0

        pbar_val = tqdm(val_loader,
                        desc=f"Epoch {epoch+1:02d}/{epochs} [val]  ", leave=False)
        with torch.no_grad():
            for gops, labels in pbar_val:
                gops      = gops.to(device)
                labels    = labels.to(device)
                gops_norm = (gops - mean) / std
                preds     = model(gops_norm).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += gops.shape[0]
                pbar_val.set_postfix({'acc': f"{val_correct/val_total*100:.1f}%"})

        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"loss={loss_sum/len(train_loader):.4f} | "
              f"train={train_acc:.1f}% | val={val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.head.state_dict(), save_path)
            print(f"  ✓ Best saved ({best_acc:.1f}%) → {save_path}")

        scheduler.step()

    print(f"\n[TSN Finetune] Done. Best Val: {best_acc:.1f}% → {save_path}")
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
    parser.add_argument('--head_ckpt', default=None)
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--device',     default='cuda:0')
    parser.add_argument('--num_workers',type=int,   default=4)
    parser.add_argument('--image_size', type=int,   default=128)
    args = parser.parse_args()

    if args.mode == 'finetune':
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
    else:
        rec   = TSNRecognizer(head_ckpt=args.head_ckpt, device=args.device)
        dummy = torch.rand(5, 3, 128, 128)   # 5-frame GoP
        idx, label, conf = rec.predict_from_gop(dummy)
        print(f"[TSN] idx={idx}, label={label}, conf={conf:.3f}")