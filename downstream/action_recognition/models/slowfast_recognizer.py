# -*- coding: utf-8 -*-
"""
SlowFast R50 Action Recognizer.
- Backbone : SlowFast R50 pretrained on Kinetics-400 (pytorchvideo)
- Head     : Linear(2304, 101) fine-tuned on UCF101 (backbone FROZEN)

CNN-based (3D ResNet) — architecturally consistent with DeepJSCC encoder/decoder.

Install:
    pip install pytorchvideo av

Usage:
    # 1. Fine-tune head overnight on 3090 Ti:
    python downstream/action_recognition/finetune_slowfast.py --device cuda:1

    # 2. Use in pipeline:
    rec = SlowFastRecognizer(head_ckpt='downstream/action_recognition/weights/slowfast_ucf101_head.pth')
    idx, label, conf = rec.predict_from_gop(gop_tensor)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

WEIGHTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'weights')
UCF_CLASSIND = os.path.join(
    PROJECT_ROOT,
    'datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt'
)

# SlowFast preprocessing constants (Kinetics-400 training stats)
SLOWFAST_MEAN  = [0.45, 0.45, 0.45]
SLOWFAST_STD   = [0.225, 0.225, 0.225]
SIDE_SIZE      = 256
CROP_SIZE      = 256
NUM_FRAMES     = 32
SLOWFAST_ALPHA = 4        # slow = NUM_FRAMES // ALPHA = 8 frames
NUM_CLASSES_K400 = 400
NUM_CLASSES_UCF  = 101


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
# Preprocessing
# ---------------------------------------------------------------------------

def build_slowfast_transform():
    from torchvision.transforms import Compose, Lambda
    from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

    class PackPathway(nn.Module):
        def forward(self, frames: torch.Tensor) -> List[torch.Tensor]:
            fast = frames
            slow = torch.index_select(
                frames, 1,
                torch.linspace(0, frames.shape[1] - 1,
                               frames.shape[1] // SLOWFAST_ALPHA).long()
            )
            return [slow, fast]

    return Compose([
        UniformTemporalSubsample(NUM_FRAMES),
        Lambda(lambda x: x / 255.0),
        NormalizeVideo(mean=SLOWFAST_MEAN, std=SLOWFAST_STD),
        ShortSideScale(size=SIDE_SIZE),
        CenterCropVideo(CROP_SIZE),
        PackPathway(),
    ])


# ---------------------------------------------------------------------------
# SlowFast with replaceable UCF101 head
# ---------------------------------------------------------------------------

class SlowFastWithUCF101Head(nn.Module):
    """SlowFast R50 backbone (frozen) + Linear head for UCF101."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        print("[SlowFast] Loading backbone from torch.hub...")
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo', 'slowfast_r50',
            pretrained=pretrained, verbose=False,
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Replace final projection: K400 → UCF101
        # SlowFast R50 feature dim = 2048 (slow) + 256 (fast) = 2304
        self.head = nn.Linear(2304, NUM_CLASSES_UCF)
        print(f"[SlowFast] Head: Linear(2304, {NUM_CLASSES_UCF})")

    def forward(self, pathway_list: List[torch.Tensor]) -> torch.Tensor:
        orig_proj = self.backbone.blocks[-1].proj
        self.backbone.blocks[-1].proj = nn.Identity()
        with torch.set_grad_enabled(self.training):
            features = self.backbone(pathway_list)   # (B, 2304)
        self.backbone.blocks[-1].proj = orig_proj
        return self.head(features)                   # (B, 101)


# ---------------------------------------------------------------------------
# Main recognizer
# ---------------------------------------------------------------------------

class SlowFastRecognizer:
    """
    SlowFast R50 action recognizer for UCF101.

    Args:
        head_ckpt   : path to fine-tuned head .pth
                      If None, uses Kinetics-400 pretrained (400 classes)
        device      : 'cuda:0', 'cuda:1', 'cpu'
        ucf_classind: path to classInd.txt
    """

    def __init__(
        self,
        head_ckpt:    str = None,
        device:       str = 'cuda:0',
        ucf_classind: str = UCF_CLASSIND,
    ):
        self.device    = device
        self.transform = build_slowfast_transform()
        self.idx2label = load_ucf101_labels(ucf_classind)

        if head_ckpt and os.path.exists(head_ckpt):
            print(f"[SlowFast] Loading UCF101 head from {head_ckpt}")
            self.model = SlowFastWithUCF101Head(pretrained=True)
            self.model.head.load_state_dict(torch.load(head_ckpt, map_location=device))
            self.num_classes = NUM_CLASSES_UCF
        else:
            print("[SlowFast] No head_ckpt — using Kinetics-400 pretrained (400 classes)")
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo', 'slowfast_r50',
                pretrained=True, verbose=False,
            )
            self.num_classes = NUM_CLASSES_K400

        self.model = self.model.to(device)
        self.model.eval()

    def _frames_to_pathway(
        self, frames: Union[np.ndarray, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Convert frames to [slow, fast] pathway tensors."""
        if isinstance(frames, np.ndarray):
            t = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        else:
            t = frames.float()
            if t.max() <= 1.0:
                t = t * 255.0
            if t.shape[-1] == 3:              # (T, H, W, 3) → (T, 3, H, W)
                t = t.permute(0, 3, 1, 2)

        t       = t.permute(1, 0, 2, 3)      # (3, T, H, W)
        pathway = self.transform(t)           # [slow, fast]
        return [p.unsqueeze(0).to(self.device) for p in pathway]

    def predict_from_frames(
        self, frames: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[int, str, float]:
        """
        Args:
            frames: (T, H, W, 3) uint8 numpy  OR  (T, 3, H, W) float [0,1] tensor
        Returns:
            (class_idx, class_label, confidence)
        """
        pathway = self._frames_to_pathway(frames)
        with torch.no_grad():
            logits = self.model(pathway)
        scores    = logits.squeeze(0).softmax(0).cpu()
        class_idx = int(scores.argmax().item())
        conf      = float(scores[class_idx].item())
        label     = self.idx2label.get(class_idx, f"class_{class_idx}")
        return class_idx, label, conf

    def predict_from_gop(self, gop: torch.Tensor) -> Tuple[int, str, float]:
        """
        Args:
            gop: (N, 3, H, W) float [0,1] — single GoP, no batch dim
        """
        return self.predict_from_frames(gop)


# ---------------------------------------------------------------------------
# Fine-tuning function (called by finetune_slowfast.py)
# ---------------------------------------------------------------------------

def _build_pathways_batch(gops: torch.Tensor, transform) -> List[torch.Tensor]:
    """
    Build SlowFast [slow, fast] pathways for a batch of GoPs efficiently.
    Vectorized: applies transform per sample but stacks into batch tensors.

    Args:
        gops      : (B, N, 3, H, W) float [0, 1]
        transform : SlowFast transform pipeline

    Returns:
        [slow (B, 3, T/4, H, W), fast (B, 3, T, H, W)]
    """
    B = gops.shape[0]
    slow_list, fast_list = [], []
    for b in range(B):
        frames  = gops[b].permute(1, 0, 2, 3) * 255.0   # (3, N, H, W)
        pathway = transform(frames)                        # [slow, fast]
        slow_list.append(pathway[0])
        fast_list.append(pathway[1])
    return [torch.stack(slow_list), torch.stack(fast_list)]


def finetune_slowfast_head(
    frames_root:     str,
    annotation_path: str,
    save_path:       str   = None,
    epochs:          int   = 20,
    batch_size:      int   = 4,
    lr:              float = 1e-3,
    device:          str   = 'cuda:0',
    num_workers:     int   = 4,
    image_size:      int   = 128,
    log_every:       int   = 50,    # print batch progress every N batches
):
    """
    Fine-tune SlowFast head on UCF101. Backbone fully frozen.
    Only trains Linear(2304, 101).

    Key fix: num_workers=4 reduces CPU data loading bottleneck.
    Batch progress logged every `log_every` batches so you can
    see training is progressing without waiting for full epoch.
    """
    import torch.optim as optim
    from tqdm import tqdm
    from data.ucf101_dataloader import build_dataloaders

    if save_path is None:
        save_path = os.path.join(WEIGHTS_DIR, 'slowfast_ucf101_head.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[SlowFast Finetune] device={device}, epochs={epochs}, "
          f"batch={batch_size}, lr={lr}, num_workers={num_workers}")

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

    model     = SlowFastWithUCF101Head(pretrained=True).to(device)
    optimizer = optim.Adam(model.head.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    transform = build_slowfast_transform()
    best_acc  = 0.0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [train]", leave=False)
        for batch_idx, (gops, labels) in enumerate(pbar):
            B = gops.shape[0]
            optimizer.zero_grad()

            pathways = _build_pathways_batch(gops, transform)
            slow     = pathways[0].to(device)
            fast     = pathways[1].to(device)
            labels   = labels.to(device)

            logits = model([slow, fast])
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            correct  += (logits.argmax(1) == labels).sum().item()
            total    += B
            loss_sum += loss.item()

            # Live batch progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc' : f"{correct/total*100:.1f}%",
            })

        train_acc = correct / total * 100

        # --- Validate ---
        model.eval()
        val_total, val_correct = 0, 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [val]  ", leave=False)
        with torch.no_grad():
            for gops, labels in pbar_val:
                pathways = _build_pathways_batch(gops, transform)
                slow     = pathways[0].to(device)
                fast     = pathways[1].to(device)
                labels   = labels.to(device)
                preds    = model([slow, fast]).argmax(1)
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

    print(f"\n[SlowFast Finetune] Done. Best Val: {best_acc:.1f}% → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rec   = SlowFastRecognizer(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    dummy = np.random.randint(0, 255, (32, 256, 256, 3), dtype=np.uint8)
    idx, label, conf = rec.predict_from_frames(dummy)
    print(f"[SlowFast] idx={idx}, label={label}, conf={conf:.3f}")