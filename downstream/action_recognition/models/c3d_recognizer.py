# -*- coding: utf-8 -*-
"""
C3D Action Recognizer — pretrained on Sports1M, fine-tuned on UCF101.
Loaded via MMAction2. No fine-tuning needed.

Install:
    pip install openmim
    mim install mmengine mmcv mmaction2

UCF101 Top-1: ~52.8% — used as lightweight CNN baseline.
"""

import os
import sys
import torch
import numpy as np
import urllib.request
from typing import Union, List, Tuple

# Project root: ta-vjscc/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

try:
    from mmaction.apis import inference_recognizer, init_recognizer
    MMACTION_AVAILABLE = True
except ImportError:
    MMACTION_AVAILABLE = False
    print("[C3D] MMAction2 not installed. Run: pip install openmim && mim install mmengine mmcv mmaction2")

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

C3D_CONFIG_URL = (
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/"
    "recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py"
)
C3D_CKPT_URL = (
    "https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/"
    "c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/"
    "c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth"
)

# Saved under downstream/action_recognition/weights/
WEIGHTS_DIR      = os.path.join(os.path.dirname(__file__), '..', 'weights')
C3D_CONFIG_LOCAL = os.path.join(WEIGHTS_DIR, 'c3d_ucf101.py')
C3D_CKPT_LOCAL   = os.path.join(WEIGHTS_DIR, 'c3d_ucf101.pth')

UCF_CLASSIND = os.path.join(
    PROJECT_ROOT,
    'datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt'
)


def _download_if_missing(url: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"[C3D] Downloading {os.path.basename(local_path)} ...")
        urllib.request.urlretrieve(url, local_path)
        print(f"[C3D] Saved to {local_path}")


# ---------------------------------------------------------------------------
# C3D Recognizer
# ---------------------------------------------------------------------------

class C3DRecognizer:
    """
    C3D pretrained Sports1M → fine-tuned UCF101 via MMAction2.
    No fine-tuning needed — direct UCF101 pretrained weights.

    Args:
        config_path  : MMAction2 config .py  (auto-downloaded if None)
        ckpt_path    : checkpoint .pth        (auto-downloaded if None)
        device       : 'cuda:0' or 'cpu'
        ucf_classind : path to classInd.txt
    """

    def __init__(
        self,
        config_path:  str = None,
        ckpt_path:    str = None,
        device:       str = 'cuda:0',
        ucf_classind: str = UCF_CLASSIND,
    ):
        assert MMACTION_AVAILABLE, "MMAction2 not installed."

        config_path = config_path or C3D_CONFIG_LOCAL
        ckpt_path   = ckpt_path   or C3D_CKPT_LOCAL

        _download_if_missing(C3D_CONFIG_URL, config_path)
        _download_if_missing(C3D_CKPT_URL,   ckpt_path)

        self.device = device
        self.model  = init_recognizer(config_path, ckpt_path, device=device)
        self.model.eval()

        # 0-based idx → class name
        self.idx2label = {}
        if os.path.exists(ucf_classind):
            with open(ucf_classind) as f:
                for line in f:
                    idx, name = line.strip().split()
                    self.idx2label[int(idx) - 1] = name

    def predict_from_video(self, video_path: str) -> Tuple[int, str, float]:
        """Run inference on a video file."""
        result    = inference_recognizer(self.model, video_path)
        scores    = result.pred_scores.item.cpu()
        class_idx = int(scores.argmax().item())
        conf      = float(scores.softmax(0)[class_idx].item())
        label     = self.idx2label.get(class_idx, f"class_{class_idx}")
        return class_idx, label, conf

    def predict_from_frames(
        self,
        frames:   Union[np.ndarray, torch.Tensor],
        fps:      int = 25,
        tmp_path: str = "/tmp/c3d_tmp.mp4",
    ) -> Tuple[int, str, float]:
        """
        Run inference on a frame sequence.

        Args:
            frames: (T, H, W, 3) uint8 numpy  OR  (T, 3, H, W) float [0,1] tensor
        """
        import cv2

        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu()
            if frames.shape[1] == 3:
                frames = frames.permute(0, 2, 3, 1)
            frames = (frames * 255).clamp(0, 255).numpy().astype(np.uint8)

        T, H, W, _ = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        return self.predict_from_video(tmp_path)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rec = C3DRecognizer(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    print("[C3D] Model loaded OK")
    dummy = np.random.randint(0, 255, (16, 112, 112, 3), dtype=np.uint8)
    idx, label, conf = rec.predict_from_frames(dummy)
    print(f"[C3D] Prediction: idx={idx}, label={label}, conf={conf:.3f}")