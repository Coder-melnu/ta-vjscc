"""Five-frame TSM-ResNet-50 for UCF101.

The module is checkpoint-compatible with the official MIT Han Lab RGB
TSM-ResNet-50 model.  Temporal shift has no learned parameters, so the
Kinetics-400 checkpoint trained with eight segments can be instantiated with
five segments and then fine-tuned on the five-frame TA-VJSCC GoPs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.request import urlretrieve

import torch
from torch import nn
from torchvision.models import resnet50


KINETICS_TSM_URL = (
    "https://hanlab18.mit.edu/projects/tsm/models/"
    "TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth"
)


class TemporalShift(nn.Module):
    """Parameter-free channel shift used by the official TSM implementation."""

    def __init__(self, net: nn.Module, num_segments: int, fold_div: int = 8):
        super().__init__()
        self.net = net
        self.num_segments = num_segments
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = x.shape
        if nt % self.num_segments != 0:
            raise ValueError(
                f"Frame batch ({nt}) is not divisible by num_segments "
                f"({self.num_segments}). Expected input shaped (B,T,C,H,W)."
            )

        batch = nt // self.num_segments
        x = x.view(batch, self.num_segments, channels, height, width)
        fold = channels // self.fold_div

        shifted = torch.zeros_like(x)
        shifted[:, :-1, :fold] = x[:, 1:, :fold]
        shifted[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        shifted[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return self.net(shifted.view(nt, channels, height, width))


def _insert_temporal_shift(
    backbone: nn.Module, num_segments: int, fold_div: int = 8
) -> None:
    """Match the official ``shift_place=blockres`` ResNet-50 layout."""

    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(backbone, stage_name)
        for block in stage.children():
            block.conv1 = TemporalShift(
                block.conv1, num_segments=num_segments, fold_div=fold_div
            )


class TSMResNet50(nn.Module):
    """RGB TSM-ResNet-50 with average consensus over five frames."""

    def __init__(
        self,
        num_classes: int = 101,
        num_segments: int = 5,
        dropout: float = 0.8,
        shift_div: int = 8,
    ):
        super().__init__()
        self.num_segments = num_segments

        # weights=None: the official Kinetics checkpoint supplies all weights.
        self.base_model = resnet50(weights=None)
        _insert_temporal_shift(self.base_model, num_segments, shift_div)

        feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Dropout(p=dropout)
        self.new_fc = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.new_fc.weight, 0, 0.001)
        nn.init.zeros_(self.new_fc.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(
                f"Expected (B,T,C,H,W), received shape {tuple(frames.shape)}"
            )
        batch, segments, channels, height, width = frames.shape
        if segments != self.num_segments:
            raise ValueError(
                f"Model uses {self.num_segments} segments but input has {segments}."
            )

        features = self.base_model(
            frames.reshape(batch * segments, channels, height, width)
        )
        logits = self.new_fc(features).view(batch, segments, -1)
        return logits.mean(dim=1)


def download_kinetics_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading official Kinetics-400 TSM checkpoint to {path} ...")
        urlretrieve(KINETICS_TSM_URL, path)
    return path


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict mapping.")

    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        state_dict[key] = value
    return state_dict


def load_kinetics_backbone(
    model: TSMResNet50, checkpoint_path: str | Path
) -> Tuple[Iterable[str], Iterable[str]]:
    """Load all Kinetics weights except the incompatible 400-class head."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    state_dict.pop("new_fc.weight", None)
    state_dict.pop("new_fc.bias", None)

    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"new_fc.weight", "new_fc.bias"}
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"Missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return incompatible.missing_keys, incompatible.unexpected_keys


def configure_trainable_layers(model: TSMResNet50, mode: str) -> int:
    """Select head-only, layer4+head, or full-network fine-tuning."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode == "head":
        modules = (model.new_fc,)
    elif mode == "layer4":
        modules = (model.base_model.layer4, model.new_fc)
    elif mode == "all":
        modules = (model,)
    else:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_batch_norm_statistics(model: nn.Module) -> None:
    """Keep pretrained BN statistics stable for small video batches."""

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
