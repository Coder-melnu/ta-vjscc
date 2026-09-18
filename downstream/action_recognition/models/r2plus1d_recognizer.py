"""Locked R(2+1)D-18 recognizer utilities for UCF101 5-frame GoPs."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class R2Plus1DRecognizer(nn.Module):
    """R(2+1)D-18 accepting repository-native ``[B,T,C,H,W]`` clips."""

    def __init__(self, pretrained=True, num_classes=101):
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.network = r2plus1d_18(weights=weights)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

    def forward(self, clips):
        if clips.ndim != 5 or clips.shape[2] != 3:
            raise ValueError(f'Expected [B,T,3,H,W], received {tuple(clips.shape)}')
        # TorchVision video models require [B,C,T,H,W].
        return self.network(clips.permute(0, 2, 1, 3, 4).contiguous())

    def configure_finetuning(self, mode='layer4'):
        if mode not in {'head', 'layer4', 'full'}:
            raise ValueError(f'Unknown fine-tuning mode: {mode}')
        for parameter in self.parameters():
            parameter.requires_grad_(mode == 'full')
        if mode in {'head', 'layer4'}:
            for parameter in self.network.fc.parameters():
                parameter.requires_grad_(True)
        if mode == 'layer4':
            for parameter in self.network.layer4.parameters():
                parameter.requires_grad_(True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_batchnorm(self):
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)


def normalization_tensors(device):
    mean = torch.tensor(KINETICS_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(KINETICS_STD, device=device).view(1, 1, 3, 1, 1)
    return mean, std


def load_r2plus1d_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise KeyError('Expected checkpoint containing model_state_dict')
    if checkpoint.get('architecture') != 'torchvision_r2plus1d_18':
        raise RuntimeError('Checkpoint architecture is not torchvision_r2plus1d_18')
    if checkpoint.get('class_count') != 101:
        raise RuntimeError('Checkpoint does not contain a 101-class evaluator')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    metadata = dict(checkpoint)
    metadata.pop('model_state_dict', None)
    metadata['checkpoint_path'] = str(Path(checkpoint_path))
    return metadata
