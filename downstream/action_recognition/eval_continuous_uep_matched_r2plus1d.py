#!/usr/bin/env python
"""Run an existing matched continuous-UEP evaluator with locked R(2+1)D.

Use --evaluator continuous for ordinary continuous UEP or --evaluator staged
for both staged variance-normalised and staged-recovery checkpoints.
"""

import importlib
import sys
from pathlib import Path

import torch.nn as nn

from downstream.action_recognition.models.r2plus1d_recognizer import (
    R2Plus1DRecognizer,
    load_r2plus1d_checkpoint,
    normalization_tensors,
)


R2PLUS1D_CHECKPOINT = None
BASE = None
BASE_LOAD_MODEL = None


class NormalizedR2Plus1D(nn.Module):
    """Apply locked Kinetics normalization before R(2+1)D inference."""

    def __init__(self, recognizer, device):
        super().__init__()
        self.recognizer = recognizer
        mean, std = normalization_tensors(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, clips):
        return self.recognizer((clips - self.mean) / self.std)


def load_model(args, device, evaluation_manifest_hash):
    model, checkpoint_metadata = BASE_LOAD_MODEL(
        args, device, evaluation_manifest_hash
    )
    recognizer = R2Plus1DRecognizer(pretrained=False, num_classes=101)
    metadata = load_r2plus1d_checkpoint(recognizer, R2PLUS1D_CHECKPOINT)
    if metadata.get("split_manifest_sha256") != evaluation_manifest_hash:
        raise RuntimeError("R(2+1)D/evaluation manifest mismatch")
    if metadata.get("batchnorm_frozen") is not True:
        raise RuntimeError("R(2+1)D checkpoint does not certify frozen BatchNorm")
    recognizer.to(device).eval()
    recognizer.freeze_batchnorm()
    for parameter in recognizer.parameters():
        parameter.requires_grad_(False)
    model.tsn = NormalizedR2Plus1D(recognizer, device).to(device).eval()
    model.eval()
    checkpoint_metadata.update({
        "action_evaluator": "torchvision_r2plus1d_18",
        "r2plus1d_checkpoint": str(Path(R2PLUS1D_CHECKPOINT)),
        "r2plus1d_checkpoint_sha256": BASE.sha256_file(R2PLUS1D_CHECKPOINT),
        "r2plus1d_checkpoint_epoch": metadata.get("epoch"),
        "r2plus1d_manifest_sha256": metadata.get("split_manifest_sha256"),
        "r2plus1d_batchnorm_frozen": True,
        "r2plus1d_normalization_applied": True,
    })
    return model, checkpoint_metadata


def pop_option(name, default=None):
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    value = sys.argv[index + 1]
    del sys.argv[index:index + 2]
    return value


def main():
    global R2PLUS1D_CHECKPOINT, BASE, BASE_LOAD_MODEL
    evaluator = pop_option("--evaluator")
    if evaluator not in {"continuous", "staged"}:
        raise SystemExit("--evaluator must be continuous or staged")
    R2PLUS1D_CHECKPOINT = pop_option(
        "--r2plus1d_ckpt",
        "downstream/action_recognition/weights/"
        "r2plus1d_ucf101_layer4_locked_split_best.pt",
    )
    module = (
        "downstream.action_recognition.eval_continuous_uep_matched"
        if evaluator == "continuous"
        else "downstream.action_recognition.eval_staged_varnorm_uep_matched"
    )
    BASE = importlib.import_module(module)
    BASE_LOAD_MODEL = BASE.load_model
    BASE.load_model = load_model
    BASE.main()


if __name__ == "__main__":
    main()
