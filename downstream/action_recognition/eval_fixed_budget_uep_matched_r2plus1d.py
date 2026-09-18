#!/usr/bin/env python
"""Run the existing matched fixed-budget UEP evaluator with R(2+1)D.

The base evaluator still owns loading and validating the UEP checkpoint,
locked split, paired modes, channel RNG, power audits, metrics and McNemar
tests.  This adapter replaces only its frozen TSN classifier after the full
checkpoint has been loaded strictly.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn

from downstream.action_recognition import eval_fixed_budget_uep_matched as base
from downstream.action_recognition.models.r2plus1d_recognizer import (
    R2Plus1DRecognizer,
    load_r2plus1d_checkpoint,
    normalization_tensors,
)


_base_load_model = base.load_model
R2PLUS1D_CHECKPOINT = None


class NormalizedR2Plus1D(nn.Module):
    """Apply the locked Kinetics normalization before R(2+1)D inference."""

    def __init__(self, recognizer, device):
        super().__init__()
        self.recognizer = recognizer
        mean, std = normalization_tensors(device)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, clips):
        return self.recognizer((clips - self.mean) / self.std)


def load_model(args, device, evaluation_manifest_hash):
    # Construct the original architecture and strictly load every saved tensor,
    # including the training-time TSN.  Only then replace the evaluator.
    model, checkpoint_metadata = _base_load_model(
        args, device, evaluation_manifest_hash
    )

    recognizer = R2Plus1DRecognizer(pretrained=False, num_classes=101)
    recognizer_metadata = load_r2plus1d_checkpoint(
        recognizer, R2PLUS1D_CHECKPOINT
    )
    if recognizer_metadata.get("split_manifest_sha256") != evaluation_manifest_hash:
        raise RuntimeError("R(2+1)D/evaluation manifest mismatch")
    if recognizer_metadata.get("batchnorm_frozen") is not True:
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
        "r2plus1d_checkpoint_sha256": base.sha256_file(R2PLUS1D_CHECKPOINT),
        "r2plus1d_checkpoint_epoch": recognizer_metadata.get("epoch"),
        "r2plus1d_manifest_sha256": recognizer_metadata.get(
            "split_manifest_sha256"
        ),
        "r2plus1d_batchnorm_frozen": True,
    })
    return model, checkpoint_metadata


def main():
    global R2PLUS1D_CHECKPOINT
    default_checkpoint = (
        "downstream/action_recognition/weights/"
        "r2plus1d_ucf101_layer4_locked_split_best.pt"
    )
    if "--r2plus1d_ckpt" in sys.argv:
        index = sys.argv.index("--r2plus1d_ckpt")
        if index + 1 >= len(sys.argv):
            raise SystemExit("--r2plus1d_ckpt requires a path")
        R2PLUS1D_CHECKPOINT = sys.argv[index + 1]
        del sys.argv[index:index + 2]
    else:
        R2PLUS1D_CHECKPOINT = default_checkpoint
    base.load_model = load_model
    base.main()


if __name__ == "__main__":
    main()
