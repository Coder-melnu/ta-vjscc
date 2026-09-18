# -*- coding: utf-8 -*-
"""
Task-Aware Video DeepJSCC (TA-VideoJSCC) — direct joint fine-tuning.

Extends VideoJSCC with:
    1. TaskAwareSelector between encoder and channel
    2. Re-normalisation after masking to restore power constraint  [FIX Bug #3]
    3. Frozen TSN for task loss — gradients flow through x_hat into decoder/encoder
    4. joint_loss() returns (loss, info, x_refined, logits) to avoid double
       forward pass in evaluate_epoch                              [FIX Bug #1]
    5. Selector active for every optimisation step. The reconstruction-only
       VideoJSCC checkpoint provides the warm start; no reconstruction stage
       is repeated here.

Pipeline:
    x (B,N,3,H,W)
    → encoder        → z (B*N, 2c, H', W')  [power-normalised inside encoder]
    → selector       → z_masked (B*N, 2c, H', W')  [bypassed in Stage 1]
    → re-normalise   → z_norm   (B*N, 2c, H', W')  [restore power constraint]
    → channel        → z_rx
    → decoder        → x_hat (B*N, 3, H, W)
    → temporal       → x_refined (B, N, 3, H, W)
    → FrozenTSN      → logits (B, 101)  [frozen weights, grad flows through frames]
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.jscc import DeepJSCC
from model.temporal import TemporalFusionModule
from model.importance import TaskAwareSelector


# ---------------------------------------------------------------------------
# Frozen TSN wrapper
# ---------------------------------------------------------------------------

class FrozenTSN(nn.Module):
    """
    ResNet-50 backbone + fine-tuned Linear head, both fully frozen.

    Gradients flow backward through reconstructed frames (x_refined) into
    the decoder, channel, and encoder — NOT through TSN weights.

    Note: during training, joint_loss() must NOT be wrapped in torch.no_grad(),
    otherwise the task gradient signal is lost. evaluate_epoch wraps in
    no_grad intentionally (accuracy only, no parameter update needed).
    """

    def __init__(self, head_ckpt: str, device: str = 'cuda:0'):
        super().__init__()

        from downstream.action_recognition.models.tsn_recognizer import (
            TSNModel, load_tsn_checkpoint,
        )
        self.model = TSNModel(pretrained=True, num_classes=101)
        metadata = load_tsn_checkpoint(self.model, head_ckpt)
        self.checkpoint_metadata = metadata
        print(
            f"[FrozenTSN] Loaded {head_ckpt} | "
            f"finetune={metadata.get('finetune', 'unknown')}"
        )

        # Freeze all TSN weights
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation buffers (move to correct device with .to())
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        self.eval()

    def train(self, mode: bool = True):
        """Keep the frozen ResNet and its BatchNorm statistics in eval mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (B, N, 3, H, W) float [0,1]
        Returns:
            logits : (B, 101)
        """
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(
                f"Expected frames shaped (B,N,3,H,W), got {tuple(frames.shape)}"
            )
        frames_norm = (frames.clamp(0, 1) - self.mean) / self.std
        return self.model(frames_norm)


# ---------------------------------------------------------------------------
# TA-VideoJSCC
# ---------------------------------------------------------------------------

class TAVideoJSCC(nn.Module):
    """
    Task-Aware Video DeepJSCC for direct joint fine-tuning.

    Args:
        c             : encoder bottleneck depth (complex symbols per spatial location)
        channel_type  : 'AWGN' or 'Rayleigh'
        snr           : channel SNR in dB
        P             : transmit power constraint (default 1.0)
        n_frames      : GoP size (default 5)
        hidden_dim    : TemporalFusionModule hidden channels
        tsn_head_ckpt : path to tsn_ucf101_head.pth
        lambda_task   : task loss weight   (mentor recipe default: 0.001)
        lambda_recon  : reconstruction loss weight (default: 1.0)
        lambda_rate   : rate loss weight   (memo default: 0.01)
        tau           : initial Gumbel-Softmax temperature
        device        : device string (for TSN weight loading)
    """

    def __init__(
        self,
        c:             int,
        channel_type:  str   = 'AWGN',
        snr:           float = None,
        P:             float = 1.0,
        n_frames:      int   = 5,
        hidden_dim:    int   = 16,
        tsn_head_ckpt: str   = None,
        lambda_task:   float = 0.001,
        lambda_recon:  float = 1.0,
        lambda_rate:   float = 0.01,
        tau:           float = 1.0,
        device:        str   = 'cuda:0',
    ):
        super().__init__()

        self.n_frames     = n_frames
        self.lambda_task  = lambda_task
        self.lambda_recon = lambda_recon
        self.lambda_rate  = lambda_rate

        # Core JSCC (shared across all frames in GoP)
        self.jscc     = DeepJSCC(c=c, channel_type=channel_type, snr=snr, P=P)
        self.temporal = TemporalFusionModule(channels=3, hidden_dim=hidden_dim)

        # Task-aware selector (latent has 2c channels after encoder)
        self.selector = TaskAwareSelector(channels=2 * c, tau=tau)
        self.selector_mode = 'learned'
        self.min_soft_weight = 0.05
        self.random_mask_seed = 1729
        self.last_power_audit = {}
        self.last_spatial_power = None

        # Warm-start at p(keep)=0.9. The existing selector represents the two
        # classes as [-a,+a], so p(keep)=sigmoid(2a).
        final_conv = self.selector.scorer.net[-1]
        if not isinstance(final_conv, nn.Conv2d):
            raise TypeError('Unexpected ImportanceScorer output layer')
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, 0.5 * math.log(0.9 / 0.1))

        # Frozen TSN
        if tsn_head_ckpt is not None:
            self.tsn = FrozenTSN(head_ckpt=tsn_head_ckpt, device=device)
        else:
            self.tsn = None
            print("[TAVideoJSCC] No TSN ckpt provided — task loss will be zero")

    # ------------------------------------------------------------------
    # Selector controls
    # ------------------------------------------------------------------

    def set_selector_mode(self, mode: str):
        """Choose bypass, exact all-ones, learned, or matched-random mode."""
        valid = {'bypass', 'all_ones', 'learned', 'random'}
        if mode not in valid:
            raise ValueError(f"selector mode must be one of {sorted(valid)}")
        self.selector_mode = mode

    @staticmethod
    def _renormalize_to_reference(z_weighted, z_reference, eps=1e-12):
        """Restore each sample's pre-selector energy with stable arithmetic."""
        dims = tuple(range(1, z_weighted.ndim))
        reference_energy = z_reference.square().sum(dim=dims, keepdim=True)
        weighted_energy = z_weighted.square().sum(dim=dims, keepdim=True)
        scale = torch.sqrt(reference_energy / weighted_energy.clamp_min(eps))
        return z_weighted * scale

    def _selector_weights(self, z):
        """Return nonzero soft weights; deterministic during evaluation."""
        logits = self.selector.scorer(z)
        # True soft weighting: no Gumbel sampling and no hard decisions.
        # The zero-initialized final scorer and positive bias initially produce
        # an approximately uniform p(keep)=0.9 map. Uniform scaling is cancelled
        # by energy normalization, preserving the pretrained VideoJSCC output.
        raw = torch.sigmoid(2.0 * logits / self.selector.tau)
        weights = self.min_soft_weight + (1.0 - self.min_soft_weight) * raw

        if self.selector_mode == 'random':
            # Same weight values and mean, but randomized spatial placement.
            flat = weights.flatten(1)
            permuted = []
            for row_index, row in enumerate(flat):
                generator = torch.Generator(device=row.device)
                generator.manual_seed(self.random_mask_seed + row_index)
                permutation = torch.randperm(
                    row.numel(), device=row.device, generator=generator
                )
                permuted.append(row[permutation])
            flat = torch.stack(permuted)
            weights = flat.view_as(weights)
        return weights, logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass with task-aware selection (or an explicit audit mode).

        Args:
            x : (B, N, 3, H, W) — GoP of N frames, float [0,1]
        Returns:
            x_refined : (B, N, 3, H, W) — reconstructed frames
            mask      : (B*N, 1, H', W') — selection mask per frame
                        (all-ones only in bypass/audit mode)
            logits    : (B*N, 1, H', W') — raw importance logits per frame
                        (all-zeros only in bypass/audit mode)
        """
        B, N, C, H, W = x.shape
        assert N == self.n_frames, f"Expected {self.n_frames} frames, got {N}"

        x_flat = x.view(B * N, C, H, W)                 # (B*N, 3, H, W)

        # Encode — norm layer inside encoder produces power-normalised z
        z = self.jscc.encoder(x_flat)                    # (B*N, 2c, H', W')

        # Bypass avoids re-normalisation; all_ones deliberately traverses it
        # so the two paths can be tested for exact equivalence.
        if self.selector_mode == 'bypass':
            z_masked = z
            mask     = torch.ones(
                B * N, 1, z.shape[2], z.shape[3],
                device=z.device, dtype=z.dtype
            )
            logits   = torch.zeros_like(mask)
            z_norm   = z
        elif self.selector_mode == 'all_ones':
            mask = torch.ones(
                B * N, 1, z.shape[2], z.shape[3],
                device=z.device, dtype=z.dtype
            )
            logits = torch.zeros_like(mask)
            z_masked = z * mask
            z_norm = self._renormalize_to_reference(z_masked, z)
        else:
            mask, logits = self._selector_weights(z)
            z_masked = z * mask
            z_norm = self._renormalize_to_reference(z_masked, z)

        self.last_power_audit = {
            'mask_mean': mask.detach().mean().item(),
            'masked_vs_z_max': (z_masked.detach() - z.detach()).abs().max().item(),
            'renorm_vs_z_max': (z_norm.detach() - z.detach()).abs().max().item(),
            'energy_ratio': (
                z_norm.detach().square().sum()
                / z.detach().square().sum().clamp_min(1e-12)
            ).item(),
        }
        # Actual post-renormalisation transmit-power share at each latent
        # spatial location.  This—not the mean soft weight—is the quantity
        # visualised as a spatial power-allocation map.
        spatial_power = z_norm.detach().square().sum(dim=1, keepdim=True)
        self.last_spatial_power = spatial_power / spatial_power.flatten(1).sum(
            dim=1, keepdim=True
        ).view(-1, 1, 1, 1).clamp_min(1e-12)

        # Channel
        z_rx = self.jscc.channel(z_norm) if self.jscc.channel is not None \
               else z_norm                               # (B*N, 2c, H', W')

        # Decode
        x_hat     = self.jscc.decoder(z_rx)              # (B*N, 3, H, W)
        x_hat     = x_hat.view(B, N, C, H, W)           # (B, N, 3, H, W)
        x_refined = self.temporal(x_hat)                 # (B, N, 3, H, W)

        return x_refined, mask, logits

    def forward_no_selection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ablation baseline: VideoJSCC without task-aware selection.
        Bypasses the selector entirely — same as plain VideoJSCC.
        """
        B, N, C, H, W = x.shape
        x_flat    = x.view(B * N, C, H, W)
        x_hat     = self.jscc(x_flat)
        x_hat     = x_hat.view(B, N, C, H, W)
        return self.temporal(x_hat)

    # ------------------------------------------------------------------
    # Joint loss
    # ------------------------------------------------------------------

    def joint_loss(self, x: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Compute the direct joint fine-tuning loss.

        loss = lambda_task*L_task + lambda_recon*L_recon + lambda_rate*L_rate

        FIX (Bug #1): returns x_refined and tsn_logits so the caller can
        compute accuracy from the SAME forward pass — no second forward call.

        Args:
            x      : (B, N, 3, H, W) — input GoP
            labels : (B,) int64 — UCF101 class indices
        Returns:
            loss       : scalar weighted joint loss
            info       : dict of individual components for logging
            x_refined  : (B, N, 3, H, W) — reconstructed frames (for accuracy)
            tsn_logits : (B, 101) or None — TSN predictions (for accuracy)
        """
        x_refined, mask, logits = self.forward(x)

        # Reconstruction loss — always computed
        l_recon = F.mse_loss(x_refined, x)

        # Selector regularizer. With soft weighting the nominal CBR is fixed;
        # this term shapes allocation weights rather than changing symbol count.
        l_rate = self.selector.rate_loss(mask)

        if self.tsn is None:
            raise RuntimeError('Joint fine-tuning requires a frozen TSN checkpoint')
        tsn_logits = self.tsn(x_refined)
        l_task = F.cross_entropy(tsn_logits, labels)

        loss = (self.lambda_task  * l_task
              + self.lambda_recon * l_recon
              + self.lambda_rate  * l_rate)

        # Sigmoid of logits for human-readable score logging
        score_mean = torch.sigmoid(logits).mean().item()

        info = {
            'loss'       : loss.item(),
            'l_task'     : l_task.item(),
            'l_recon'    : l_recon.item(),
            'l_rate'     : l_rate.item(),
            'weighted_task'  : (self.lambda_task * l_task).item(),
            'weighted_recon' : (self.lambda_recon * l_recon).item(),
            'weighted_rate'  : (self.lambda_rate * l_rate).item(),
            'psnr'       : -10.0 * torch.log10(l_recon.clamp_min(1e-12)).item(),
            'rate_mean'  : mask.mean().item(),
            'score_mean' : score_mean,
            'tau'        : self.selector.tau,
        }

        return loss, info, x_refined, tsn_logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_temperature(self, tau: float):
        self.selector.set_temperature(tau)

    def change_channel(self, channel_type: str = 'AWGN', snr: float = None):
        self.jscc.change_channel(channel_type, snr)

    def get_channel(self) -> dict:
        return self.jscc.get_channel()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    TSN_HEAD = os.path.join(
        PROJECT_ROOT,
        'downstream/action_recognition/weights/tsn_ucf101_head.pth'
    )

    model = TAVideoJSCC(
        c=8,
        channel_type='AWGN',
        snr=10,
        n_frames=5,
        tsn_head_ckpt=TSN_HEAD,
        lambda_task=0.001,
        lambda_recon=1.0,
        lambda_rate=0.01,
        device=device,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nTrainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}  (TSN backbone + head)")

    B, N = 2, 5
    x      = torch.rand(B, N, 3, 256, 256).to(device)
    labels = torch.randint(0, 101, (B,)).to(device)

    print(f"\nInput shape : {x.shape}")

    print("\n=== Direct joint fine-tuning ===")
    model.zero_grad()
    model.set_selector_mode('learned')
    x_ref, mask, raw_logits = model(x)
    print(f"mask mean (< 1.0 expected): {mask.mean():.3f}")
    loss, info, x_ref2, tsn_logits = model.joint_loss(x, labels)
    print(f"Joint loss breakdown:")
    for k, v in info.items():
        print(f"  {k:12s} : {v}")
    loss.backward()
    print("Backward pass OK")
    grad = model.selector.scorer.net[0].weight.grad
    print(f"Selector grad norm : {grad.norm().item():.4f}")

    print("\nTAVideoJSCC direct joint training OK")
