# -*- coding: utf-8 -*-
"""
Task-Aware Video DeepJSCC (TA-VideoJSCC) — Week 5 Phase A + Staged Training.

Extends VideoJSCC with:
    1. TaskAwareSelector between encoder and channel
    2. Re-normalisation after masking to restore power constraint  [FIX Bug #3]
    3. Frozen TSN for task loss — gradients flow through x_hat into decoder/encoder
    4. joint_loss() returns (loss, info, x_refined, logits) to avoid double
       forward pass in evaluate_epoch                              [FIX Bug #1]
    5. Staged training support:
       Stage 1 (epochs 1–stage1_epochs): reconstruction loss only, selector bypassed
       Stage 2 (epochs stage1_epochs+1–end): full joint loss, selector active
    6. PSNR logging: per-frame PSNR of x_refined vs x logged in info dict
       before passing to TSN — diagnostic for reconstruction quality

Pipeline:
    x (B,N,3,H,W)
    → encoder        → z (B*N, 2c, H', W')  [power-normalised inside encoder]
    → selector       → z_masked (B*N, 2c, H', W')  [bypassed in Stage 1]
    → re-normalise   → z_norm   (B*N, 2c, H', W')  [restore power constraint]
    → channel        → z_rx
    → decoder        → x_hat (B*N, 3, H, W)
    → temporal       → x_refined (B, N, 3, H, W)
    → [PSNR logged here — before TSN sees frames]
    → FrozenTSN      → logits (B, 101)  [frozen weights, grad flows through frames]
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.jscc import DeepJSCC
from model.temporal import TemporalFusionModule
from model.importance import TaskAwareSelector


# ---------------------------------------------------------------------------
# PSNR helper
# ---------------------------------------------------------------------------

def compute_psnr(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR between reconstructed and original frames.

    Args:
        x_hat   : reconstructed tensor, any shape, float [0, max_val]
        x       : original tensor, same shape as x_hat
        max_val : peak signal value (default 1.0 for normalised frames)
    Returns:
        psnr_db : float — PSNR in dB (higher is better)

    Note: computed with torch.no_grad() internally so it never contributes
    gradients, even when called inside joint_loss() during training.
    """
    with torch.no_grad():
        mse = F.mse_loss(x_hat.detach(), x.detach())
        if mse.item() == 0.0:
            return float('inf')
        psnr_db = 20.0 * math.log10(max_val) - 10.0 * torch.log10(mse).item()
    return psnr_db


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

        from torchvision.models import resnet50, ResNet50_Weights

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Linear(2048, 101)

        if head_ckpt and os.path.exists(head_ckpt):
            self.head.load_state_dict(
                torch.load(head_ckpt, map_location=device, weights_only=True)
            )
            print(f"[FrozenTSN] Loaded head from {head_ckpt}")
        else:
            print(f"[FrozenTSN] WARNING: head_ckpt not found: {head_ckpt}")

        # Freeze all TSN weights
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation buffers (move to correct device with .to())
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (B, N, 3, H, W) float [0,1]
        Returns:
            logits : (B, 101)
        """
        frames_norm = (frames - self.mean) / self.std   # (B, N, 3, H, W)
        B, N, C, H, W = frames_norm.shape
        frames_flat = frames_norm.reshape(B * N, C, H, W)
        features    = self.feature_extractor(frames_flat).flatten(1)  # (B*N, 2048)
        logits_flat = self.head(features)                              # (B*N, 101)
        logits      = logits_flat.reshape(B, N, 101).mean(dim=1)      # (B, 101)
        return logits


# ---------------------------------------------------------------------------
# TA-VideoJSCC
# ---------------------------------------------------------------------------

class TAVideoJSCC(nn.Module):
    """
    Task-Aware Video DeepJSCC with staged training support.

    Args:
        c             : encoder bottleneck depth (complex symbols per spatial location)
        channel_type  : 'AWGN' or 'Rayleigh'
        snr           : channel SNR in dB
        P             : transmit power constraint (default 1.0)
        n_frames      : GoP size (default 5)
        hidden_dim    : TemporalFusionModule hidden channels
        tsn_head_ckpt : path to tsn_ucf101_head.pth
        lambda_task   : task loss weight   (memo default: 1.0)
        lambda_recon  : recon loss weight  (memo default: 0.1)
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
        lambda_task:   float = 1.0,
        lambda_recon:  float = 0.1,
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

        # Staged training: 1 = reconstruction only, 2 = full joint loss
        # Start in Stage 1 by default
        self.stage = 1

        # Frozen TSN
        if tsn_head_ckpt is not None:
            self.tsn = FrozenTSN(head_ckpt=tsn_head_ckpt, device=device)
        else:
            self.tsn = None
            print("[TAVideoJSCC] No TSN ckpt provided — task loss will be zero")

    # ------------------------------------------------------------------
    # Staged training control
    # ------------------------------------------------------------------

    def set_stage(self, stage: int):
        """
        Switch between staged training modes.

        Stage 1: bypass selector entirely, reconstruction loss only.
                 Selector parameters are frozen — no wasted gradients.
        Stage 2: activate selector, full joint loss
                 (lambda_task * L_task + lambda_recon * L_recon + lambda_rate * L_rate).

        Call this at the start of each epoch in the training loop.
        """
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")

        self.stage = stage

        # Freeze selector in Stage 1 (no gradient computation needed)
        # Unfreeze in Stage 2 so it can learn
        for p in self.selector.parameters():
            p.requires_grad = (stage == 2)

        print(f"[TAVideoJSCC] Stage set to {stage} — "
              f"selector {'ACTIVE' if stage == 2 else 'BYPASSED'}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass with task-aware selection (or bypass in Stage 1).

        Args:
            x : (B, N, 3, H, W) — GoP of N frames, float [0,1]
        Returns:
            x_refined : (B, N, 3, H, W) — reconstructed frames
            mask      : (B*N, 1, H', W') — selection mask per frame
                        (all-ones in Stage 1)
            logits    : (B*N, 1, H', W') — raw importance logits per frame
                        (all-zeros in Stage 1)
        """
        B, N, C, H, W = x.shape
        assert N == self.n_frames, f"Expected {self.n_frames} frames, got {N}"

        x_flat = x.view(B * N, C, H, W)                 # (B*N, 3, H, W)

        # Encode — norm layer inside encoder produces power-normalised z
        z = self.jscc.encoder(x_flat)                    # (B*N, 2c, H', W')

        # Task-aware selection — bypassed in Stage 1
        if self.stage == 1:
            # Pass all features through unchanged
            z_masked = z
            mask     = torch.ones(
                B * N, 1, z.shape[2], z.shape[3],
                device=z.device, dtype=z.dtype
            )
            logits   = torch.zeros_like(mask)
        else:
            # Stage 2: full Gumbel-Softmax selection
            z_masked, mask, logits = self.selector(z)    # (B*N, 2c/1/1, H', W')

        # FIX Bug #3: re-normalise after masking to restore power constraint.
        # encoder.norm is the same closure used inside _Encoder.forward().
        # In Stage 1, z_masked == z so this is a no-op on an already-normalised z.
        z_norm = self.jscc.encoder.norm(z_masked)        # (B*N, 2c, H', W')

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
        Compute loss according to current stage.

        Stage 1: loss = L_recon only (MSE)
        Stage 2: loss = lambda_task*L_task + lambda_recon*L_recon + lambda_rate*L_rate

        FIX (Bug #1): returns x_refined and tsn_logits so the caller can
        compute accuracy from the SAME forward pass — no second forward call.

        PSNR diagnostic: psnr_before_tsn is logged in info at every call.
        This measures reconstruction quality of x_refined vs x BEFORE TSN
        sees the frames. Use this to diagnose whether low accuracy is caused
        by poor reconstruction or by the importance module / loss weights.

        Expected ranges:
            ~22–25 dB  → 128×128 resolution bottleneck is limiting quality
            ~28–35 dB  → reconstruction is reasonable; issue is elsewhere
            Drop Stage1→Stage2 > 3 dB → joint loss is hurting reconstruction

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

        # ------------------------------------------------------------------
        # PSNR diagnostic — computed before TSN forward pass
        # Detached so it never affects gradients
        # ------------------------------------------------------------------
        psnr_before_tsn = compute_psnr(x_refined, x, max_val=1.0)

        # Reconstruction loss — always computed
        l_recon = F.mse_loss(x_refined, x)

        # Rate loss — zero in Stage 1 (mask is all-ones, mean=1.0)
        # We still compute it for logging but don't include in Stage 1 loss
        l_rate = self.selector.rate_loss(mask)

        # Task loss — zero in Stage 1 (TSN not called to save compute)
        tsn_logits = None
        if self.stage == 2 and self.tsn is not None:
            tsn_logits = self.tsn(x_refined)              # (B, 101)
            l_task     = F.cross_entropy(tsn_logits, labels)
        else:
            l_task = x.new_tensor(0.0)

        # Weighted sum — depends on stage
        if self.stage == 1:
            loss = l_recon
        else:
            loss = (self.lambda_task  * l_task
                  + self.lambda_recon * l_recon
                  + self.lambda_rate  * l_rate)

        # Sigmoid of logits for human-readable score logging
        score_mean = torch.sigmoid(logits).mean().item()

        info = {
            'loss'            : loss.item(),
            'l_task'          : l_task.item(),
            'l_recon'         : l_recon.item(),
            'l_rate'          : l_rate.item(),
            'rate_mean'       : mask.mean().item(),
            'score_mean'      : score_mean,
            'psnr_before_tsn' : psnr_before_tsn,   # <-- NEW: dB, detached
            'tau'             : self.selector.tau,
            'stage'           : float(self.stage),
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
        lambda_task=1.0,
        lambda_recon=0.1,
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

    # --- Test Stage 1 ---
    print("\n=== Stage 1 (reconstruction only) ===")
    model.set_stage(1)
    x_ref, mask, raw_logits = model(x)
    print(f"mask mean (should be 1.0) : {mask.mean():.3f}")
    loss, info, x_ref2, tsn_logits = model.joint_loss(x, labels)
    print(f"loss = l_recon only           : {info['loss']:.4f}")
    print(f"l_task (should be 0.0)        : {info['l_task']:.4f}")
    print(f"psnr_before_tsn (Stage 1)     : {info['psnr_before_tsn']:.2f} dB")
    print(f"tsn_logits (should be None)   : {tsn_logits}")
    loss.backward()
    print("Backward pass OK (Stage 1)")

    # --- Test Stage 2 ---
    print("\n=== Stage 2 (full joint loss) ===")
    model.zero_grad()
    model.set_stage(2)
    x_ref, mask, raw_logits = model(x)
    print(f"mask mean (< 1.0 expected): {mask.mean():.3f}")
    loss, info, x_ref2, tsn_logits = model.joint_loss(x, labels)
    print(f"Joint loss breakdown:")
    for k, v in info.items():
        print(f"  {k:20s} : {v}")
    loss.backward()
    print("Backward pass OK (Stage 2)")
    grad = model.selector.scorer.net[0].weight.grad
    print(f"Selector grad norm : {grad.norm().item():.4f}")

    print("\nTAVideoJSCC staged training OK")