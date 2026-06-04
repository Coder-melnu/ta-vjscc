# -*- coding: utf-8 -*-
"""
Task-Aware Feature Importance Module (Week 5 — Phase A).

Implements the memo's approach:
    1. ImportanceScorer  : 1×1 Conv on latent z_t → raw logits (no sigmoid)
    2. GumbelSelector    : differentiable soft/hard selection via Gumbel-Softmax
    3. TaskAwareSelector : assembles scorer + selector + rate loss

FIX (Bug #4): ImportanceScorer now outputs raw logits (no Sigmoid).
GumbelSelector receives raw logits directly — no double-log issue.
The sigmoid probability for logging is computed separately when needed.

Insertion point in the pipeline:
    z         = encoder(x)           # (B, 2c, H', W') — power-normalised
    z_masked  = selector(z)          # (B, 2c, H', W') — masked + re-normalised
    z_rx      = channel(z_masked)
    x_hat     = decoder(z_rx)

Loss:
    L = λ_task × L_task  (cross-entropy, action recognition)
      + λ_recon × L_recon (MSE reconstruction)
      + λ_rate × L_rate   (mean selection rate — penalise transmitting everything)

Starting λ values (per memo): λ_task=1.0, λ_recon=0.1, λ_rate=0.01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Importance Scorer — outputs raw logits (no Sigmoid)
# ---------------------------------------------------------------------------

class ImportanceScorer(nn.Module):
    """
    Lightweight 1×1 Conv scorer.

    Takes encoder latent z (B, 2c, H', W') and outputs a per-spatial-location
    raw logit (unbounded). The sigmoid probability is derived in GumbelSelector.

    FIX: Removed final Sigmoid. GumbelSelector uses logits directly to avoid
    the double-log issue (log(sigmoid(x)) = log_sigmoid(x) ≤ 0 always).

    Architecture:
        Conv1×1(2c → 2c) → PReLU → Conv1×1(2c → 1)   [no Sigmoid]
    """

    def __init__(self, channels: int):
        """
        Args:
            channels : number of latent channels (= 2c from encoder)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
            # No Sigmoid here — raw logits passed to GumbelSelector
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z      : (B, 2c, H', W') — encoder latent
        Returns:
            logits : (B, 1, H', W') — raw importance logits (unbounded)
        """
        return self.net(z)

    def scores(self, z: torch.Tensor) -> torch.Tensor:
        """Convenience: sigmoid of logits for logging/visualisation."""
        return torch.sigmoid(self.forward(z))


# ---------------------------------------------------------------------------
# 2. Gumbel-Softmax Selector — takes raw logits directly
# ---------------------------------------------------------------------------

class GumbelSelector(nn.Module):
    """
    Differentiable binary feature selection via Gumbel-Softmax.

    FIX: Accepts raw logits (not probabilities). Builds 2-class logits as
    [-logit, +logit] (symmetric), which is the correct Bernoulli parameterisation.
    No log(sigmoid) double-log issue.

    Args:
        tau  : Gumbel temperature. Higher = softer. Lower = harder (→ argmax).
               Start at 1.0, anneal down during training.
        hard : if True, straight-through estimator (hard {0,1} mask, soft gradient).
               Use hard=False during training, hard=True at inference.
    """

    def __init__(self, tau: float = 1.0, hard: bool = False):
        super().__init__()
        self.tau  = tau
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H', W') — raw importance logits from ImportanceScorer
        Returns:
            mask   : (B, 1, H', W') soft ∈ (0,1) during training,
                                     hard ∈ {0,1} when self.hard=True
        """
        B, _, H, W = logits.shape

        # Symmetric 2-class Bernoulli logits: [drop=-logit, keep=+logit]
        # This is the correct parameterisation: p(keep) = sigmoid(logit)
        keep_logit = logits                          # (B, 1, H', W')
        drop_logit = -logits                         # (B, 1, H', W')
        two_class  = torch.cat([drop_logit, keep_logit], dim=1)  # (B, 2, H', W')

        # Reshape to (..., num_classes) for gumbel_softmax
        two_class_flat = two_class.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H'*W', 2)

        soft = F.gumbel_softmax(two_class_flat, tau=self.tau, hard=self.hard)
        # soft[:, 1] = "keep" probability

        mask = soft[:, 1].reshape(B, 1, H, W)       # (B, 1, H', W')
        return mask

    def set_temperature(self, tau: float):
        """Anneal temperature during training."""
        self.tau = tau


# ---------------------------------------------------------------------------
# 3. TaskAwareSelector — assembled module
# ---------------------------------------------------------------------------

class TaskAwareSelector(nn.Module):
    """
    Full task-aware selection module: scorer + Gumbel selector + masking.

    Sits between encoder output and channel input.

    Args:
        channels  : latent channels (= 2c from encoder)
        tau       : initial Gumbel temperature
        hard      : use straight-through at train time (default: False = soft)
    """

    def __init__(self, channels: int, tau: float = 1.0, hard: bool = False):
        super().__init__()
        self.scorer   = ImportanceScorer(channels)
        self.selector = GumbelSelector(tau=tau, hard=hard)

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z : (B, 2c, H', W') — encoder latent (power-normalised)
        Returns:
            z_masked : (B, 2c, H', W') — masked latent (NOT re-normalised here;
                       caller must re-normalise before channel — see TAVideoJSCC)
            mask     : (B, 1, H', W')  — soft/hard selection mask
            logits   : (B, 1, H', W')  — raw importance logits (for logging)
        """
        logits   = self.scorer(z)           # (B, 1, H', W') raw
        mask     = self.selector(logits)    # (B, 1, H', W') soft/hard
        z_masked = z * mask                 # broadcast across channel dim
        return z_masked, mask, logits

    def rate_loss(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Rate penalty: mean selection rate across batch and spatial locations.
        L_rate = mean(mask) ∈ [0,1]. Encourages sparsity.

        Args:
            mask : (B, 1, H', W')
        Returns:
            scalar
        """
        return mask.mean()

    def set_temperature(self, tau: float):
        self.selector.set_temperature(tau)

    @property
    def tau(self):
        return self.selector.tau


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(42)

    B, c, H, W = 4, 8, 64, 64
    latent_channels = 2 * c         # 16

    selector = TaskAwareSelector(channels=latent_channels, tau=1.0, hard=False)
    print(selector)

    z = torch.randn(B, latent_channels, H, W)
    z.requires_grad_(True)

    z_masked, mask, logits = selector(z)
    scores = torch.sigmoid(logits)  # probability view for logging

    print(f"\nInput  z      : {z.shape}")
    print(f"Logits        : {logits.shape}  min={logits.min():.3f}  max={logits.max():.3f}")
    print(f"Scores (sig)  : {scores.shape}  min={scores.min():.3f}  max={scores.max():.3f}")
    print(f"Mask          : {mask.shape}    min={mask.min():.3f}    max={mask.max():.3f}")
    print(f"z_masked      : {z_masked.shape}")
    print(f"Rate loss     : {selector.rate_loss(mask).item():.4f}")

    # Verify symmetric logits: keep_mean ≈ 0.5 at init
    print(f"Mean mask (should be ~0.5 at init) : {mask.mean():.3f}")

    # Gradient check
    loss = z_masked.sum()
    loss.backward()
    print(f"\nGradient check:")
    print(f"  scorer conv0 grad norm : {selector.scorer.net[0].weight.grad.norm():.4f}")
    print(f"  z.grad norm            : {z.grad.norm():.4f}")

    n_params = sum(p.numel() for p in selector.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    print("TaskAwareSelector OK")