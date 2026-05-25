# -*- coding: utf-8 -*-
"""
Image quality metrics — PSNR and MS-SSIM.
Shared across all downstream tasks.

Operates on:
  - Single frame : (3, H, W)
  - Single GoP   : (N, 3, H, W)
  - Batched GoP  : (B, N, 3, H, W)
"""

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    pred:      torch.Tensor,
    target:    torch.Tensor,
    max_val:   float = 1.0,
    reduction: str   = 'mean',
) -> float:
    """PSNR in dB. Higher is better."""
    pred   = pred.float().clamp(0, max_val)
    target = target.float().clamp(0, max_val)

    if pred.dim() == 5:                         # (B, N, 3, H, W)
        B, N, C, H, W = pred.shape
        pred_f   = pred.reshape(B * N, -1)
        target_f = target.reshape(B * N, -1)
    elif pred.dim() == 4:                       # (N, 3, H, W)
        pred_f   = pred.reshape(pred.shape[0], -1)
        target_f = target.reshape(target.shape[0], -1)
    else:                                       # (3, H, W)
        pred_f   = pred.reshape(1, -1)
        target_f = target.reshape(1, -1)

    mse = torch.mean((pred_f - target_f) ** 2, dim=1).clamp(min=1e-10)
    psnr_vals = 10 * torch.log10(torch.tensor(max_val ** 2) / mse)

    return psnr_vals if reduction == 'none' else float(psnr_vals.mean().item())


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32) - size // 2
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _ssim_single(pred, target, window, C1=0.01**2, C2=0.03**2):
    B, C, H, W  = pred.shape
    w2d = (window.unsqueeze(0) * window.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    w2d = w2d.expand(C, 1, -1, -1).to(pred.device)
    pad = window.shape[0] // 2

    mu1 = F.conv2d(pred,   w2d, padding=pad, groups=C)
    mu2 = F.conv2d(target, w2d, padding=pad, groups=C)
    s1  = F.conv2d(pred * pred,     w2d, padding=pad, groups=C) - mu1 * mu1
    s2  = F.conv2d(target * target, w2d, padding=pad, groups=C) - mu2 * mu2
    s12 = F.conv2d(pred * target,   w2d, padding=pad, groups=C) - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return (num / den).mean(dim=[1, 2, 3])


def ms_ssim(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    levels:     int  = 5,
    reduction:  str  = 'mean',
) -> float:
    """MS-SSIM in [0,1]. Higher is better."""
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:levels])

    if pred.dim() == 5:
        B, N, C, H, W = pred.shape
        pred   = pred.reshape(B * N, C, H, W)
        target = target.reshape(B * N, C, H, W)
    elif pred.dim() == 3:
        pred   = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    pred   = pred.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    window = _gaussian_kernel().to(pred.device)
    val    = torch.ones(pred.shape[0], device=pred.device)

    for i in range(levels):
        s = _ssim_single(pred, target, window)
        val = val * (s ** weights[i].to(pred.device))
        if i < levels - 1:
            pred   = F.avg_pool2d(pred,   2)
            target = F.avg_pool2d(target, 2)

    return val if reduction == 'none' else float(val.mean().item())


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def compute_gop_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute PSNR and MS-SSIM for a GoP or batch of GoPs.

    Args:
        pred, target: (N, 3, H, W) or (B, N, 3, H, W) float [0,1]

    Returns:
        {'psnr_db': float, 'ms_ssim': float}
    """
    return {
        'psnr_db' : psnr(pred, target),
        'ms_ssim' : ms_ssim(pred, target),
    }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    x = torch.rand(2, 5, 3, 128, 128)
    print(f"PSNR (identical) : {psnr(x, x):.2f} dB")

    noisy = (x + torch.randn_like(x) * 0.1).clamp(0, 1)
    print(f"PSNR (noisy)     : {psnr(noisy, x):.2f} dB")
    print(f"MS-SSIM (noisy)  : {ms_ssim(noisy, x):.4f}")
    print(f"GoP metrics      : {compute_gop_metrics(noisy, x)}")