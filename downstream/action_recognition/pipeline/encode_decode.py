# -*- coding: utf-8 -*-
"""
DeepJSCC encode→channel→decode pipeline wrapper.
Loads VideoJSCC checkpoints from out_video/checkpoints/ and runs inference.

Supports:
- Auto-parsing c, snr, channel from checkpoint directory name
- SNR hot-swap via model.change_channel() — no model reload needed
- GoP input (B, N, 3, H, W) or single GoP (N, 3, H, W)
"""

import os
import sys
import re
import glob
import torch
from typing import Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model.video_jscc import VideoJSCC


# ---------------------------------------------------------------------------
# Checkpoint parsing
# ---------------------------------------------------------------------------

def parse_checkpoint_name(ckpt_dir: str) -> Dict:
    """
    Parse config from checkpoint directory name.
    Format: VideoJSCC_{channel}_c{c}_snr{snr}_ratio{ratio}_{timestamp}
    """
    name = os.path.basename(ckpt_dir.rstrip('/'))
    info = {}
    for pattern, key, cast in [
        (r'_(AWGN|Rayleigh)_', 'channel', str),
        (r'_c(\d+)_',          'c',       int),
        (r'_snr([\d.]+)_',     'snr',     float),
        (r'_ratio([\d.]+)',    'ratio',   float),
    ]:
        m = re.search(pattern, name)
        if m:
            info[key] = cast(m.group(1))
    return info


def find_checkpoint_file(ckpt_dir: str, prefer_best: bool = True) -> Optional[str]:
    """Return path to best.pkl or latest epoch_N.pkl in ckpt_dir."""
    if prefer_best:
        best = os.path.join(ckpt_dir, 'best.pkl')
        if os.path.exists(best):
            return best
    epoch_files = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pkl')))
    return epoch_files[-1] if epoch_files else None


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class DeepJSCCPipeline:
    """
    VideoJSCC encode→channel→decode pipeline.

    Args:
        ckpt_path    : path to best.pkl or epoch_N.pkl
        channel_type : 'AWGN' or 'Rayleigh'
        snr          : initial SNR in dB (hot-swappable)
        c            : bottleneck depth (must match training)
        n_frames     : GoP size (must match training, default 5)
        hidden_dim   : TemporalFusionModule hidden channels (default 16)
        device       : 'cuda:0', 'cuda:1', 'cpu'
    """

    def __init__(
        self,
        ckpt_path:    str,
        channel_type: str   = 'AWGN',
        snr:          float = 13.0,
        c:            int   = 8,
        n_frames:     int   = 5,
        hidden_dim:   int   = 16,
        device:       str   = 'cuda:0',
    ):
        self.device       = device
        self.channel_type = channel_type
        self.current_snr  = snr
        self.c            = c
        self.n_frames     = n_frames

        model = VideoJSCC(
            c=c, channel_type=channel_type, snr=snr,
            n_frames=n_frames, hidden_dim=hidden_dim,
        )
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        self.model = model.to(device)
        self.model.eval()
        print(f"[pipeline] Loaded: c={c}, {channel_type}, SNR={snr} dB  ← {os.path.basename(ckpt_path)}")

    def set_snr(self, snr: float):
        """Hot-swap SNR without reloading model."""
        self.current_snr = snr
        self.model.change_channel(self.channel_type, snr)

    def set_channel(self, channel_type: str, snr: float):
        self.channel_type = channel_type
        self.current_snr  = snr
        self.model.change_channel(channel_type, snr)

    @torch.no_grad()
    def reconstruct_gop(
        self,
        gop:          torch.Tensor,
        use_temporal: bool = True,
    ) -> torch.Tensor:
        """
        Encode → channel → decode a GoP.

        Args:
            gop          : (B, N, 3, H, W) or (N, 3, H, W) float [0,1]
            use_temporal : use full VideoJSCC (True) or per-frame only (False)

        Returns:
            reconstructed GoP, same shape, float [0,1], on CPU
        """
        squeezed = gop.dim() == 4
        if squeezed:
            gop = gop.unsqueeze(0)

        gop = gop.to(self.device)
        out = self.model(gop) if use_temporal else self.model.forward_no_temporal(gop)
        out = out.clamp(0.0, 1.0).cpu()

        return out.squeeze(0) if squeezed else out


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_pipeline_from_dir(
    ckpt_dir:    str,
    snr:         float = None,
    device:      str   = 'cuda:0',
    prefer_best: bool  = True,
) -> DeepJSCCPipeline:
    """
    Build DeepJSCCPipeline from a checkpoint directory.
    Auto-parses c, channel, snr from directory name.

    Args:
        ckpt_dir    : e.g. out_video/checkpoints/VideoJSCC_AWGN_c8_snr13.0_ratio0.1667_...
        snr         : override SNR (None = use SNR from directory name)
        device      : torch device string
        prefer_best : use best.pkl if available
    """
    info      = parse_checkpoint_name(ckpt_dir)
    ckpt_file = find_checkpoint_file(ckpt_dir, prefer_best=prefer_best)
    assert ckpt_file, f"No checkpoint found in {ckpt_dir}"

    return DeepJSCCPipeline(
        ckpt_path=ckpt_file,
        channel_type=info.get('channel', 'AWGN'),
        snr=snr if snr is not None else info.get('snr', 13.0),
        c=info.get('c', 8),
        device=device,
    )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ckpt_dirs = sorted(glob.glob(
        os.path.join(PROJECT_ROOT, 'out_video/checkpoints/VideoJSCC_*')
    ))
    valid = [d for d in ckpt_dirs if find_checkpoint_file(d)]

    if not valid:
        print("[pipeline] No checkpoints found.")
    else:
        pipeline = build_pipeline_from_dir(valid[0], device='cuda:0')
        dummy    = torch.rand(1, 5, 3, 128, 128)
        out      = pipeline.reconstruct_gop(dummy)
        print(f"[pipeline] Input:  {dummy.shape}")
        print(f"[pipeline] Output: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")

        pipeline.set_snr(7.0)
        out2 = pipeline.reconstruct_gop(dummy)
        print(f"[pipeline] SNR=7 output: {out2.shape}")
        print("[pipeline] OK")