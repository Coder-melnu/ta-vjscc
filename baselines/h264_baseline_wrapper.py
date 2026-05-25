# -*- coding: utf-8 -*-
"""
H.264+LDPC baseline wrapper — shared across all downstream tasks.

Wraps baselines/h264_ldpc_baseline.py to:
1. Accept a GoP tensor (N, 3, H, W) as input
2. Run H.264 encode → LDPC channel → LDPC decode → H.264 decode
3. Return (reconstructed_gop_tensor, metrics_dict) — compatible with
   the downstream evaluation pipeline

Placed in baselines/ (not downstream/) because this baseline is shared
across all downstream tasks: action recognition, object detection, segmentation.

Bandwidth matching:
    target_bpp = CBR × 2 × ldpc_rate
    (factor 2: real + imaginary components of complex JSCC symbols)
"""

import os
import sys
import tempfile
import numpy as np
import torch
import cv2
from typing import Tuple, Dict
from pathlib import Path
from PIL import Image

# baselines/ is the same directory as this file
BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(BASELINES_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BASELINES_DIR)

# Lazy imports — only loaded when H264LDPCPipeline.reconstruct_gop() is called
# This allows run_pipeline.py to import H264LDPCPipeline in ta-vjscc env
# without requiring tensorflow/sionna at import time
def _get_baseline_funcs():
    from h264_ldpc_baseline import (
        encode_frames_to_video_2pass,
        decode_video_to_frames,
        simulate_ldpc_channel,
        compute_psnr_msssim,
    )
    return encode_frames_to_video_2pass, decode_video_to_frames, \
           simulate_ldpc_channel, compute_psnr_msssim


# ---------------------------------------------------------------------------
# CBR → target BPP
# ---------------------------------------------------------------------------

def cbr_to_target_bpp(cbr: float, ldpc_rate: float = 0.6667) -> float:
    """
    Convert DeepJSCC CBR to H.264 target source BPP.

    channel_bpp = CBR × 2   (real + imaginary)
    source_bpp  = channel_bpp × ldpc_rate
    """
    return cbr * 2 * ldpc_rate


# ---------------------------------------------------------------------------
# H264LDPCPipeline
# ---------------------------------------------------------------------------

class H264LDPCPipeline:
    """
    H.264+LDPC baseline — same interface as DeepJSCCPipeline.

    Args:
        cbr          : bandwidth ratio matching DeepJSCC (e.g. 1/6 ≈ 0.1667)
        channel_type : 'AWGN' or 'Rayleigh'
        snr          : initial SNR in dB (hot-swappable via set_snr)
        ldpc_rate    : LDPC code rate (default 2/3)
        fps          : video fps for FFmpeg (default 25)
        codec        : 'h264' or 'h265'
    """

    def __init__(
        self,
        cbr:          float = 1/6,
        channel_type: str   = 'AWGN',
        snr:          float = 13.0,
        ldpc_rate:    float = 0.6667,
        fps:          int   = 25,
        codec:        str   = 'h264',
    ):
        self.cbr          = cbr
        self.channel_type = channel_type
        self.current_snr  = snr
        self.ldpc_rate    = ldpc_rate
        self.fps          = fps
        self.codec        = codec
        self.target_bpp   = cbr_to_target_bpp(cbr, ldpc_rate)

        # Expose same attributes as DeepJSCCPipeline for compatibility
        self.c = int(round(cbr * 48))   # approximate c equivalent for labeling

        print(f"[H264+LDPC] CBR={cbr:.4f}, target_bpp={self.target_bpp:.4f}, "
              f"channel={channel_type}, SNR={snr} dB, LDPC R={ldpc_rate:.4f}")

    def set_snr(self, snr: float):
        """Match DeepJSCCPipeline interface."""
        self.current_snr = snr

    def set_channel(self, channel_type: str, snr: float):
        self.channel_type = channel_type
        self.current_snr  = snr

    def reconstruct_gop(
        self,
        gop:        torch.Tensor,
        image_size: int = 128,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Reconstruct a GoP through H.264+LDPC pipeline.

        Args:
            gop        : (B, N, 3, H, W) or (N, 3, H, W) float [0,1]
            image_size : spatial resolution

        Returns:
            (reconstructed_gop, metrics)
            reconstructed_gop : (N, 3, H, W) float [0,1] on CPU
            metrics           : {'psnr_db': float, 'ms_ssim': float, 'ber': float}
        """
        if gop.dim() == 5:
            gop = gop[0]

        gop = gop.detach().cpu()
        N, C, H, W = gop.shape

        with tempfile.TemporaryDirectory() as tmpdir:
            encode_frames_to_video_2pass, decode_video_to_frames, \
            simulate_ldpc_channel, compute_psnr_msssim = _get_baseline_funcs()

            orig_dir   = os.path.join(tmpdir, 'original')
            recon_dir  = os.path.join(tmpdir, 'reconstructed')
            video_path = os.path.join(tmpdir, 'compressed.mp4')
            recov_path = os.path.join(tmpdir, 'recovered.mp4')
            os.makedirs(orig_dir,  exist_ok=True)
            os.makedirs(recon_dir, exist_ok=True)

            # Step 1: Save GoP frames as PNGs
            frames_np = gop.permute(0, 2, 3, 1).numpy()
            frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
            for i, frame in enumerate(frames_np):
                Image.fromarray(frame).save(os.path.join(orig_dir, f'{i+1:04d}.png'))

            # Step 2: H.264 2-pass encode (bandwidth-matched to CBR)
            target_kbps = max(
                int(self.target_bpp * image_size * image_size * self.fps / 1000),
                10   # minimum 10 kbps
            )

            try:
                encode_frames_to_video_2pass(
                    orig_dir, video_path,
                    codec=self.codec,
                    target_bitrate_kbps=target_kbps,
                    fps=self.fps,
                )
            except RuntimeError as e:
                print(f"[H264+LDPC] FFmpeg encode failed: {e}")
                return gop, {'psnr_db': 0.0, 'ms_ssim': 0.0, 'ber': 1.0}

            # Step 3: LDPC encode → channel → LDPC decode
            try:
                ber = simulate_ldpc_channel(
                    video_path, recov_path,
                    snr_db=self.current_snr,
                    channel_type=self.channel_type,
                    ldpc_rate=self.ldpc_rate,
                )
            except Exception as e:
                print(f"[H264+LDPC] LDPC simulation failed: {e}")
                return gop, {'psnr_db': 0.0, 'ms_ssim': 0.0, 'ber': 1.0}

            # Step 4: Decode recovered bitstream → frames
            try:
                decode_video_to_frames(recov_path, recon_dir, image_size=image_size)
            except RuntimeError:
                print(f"[H264+LDPC] Decode failed (BER={ber:.4f}) — returning zeros")
                return torch.zeros_like(gop), {'psnr_db': 0.0, 'ms_ssim': 0.0, 'ber': ber}

            # Step 5: Load reconstructed frames → tensor
            recon_files = sorted(Path(recon_dir).glob('*.png'))
            if not recon_files:
                return torch.zeros_like(gop), {'psnr_db': 0.0, 'ms_ssim': 0.0, 'ber': ber}

            recon_frames = []
            for i in range(N):
                idx   = min(i, len(recon_files) - 1)
                frame = np.array(Image.open(recon_files[idx]).convert('RGB'))
                frame = cv2.resize(frame, (W, H))
                recon_frames.append(frame)

            recon_np  = np.stack(recon_frames).astype(np.float32) / 255.0
            recon_gop = torch.from_numpy(recon_np).permute(0, 3, 1, 2)

            # Step 6: Compute metrics
            metrics_raw = compute_psnr_msssim(orig_dir, recon_dir)
            metrics = {
                'psnr_db' : metrics_raw['psnr'],
                'ms_ssim' : metrics_raw['msssim'],
                'ber'     : ber,
            }

        return recon_gop.clamp(0, 1), metrics


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pipeline  = H264LDPCPipeline(cbr=1/6, channel_type='AWGN', snr=13.0)
    dummy_gop = torch.rand(5, 3, 128, 128)
    recon, metrics = pipeline.reconstruct_gop(dummy_gop, image_size=128)
    print(f"Input:   {dummy_gop.shape}")
    print(f"Output:  {recon.shape}, range=[{recon.min():.3f}, {recon.max():.3f}]")
    print(f"PSNR:    {metrics['psnr_db']:.2f} dB")
    print(f"MS-SSIM: {metrics['ms_ssim']:.4f}")
    print(f"BER:     {metrics['ber']:.6f}")