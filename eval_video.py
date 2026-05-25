# -*- coding: utf-8 -*-
"""
Week 3 VideoJSCC Evaluation Script.
Evaluates all 10 trained checkpoints across SNR 0-25 dB.
Computes PSNR and MS-SSIM for:
  - VideoJSCC (with temporal fusion)       ← full model
  - VideoJSCC (no temporal fusion)         ← ablation: forward_no_temporal()

This produces the key Week 3 result: temporal fusion improves PSNR/MS-SSIM
over per-frame coding.

Usage:
    cd ~/Nu/ta-vjscc
    python eval_video.py

Output:
    week3_results/
    ├── eval_results_all.csv        ← per (checkpoint, snr, method) results
    └── psnr_msssim_vs_snr.png      ← PSNR and MS-SSIM curves
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pytorch_msssim import ms_ssim as torch_ms_ssim
import matplotlib.pyplot as plt

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model.video_jscc import VideoJSCC
from data.ucf101_dataloader import build_dataloaders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_ROOT      = 'out_video/checkpoints'
FRAMES_ROOT    = 'datasets/UCF101Frames'
ANNOTATION_PATH = 'datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist'
OUTPUT_DIR     = 'week3_results'
DEVICE         = 'cuda:0'
SNR_LIST       = list(range(0, 26))   # 0 to 25 dB — matches H.264+LDPC baseline
REPEATS        = 1                    # 1 repeat sufficient with full 3783 GoPs
IMAGE_SIZE     = 128
GOP_SIZE       = 5
BATCH_SIZE     = 16                   # larger batch = better GPU utilization
NUM_WORKERS    = 4
SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_ckpt_name(name: str) -> dict:
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


def compute_psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ms_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MS-SSIM for (B, N, 3, H, W) or (B, 3, H, W) tensors in [0, 1]."""
    if pred.dim() == 5:
        B, N, C, H, W = pred.shape
        pred   = pred.reshape(B * N, C, H, W)
        target = target.reshape(B * N, C, H, W)

    pred   = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    H      = pred.shape[2]

    if H < 160:
        weights = [0.3222, 0.3363, 0.3415]
        return float(torch_ms_ssim(pred, target, data_range=1.0,
                                   win_size=7, weights=weights))
    return float(torch_ms_ssim(pred, target, data_range=1.0, win_size=7))


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(
    model:       VideoJSCC,
    test_loader,
    snr_list:    list,
    channel:     str,
    repeats:     int = 3,
    device:      str = 'cuda:0',
) -> list:
    """
    Evaluate one checkpoint across all SNR points.
    Returns list of dicts with PSNR/MS-SSIM for both temporal and no-temporal.
    """
    model.eval()
    results = []

    for snr in tqdm(snr_list, desc=f"SNR sweep"):
        model.change_channel(channel, snr)

        mse_temp_list    = []
        msssim_temp_list = []
        mse_notmp_list   = []
        msssim_notmp_list = []

        for rep in range(repeats):
            mse_temp = mse_notmp = 0.0
            ms_temp  = ms_notmp  = 0.0
            count = 0

            for gops, _ in test_loader:
                gops = gops.to(device)          # (B, N, 3, H, W)

                # Full model (with temporal fusion)
                out_temp  = model(gops).clamp(0, 1)
                # Ablation (no temporal fusion)
                out_notmp = model.forward_no_temporal(gops).clamp(0, 1)

                # MSE
                mse_temp  += torch.mean((out_temp  - gops) ** 2).item()
                mse_notmp += torch.mean((out_notmp - gops) ** 2).item()

                # MS-SSIM
                ms_temp  += compute_ms_ssim(out_temp,  gops)
                ms_notmp += compute_ms_ssim(out_notmp, gops)

                count += 1

            mse_temp_list.append(mse_temp / count)
            msssim_temp_list.append(ms_temp / count)
            mse_notmp_list.append(mse_notmp / count)
            msssim_notmp_list.append(ms_notmp / count)

        results.append({
            'snr'               : snr,
            # With temporal fusion
            'psnr_temporal'     : compute_psnr_from_mse(np.mean(mse_temp_list)),
            'msssim_temporal'   : float(np.mean(msssim_temp_list)),
            'psnr_temporal_std' : float(np.std([compute_psnr_from_mse(m) for m in mse_temp_list])),
            # Without temporal fusion (per-frame ablation)
            'psnr_notemporal'   : compute_psnr_from_mse(np.mean(mse_notmp_list)),
            'msssim_notemporal' : float(np.mean(msssim_notmp_list)),
        })

        print(f"  SNR={snr:>2d} dB | "
              f"PSNR (w/ temporal): {results[-1]['psnr_temporal']:.2f} dB | "
              f"PSNR (no temporal): {results[-1]['psnr_notemporal']:.2f} dB | "
              f"MS-SSIM: {results[-1]['msssim_temporal']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Build test dataloader (shared across all checkpoints)
    print("Loading UCF101 test set...")
    _, test_loader = build_dataloaders(
        frames_root=FRAMES_ROOT,
        annotation_path=ANNOTATION_PATH,
        mode='gop',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        gop_size=GOP_SIZE,
        gops_per_clip=1,
        seed=SEED,
    )
    print(f"Test loader ready: {len(test_loader)} batches ({len(test_loader.dataset)} GoPs)\n")

    # Find all valid checkpoints (must have best.pkl)
    ckpt_dirs = sorted(glob.glob(os.path.join(CKPT_ROOT, 'VideoJSCC_*')))
    ckpt_dirs = [d for d in ckpt_dirs
                 if os.path.exists(os.path.join(d, 'best.pkl'))]
    print(f"Found {len(ckpt_dirs)} valid checkpoints\n")

    all_rows = []

    for ckpt_dir in ckpt_dirs:
        name = os.path.basename(ckpt_dir)
        info = parse_ckpt_name(name)

        if not all(k in info for k in ['c', 'snr', 'channel', 'ratio']):
            print(f"Skipping (can't parse): {name}")
            continue

        cbr_str = f"1/{int(round(1/info['ratio']))}"
        print(f"\n{'='*60}")
        print(f"Checkpoint: c={info['c']}, SNR_train={info['snr']}, "
              f"CBR={cbr_str}, {info['channel']}")
        print(f"{'='*60}")

        # Load model
        model = VideoJSCC(
            c=info['c'],
            channel_type=info['channel'],
            snr=info['snr'],
            n_frames=GOP_SIZE,
        ).to(device)

        ckpt_path = os.path.join(ckpt_dir, 'best.pkl')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Evaluate
        results = evaluate_checkpoint(
            model=model,
            test_loader=test_loader,
            snr_list=SNR_LIST,
            channel=info['channel'],
            repeats=REPEATS,
            device=device,
        )

        # Add metadata
        for r in results:
            r.update({
                'c'        : info['c'],
                'cbr'      : cbr_str,
                'snr_train': info['snr'],
                'channel'  : info['channel'],
            })
        all_rows.extend(results)

        del model
        torch.cuda.empty_cache()

    if not all_rows:
        print("No results collected.")
        return

    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'eval_results_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print summary table
    print(f"\n{'='*75}")
    print(f"  Week 3 — VideoJSCC Evaluation Summary (averaged over SNR_train models)")
    print(f"{'='*75}")
    print(f"{'CBR':<8} {'SNR':>6} {'PSNR+Temp':>12} {'PSNR-Temp':>12} "
          f"{'ΔPSNR':>8} {'MS-SSIM+T':>12}")
    print(f"{'-'*75}")

    for cbr in ['1/12', '1/6']:
        df_cbr = df[df['cbr'] == cbr]
        if df_cbr.empty:
            continue
        # Average over snr_train models for same cbr
        df_avg = df_cbr.groupby('snr').agg({
            'psnr_temporal'   : 'mean',
            'psnr_notemporal' : 'mean',
            'msssim_temporal' : 'mean',
        }).reset_index()

        for _, row in df_avg.iterrows():
            delta = row['psnr_temporal'] - row['psnr_notemporal']
            print(f"{cbr:<8} {row['snr']:>6.0f} "
                  f"{row['psnr_temporal']:>12.2f} "
                  f"{row['psnr_notemporal']:>12.2f} "
                  f"{delta:>+8.2f} "
                  f"{row['msssim_temporal']:>12.4f}")
        print(f"{'-'*75}")

    print(f"{'='*75}")
    print("ΔPSNR = temporal - no_temporal (positive = temporal fusion helps)")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cbr_colors = {'1/6': '#1f77b4', '1/12': '#ff7f0e'}

    for cbr in ['1/12', '1/6']:
        df_cbr = df[df['cbr'] == cbr]
        if df_cbr.empty:
            continue
        df_avg = df_cbr.groupby('snr').agg({
            'psnr_temporal'   : 'mean',
            'psnr_notemporal' : 'mean',
            'msssim_temporal' : 'mean',
            'msssim_notemporal': 'mean',
        }).reset_index().sort_values('snr')

        color = cbr_colors[cbr]

        # PSNR plot
        axes[0].plot(df_avg['snr'], df_avg['psnr_temporal'],
                     color=color, linestyle='-', linewidth=2, marker='o', markersize=4,
                     label=f'VideoJSCC CBR={cbr} (w/ temporal)')
        axes[0].plot(df_avg['snr'], df_avg['psnr_notemporal'],
                     color=color, linestyle='--', linewidth=2, marker='s', markersize=4,
                     label=f'Per-frame CBR={cbr} (no temporal)')

        # MS-SSIM plot
        axes[1].plot(df_avg['snr'], df_avg['msssim_temporal'],
                     color=color, linestyle='-', linewidth=2, marker='o', markersize=4,
                     label=f'VideoJSCC CBR={cbr} (w/ temporal)')
        axes[1].plot(df_avg['snr'], df_avg['msssim_notemporal'],
                     color=color, linestyle='--', linewidth=2, marker='s', markersize=4,
                     label=f'Per-frame CBR={cbr} (no temporal)')

    axes[0].set_title('PSNR vs SNR', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('SNR (dB)', fontsize=11)
    axes[0].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('MS-SSIM vs SNR', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('SNR (dB)', fontsize=11)
    axes[1].set_ylabel('MS-SSIM', fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Week 3 — VideoJSCC vs Per-frame DeepJSCC (AWGN)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'psnr_msssim_vs_snr.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'psnr_msssim_vs_snr.png')}")
    print("\nDone!")


if __name__ == '__main__':
    main()