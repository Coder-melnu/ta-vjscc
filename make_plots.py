# -*- coding: utf-8 -*-
"""
Publication-quality plots for Week 3 and Week 4 results.

Week 3 plots:
  1. PSNR vs SNR — VideoJSCC (temporal) vs Per-frame (no temporal), 2 CBR values
  2. MS-SSIM vs SNR — same comparison
  3. Val loss curves — training convergence

Week 4 plots:
  4. Top-1 Accuracy vs SNR — DeepJSCC c4 and c8
  5. PSNR vs SNR — DeepJSCC vs H.264 Intra baseline
  6. PSNR≠Accuracy gap illustration

Usage:
    cd ~/Nu/ta-vjscc
    python make_plots.py

Output:
    plots/
    ├── week3_psnr_vs_snr.png
    ├── week3_msssim_vs_snr.png
    ├── week3_val_loss.png
    ├── week4_accuracy_vs_snr.png
    ├── week4_psnr_comparison.png
    └── week4_psnr_accuracy_gap.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style — IEEE/publication quality
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette — consistent across all plots
COLORS = {
    'c8_temporal'   : '#1f77b4',   # blue   — VideoJSCC CBR=1/6 (with temporal)
    'c8_notemporal' : '#aec7e8',   # light blue — per-frame CBR=1/6
    'c4_temporal'   : '#ff7f0e',   # orange — VideoJSCC CBR=1/12 (with temporal)
    'c4_notemporal' : '#ffbb78',   # light orange — per-frame CBR=1/12
    'h264_cbr6'     : '#2ca02c',   # green  — H.264 CBR=1/6
    'h264_cbr12'    : '#98df8a',   # light green — H.264 CBR=1/12
}

MARKERS = {
    'temporal'   : 'o',
    'notemporal' : 's',
    'h264'       : '^',
}


# ---------------------------------------------------------------------------
# Week 3 — PSNR and MS-SSIM plots
# ---------------------------------------------------------------------------

def plot_week3_psnr_msssim(csv_path: str = 'week3_results/eval_results_all.csv'):
    if not os.path.exists(csv_path):
        print(f"[plots] {csv_path} not found — skipping Week 3 PSNR/MS-SSIM plots")
        return

    df = pd.read_csv(csv_path)

    # Average over snr_train models for same (cbr, snr)
    df_avg = df.groupby(['cbr', 'snr']).agg({
        'psnr_temporal'   : 'mean',
        'psnr_notemporal' : 'mean',
        'msssim_temporal' : 'mean',
        'msssim_notemporal': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for cbr, c_temporal, c_notemporal, label in [
        ('1/6',  COLORS['c8_temporal'], COLORS['c8_notemporal'], 'CBR=1/6'),
        ('1/12', COLORS['c4_temporal'], COLORS['c4_notemporal'], 'CBR=1/12'),
    ]:
        d = df_avg[df_avg['cbr'] == cbr].sort_values('snr')
        if d.empty:
            continue

        snr = d['snr'].values

        # PSNR
        axes[0].plot(snr, d['psnr_temporal'], color=c_temporal,
                     marker=MARKERS['temporal'], markersize=4, linewidth=1.8,
                     label=f'VideoJSCC {label} (w/ temporal)')
        axes[0].plot(snr, d['psnr_notemporal'], color=c_notemporal,
                     marker=MARKERS['notemporal'], markersize=4, linewidth=1.8,
                     linestyle='--', label=f'Per-frame {label} (no temporal)')

        # MS-SSIM
        axes[1].plot(snr, d['msssim_temporal'], color=c_temporal,
                     marker=MARKERS['temporal'], markersize=4, linewidth=1.8,
                     label=f'VideoJSCC {label} (w/ temporal)')
        axes[1].plot(snr, d['msssim_notemporal'], color=c_notemporal,
                     marker=MARKERS['notemporal'], markersize=4, linewidth=1.8,
                     linestyle='--', label=f'Per-frame {label} (no temporal)')

    axes[0].set_title('PSNR vs Channel SNR')
    axes[0].set_xlabel('Channel SNR (dB)')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].legend(loc='lower right')
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(5))

    axes[1].set_title('MS-SSIM vs Channel SNR')
    axes[1].set_xlabel('Channel SNR (dB)')
    axes[1].set_ylabel('MS-SSIM')
    axes[1].legend(loc='lower right')
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.suptitle('Week 3 — VideoJSCC vs Per-frame DeepJSCC (AWGN Channel)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'week3_psnr_msssim_vs_snr.png')
    plt.savefig(out)
    plt.close()
    print(f"[plots] Saved: {out}")


# ---------------------------------------------------------------------------
# Week 3 — Val loss curves
# ---------------------------------------------------------------------------

def plot_week3_val_loss(csv_path: str = 'week3_results/val_loss_all.csv'):
    if not os.path.exists(csv_path):
        print(f"[plots] {csv_path} not found — skipping val loss plot")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    snr_colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    snr_list   = sorted(df['snr'].unique())
    color_map  = {snr: snr_colors[i] for i, snr in enumerate(snr_list)}

    for ax_idx, (c_val, cbr_str) in enumerate([(4, '1/12'), (8, '1/6')]):
        ax   = axes[ax_idx]
        df_c = df[df['c'] == c_val]

        for snr in sorted(df_c['snr'].unique()):
            d = df_c[df_c['snr'] == snr].sort_values('epoch')
            ax.plot(d['epoch'], d['val_loss'],
                    color=color_map.get(snr, 'gray'),
                    linewidth=1.8, marker='o', markersize=2,
                    label=f'SNR={int(snr)} dB')

        ax.set_title(f'VideoJSCC CBR={cbr_str} (c={c_val})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss (MSE)')
        ax.legend(fontsize=8, loc='upper right')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle('Week 3 — VideoJSCC Training Convergence (AWGN)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'week3_val_loss_curves.png')
    plt.savefig(out)
    plt.close()
    print(f"[plots] Saved: {out}")


# ---------------------------------------------------------------------------
# Week 4 — TSN Accuracy vs SNR
# ---------------------------------------------------------------------------

def plot_week4_accuracy(
    csv_path: str = 'downstream/action_recognition/results/summary_table_AWGN_TSN.csv',
):
    # Find latest TSN summary table
    import glob
    files = sorted(glob.glob(
        'downstream/action_recognition/results/summary_table_AWGN_*.csv'
    ))
    if not files:
        print("[plots] No summary table found — skipping Week 4 accuracy plot")
        return
    csv_path = files[-1]   # use most recent
    print(f"[plots] Using: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check which accuracy column exists
    acc_col = None
    for col in ['acc_recon_tsn', 'acc_recon_sf', 'acc_recon_c3d']:
        if col in df.columns:
            acc_col = col
            orig_col = col.replace('recon', 'orig')
            break

    if acc_col is None:
        print("[plots] No accuracy column found — skipping")
        return

    rec_name = {'acc_recon_tsn': 'TSN', 'acc_recon_sf': 'SlowFast',
                'acc_recon_c3d': 'C3D'}.get(acc_col, 'Recognizer')

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color, marker in [
        ('DeepJSCC_c8', COLORS['c8_temporal'], 'o'),
        ('DeepJSCC_c4', COLORS['c4_temporal'], 's'),
    ]:
        d = df[df['method'] == method].sort_values('snr_db')
        if d.empty:
            continue
        cbr = '1/6' if 'c8' in method else '1/12'
        ax.plot(d['snr_db'], d[acc_col],
                color=color, marker=marker, markersize=5, linewidth=1.8,
                label=f'VideoJSCC CBR={cbr} (reconstructed)')

    # Baseline — original video accuracy (flat line)
    d_c8 = df[df['method'] == 'DeepJSCC_c8'].sort_values('snr_db')
    if not d_c8.empty and orig_col in d_c8.columns:
        orig_acc = d_c8[orig_col].iloc[0]
        ax.axhline(orig_acc, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Original video ({rec_name}: {orig_acc:.1f}%)')

    ax.set_title(f'Week 4 — Top-1 Action Recognition Accuracy vs SNR ({rec_name})')
    ax.set_xlabel('Channel SNR (dB)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.legend(loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'week4_accuracy_vs_snr.png')
    plt.savefig(out)
    plt.close()
    print(f"[plots] Saved: {out}")


# ---------------------------------------------------------------------------
# Week 4 — PSNR comparison: DeepJSCC vs H.264 Intra
# ---------------------------------------------------------------------------

def plot_week4_psnr_comparison(
    eval_csv:     str = 'week3_results/eval_results_all.csv',
    h264_cbr6:    str = 'results/baseline_h264_awgn_cbr0.1667.csv',
    h264_cbr12:   str = 'results/baseline_h264_awgn_cbr0.0833.csv',
):
    fig, ax = plt.subplots(figsize=(8, 5))

    # DeepJSCC PSNR from eval_results
    if os.path.exists(eval_csv):
        df = pd.read_csv(eval_csv)
        df_avg = df.groupby(['cbr', 'snr']).agg(
            {'psnr_temporal': 'mean'}).reset_index()

        for cbr, color, label in [
            ('1/6',  COLORS['c8_temporal'], 'VideoJSCC CBR=1/6'),
            ('1/12', COLORS['c4_temporal'], 'VideoJSCC CBR=1/12'),
        ]:
            d = df_avg[df_avg['cbr'] == cbr].sort_values('snr')
            if not d.empty:
                ax.plot(d['snr'], d['psnr_temporal'],
                        color=color, marker='o', markersize=4, linewidth=1.8,
                        label=label)

    # H.264 Intra baselines
    for h264_path, color, label in [
        (h264_cbr6,  COLORS['h264_cbr6'],  'H.264 Intra CBR=1/6'),
        (h264_cbr12, COLORS['h264_cbr12'], 'H.264 Intra CBR=1/12'),
    ]:
        if os.path.exists(h264_path):
            dh = pd.read_csv(h264_path)
            dh = dh[dh['psnr'] > 0].sort_values('snr_db')  # skip failed SNR=0
            if not dh.empty:
                ax.plot(dh['snr_db'], dh['psnr'],
                        color=color, marker='^', markersize=4, linewidth=1.8,
                        linestyle='--', label=label)

    ax.set_title('PSNR Comparison — VideoJSCC vs H.264 Intra+LDPC (AWGN)')
    ax.set_xlabel('Channel SNR (dB)')
    ax.set_ylabel('PSNR (dB)')
    ax.legend(loc='lower right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'week4_psnr_comparison.png')
    plt.savefig(out)
    plt.close()
    print(f"[plots] Saved: {out}")


# ---------------------------------------------------------------------------
# Week 4 — PSNR ≠ Accuracy gap illustration
# ---------------------------------------------------------------------------

def plot_week4_psnr_accuracy_gap(
    gap_json: str = None,
    summary_csv: str = None,
):
    import glob

    # Find latest gap examples
    if gap_json is None:
        files = sorted(glob.glob(
            'downstream/action_recognition/results/gap_examples_AWGN_*.json'
        ))
        if not files:
            print("[plots] No gap examples found — skipping gap plot")
            return
        gap_json = files[-1]

    # Find latest summary
    if summary_csv is None:
        files = sorted(glob.glob(
            'downstream/action_recognition/results/summary_table_AWGN_*.csv'
        ))
        if not files:
            return
        summary_csv = files[-1]

    with open(gap_json) as f:
        gaps = json.load(f)

    df = pd.read_csv(summary_csv)

    # Check accuracy column
    acc_col = None
    for col in ['acc_recon_tsn', 'acc_recon_sf', 'acc_recon_c3d']:
        if col in df.columns:
            acc_col = col
            break
    if acc_col is None:
        print("[plots] No accuracy column — skipping gap plot")
        return

    rec_name = {'acc_recon_tsn': 'TSN', 'acc_recon_sf': 'SlowFast',
                'acc_recon_c3d': 'C3D'}.get(acc_col, 'Recognizer')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: PSNR vs Accuracy scatter for c8
    d_c8 = df[df['method'] == 'DeepJSCC_c8'].sort_values('snr_db')
    if not d_c8.empty:
        axes[0].scatter(d_c8['psnr_mean'], d_c8[acc_col],
                        c=d_c8['snr_db'], cmap='viridis',
                        s=60, zorder=5, label='DeepJSCC CBR=1/6')
        # Add SNR annotations for key points
        for _, row in d_c8[d_c8['snr_db'].isin([0, 5, 10, 15, 20, 25])].iterrows():
            axes[0].annotate(f"{int(row['snr_db'])}dB",
                             (row['psnr_mean'], row[acc_col]),
                             textcoords='offset points', xytext=(5, 3),
                             fontsize=7, color='gray')

    d_c4 = df[df['method'] == 'DeepJSCC_c4'].sort_values('snr_db')
    if not d_c4.empty:
        axes[0].scatter(d_c4['psnr_mean'], d_c4[acc_col],
                        c=d_c4['snr_db'], cmap='plasma',
                        s=60, marker='s', zorder=5, label='DeepJSCC CBR=1/12')

    axes[0].set_title(f'PSNR vs {rec_name} Accuracy\n(each point = one SNR value)')
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_ylabel(f'Top-1 Accuracy (%)')
    axes[0].legend()

    # Right: Gap examples bar chart
    if gaps:
        top5 = gaps[:5]
        x    = np.arange(len(top5))
        w    = 0.35

        psnr_correct   = [g['psnr_correct']   for g in top5]
        psnr_incorrect = [g['psnr_incorrect']  for g in top5]
        labels_c = [g['class_correct'][:10]   for g in top5]
        labels_i = [g['class_incorrect'][:10] for g in top5]

        axes[1].bar(x - w/2, psnr_correct,   w, label='Correct class',
                    color=COLORS['c8_temporal'], alpha=0.8)
        axes[1].bar(x + w/2, psnr_incorrect, w, label='Wrong class',
                    color=COLORS['c4_temporal'], alpha=0.8)

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [f"{c}\nvs\n{i}" for c, i in zip(labels_c, labels_i)],
            fontsize=7
        )
        axes[1].set_title(f'PSNR ≠ Accuracy Gap Examples\n(similar PSNR, different accuracy)')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].legend()
        snr_labels = [f"SNR={g['snr_db']}dB" for g in top5]
        for i, lbl in enumerate(snr_labels):
            axes[1].text(i, max(psnr_correct[i], psnr_incorrect[i]) + 0.2,
                         lbl, ha='center', fontsize=7, color='gray')

    plt.suptitle(f'Week 4 — PSNR ≠ Task Accuracy ({rec_name} Recognizer)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'week4_psnr_accuracy_gap.png')
    plt.savefig(out)
    plt.close()
    print(f"[plots] Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Generating publication-quality plots...\n")

    # Week 3
    print("--- Week 3 ---")
    plot_week3_psnr_msssim()
    plot_week3_val_loss()

    # Week 4
    print("\n--- Week 4 ---")
    plot_week4_accuracy()
    plot_week4_psnr_comparison()
    plot_week4_psnr_accuracy_gap()

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print("Files:")
    import glob
    for f in sorted(glob.glob(f'{OUTPUT_DIR}/*.png')):
        print(f"  {f}")