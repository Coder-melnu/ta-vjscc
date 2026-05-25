# -*- coding: utf-8 -*-
"""
Extract Week 3 VideoJSCC training results from TensorBoard logs.
Produces:
  - CSV: val loss per epoch for all 10 models
  - CSV: best val loss summary table (c, snr, best_val_loss, best_epoch)
  - PNG: val loss curves grouped by CBR

Usage:
    cd ~/Nu/ta-vjscc
    python extract_week3_results.py

Output:
    week3_results/
    ├── val_loss_all.csv
    ├── best_val_loss_summary.csv
    └── val_loss_curves.png
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOGS_ROOT  = 'out_video/logs'
OUTPUT_DIR = 'week3_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parse checkpoint name
# ---------------------------------------------------------------------------

def parse_run_name(name: str) -> dict:
    info = {}
    m = re.search(r'_(AWGN|Rayleigh)_', name)
    if m: info['channel'] = m.group(1)
    m = re.search(r'_c(\d+)_', name)
    if m: info['c'] = int(m.group(1))
    m = re.search(r'_snr([\d.]+)_', name)
    if m: info['snr'] = float(m.group(1))
    m = re.search(r'_ratio([\d.]+)', name)
    if m: info['ratio'] = float(m.group(1))
    return info

# ---------------------------------------------------------------------------
# Extract TensorBoard scalars
# ---------------------------------------------------------------------------

def extract_scalars(log_dir: str, tag: str = 'val/loss'):
    """Extract scalar values for a given tag from a TensorBoard log directory."""
    ea = EventAccumulator(log_dir)
    ea.Reload()

    available = ea.Tags().get('scalars', [])
    if tag not in available:
        print(f"  WARNING: tag '{tag}' not found. Available: {available}")
        return [], []

    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_dirs = sorted(glob.glob(os.path.join(LOGS_ROOT, 'VideoJSCC_*')))
    print(f"Found {len(log_dirs)} log directories\n")

    all_rows    = []   # raw per-epoch data
    summary_rows = []  # best val loss per model

    for log_dir in log_dirs:
        name = os.path.basename(log_dir)
        info = parse_run_name(name)

        if not all(k in info for k in ['c', 'snr', 'channel']):
            print(f"Skipping (can't parse): {name}")
            continue

        print(f"Processing: c={info['c']}, SNR={info['snr']}, {info['channel']}")

        steps, values = extract_scalars(log_dir, tag='val/loss')

        if not values:
            print(f"  No val/loss data found — skipping")
            continue

        for step, val in zip(steps, values):
            all_rows.append({
                'run'    : name,
                'channel': info['channel'],
                'c'      : info['c'],
                'snr'    : info['snr'],
                'ratio'  : info.get('ratio', None),
                'epoch'  : step,
                'val_loss': val,
            })

        best_val  = min(values)
        best_epoch = steps[values.index(best_val)]
        summary_rows.append({
            'channel'       : info['channel'],
            'c'             : info['c'],
            'snr'           : info['snr'],
            'ratio'         : info.get('ratio', None),
            'best_val_loss' : best_val,
            'best_epoch'    : best_epoch,
            'total_epochs'  : len(values),
        })
        print(f"  Best val loss: {best_val:.6f} at epoch {best_epoch} "
              f"(total {len(values)} epochs)")

    if not all_rows:
        print("\nNo data extracted.")
        return

    # Save CSVs
    df_all     = pd.DataFrame(all_rows)
    df_summary = pd.DataFrame(summary_rows).sort_values(['c', 'snr'])

    all_csv     = os.path.join(OUTPUT_DIR, 'val_loss_all.csv')
    summary_csv = os.path.join(OUTPUT_DIR, 'best_val_loss_summary.csv')

    df_all.to_csv(all_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)

    print(f"\nSaved: {all_csv}")
    print(f"Saved: {summary_csv}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Week 3 — VideoJSCC Best Val Loss Summary")
    print(f"{'='*60}")
    print(f"{'CBR':<8} {'c':<5} {'SNR':>6} {'Best Val Loss':>14} {'Best Epoch':>11}")
    print(f"{'-'*60}")
    for _, row in df_summary.iterrows():
        cbr_str = f"1/{int(round(1/row['ratio']))}" if row['ratio'] else '-'
        print(f"{cbr_str:<8} {int(row['c']):<5} {row['snr']:>6.1f} "
              f"{row['best_val_loss']:>14.6f} {int(row['best_epoch']):>11}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------------------
    # Plot val loss curves
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    colors    = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    snr_list  = sorted(df_all['snr'].unique())
    color_map = {snr: colors[i] for i, snr in enumerate(snr_list)}

    for ax_idx, c_val in enumerate([4, 8]):
        ax  = axes[ax_idx]
        cbr = '1/12' if c_val == 4 else '1/6'
        df_c = df_all[df_all['c'] == c_val]

        for snr in sorted(df_c['snr'].unique()):
            df_snr = df_c[df_c['snr'] == snr].sort_values('epoch')
            ax.plot(
                df_snr['epoch'], df_snr['val_loss'],
                label=f'SNR={int(snr)} dB',
                color=color_map.get(snr, 'gray'),
                linewidth=2, marker='o', markersize=3,
            )

        ax.set_title(f'VideoJSCC — CBR={cbr} (c={c_val})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Val Loss (MSE)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle('Week 3 — VideoJSCC AWGN Val Loss Curves', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'val_loss_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()