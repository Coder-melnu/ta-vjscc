# -*- coding: utf-8 -*-
"""
plot_ta_results.py — Week 5 TA-VideoJSCC Result Plots

Reads TensorBoard event files from out_ta_video/logs/ and generates:
    Plot 1 — Joint loss curves (train + val) per SNR
    Plot 2 — Task loss (L_task) vs epoch per SNR
    Plot 3 — Reconstruction loss (L_recon) vs epoch per SNR
    Plot 4 — Rate mean (feature retention) vs epoch per SNR
    Plot 5 — Val top-1 accuracy vs epoch per SNR
    Plot 6 — Final metrics summary table (PSNR proxy + accuracy at epoch 100)

Usage:
    conda activate ta-vjscc
    python plot_ta_results.py --out_dir ./out_ta_video --save_dir ./plots_ta

    # While training is still running (partial results):
    python plot_ta_results.py --out_dir ./out_ta_video --save_dir ./plots_ta --partial
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Style — consistent with Week 3/4 plots (navy/ice blue palette)
# ---------------------------------------------------------------------------

NAVY      = '#1B2A4A'
COLORS    = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
LINESTY   = ['-', '--', '-.', ':', (0,(3,1,1,1))]
FONT_SIZE = 12

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : FONT_SIZE,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 12,
    'legend.fontsize'  : 10,
    'figure.dpi'       : 150,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# TensorBoard reader
# ---------------------------------------------------------------------------

def read_tb_scalar(log_dir: str, tag: str):
    """Read a scalar tag from a TensorBoard event file."""
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return [], []
    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values


def parse_run_dir(log_dir: str) -> dict:
    """
    Parse all relevant scalars from one TensorBoard log directory.
    Returns dict: tag → (steps, values)
    """
    tags = [
        'train/loss', 'train/l_task', 'train/l_recon',
        'train/l_rate', 'train/rate_mean', 'train/score_mean',
        'val/loss',   'val/l_task',   'val/l_recon',
        'val/rate_mean', 'val/top1_acc',
        'tau', 'learning_rate',
    ]
    data = {}
    for tag in tags:
        steps, values = read_tb_scalar(log_dir, tag)
        if steps:
            data[tag] = (np.array(steps), np.array(values))
    return data


def extract_snr_from_dirname(dirname: str) -> float:
    """Extract SNR value from directory name like TAVideoJSCC_AWGN_c8_snr13.0_..."""
    import re
    m = re.search(r'_snr([\d.]+)_', dirname)
    return float(m.group(1)) if m else -1.0


# ---------------------------------------------------------------------------
# Load all runs
# ---------------------------------------------------------------------------

def load_all_runs(out_dir: str) -> list:
    """
    Load all TAVideoJSCC training runs from out_dir/logs/.
    Returns list of dicts sorted by SNR.
    """
    log_root = os.path.join(out_dir, 'logs')
    if not os.path.exists(log_root):
        raise FileNotFoundError(f"No logs directory found at {log_root}")

    run_dirs = sorted(glob.glob(os.path.join(log_root, 'TAVideoJSCC_*')))
    if not run_dirs:
        raise FileNotFoundError(f"No TAVideoJSCC runs found in {log_root}")

    runs = []
    for d in run_dirs:
        snr  = extract_snr_from_dirname(os.path.basename(d))
        data = parse_run_dir(d)
        if data:
            runs.append({'snr': snr, 'data': data, 'dir': d})
            print(f"  Loaded SNR={snr:5.1f} dB — {len(data)} tags, "
                  f"  {len(data.get('train/loss', ([],[]))[0])} epochs")

    runs.sort(key=lambda x: x['snr'])
    return runs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def smooth(values, window=5):
    """Simple moving average for noisy curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='same')


def make_fig(title, xlabel, ylabel, figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontweight='bold', color=NAVY, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def save_fig(fig, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_loss_curves(runs, save_dir, tag_train='train/loss', tag_val='val/loss',
                     title='Joint Loss (Train & Val)', filename='loss_joint.png'):
    fig, ax = make_fig(title, 'Epoch', 'Loss')
    for i, run in enumerate(runs):
        snr   = run['snr']
        color = COLORS[i % len(COLORS)]
        ls    = LINESTY[i % len(LINESTY)]
        data  = run['data']
        if tag_train in data:
            steps, vals = data[tag_train]
            ax.plot(steps, smooth(vals), color=color, ls=ls,
                    label=f'SNR={snr:.0f}dB train', linewidth=1.5)
        if tag_val in data:
            steps, vals = data[tag_val]
            ax.plot(steps, smooth(vals), color=color, ls=ls, alpha=0.5,
                    label=f'SNR={snr:.0f}dB val', linewidth=1.0)
    ax.legend(loc='upper right', ncol=2, framealpha=0.7)
    save_fig(fig, save_dir, filename)


def plot_single_tag(runs, tag, title, ylabel, filename, save_dir,
                    split='val', pct=False):
    fig, ax = make_fig(title, 'Epoch', ylabel)
    for i, run in enumerate(runs):
        snr   = run['snr']
        color = COLORS[i % len(COLORS)]
        ls    = LINESTY[i % len(LINESTY)]
        key   = f'{split}/{tag}'
        if key in run['data']:
            steps, vals = run['data'][key]
            if pct:
                vals = vals * 100
            ax.plot(steps, smooth(vals), color=color, ls=ls,
                    label=f'SNR={snr:.0f} dB', linewidth=2.0)
    ax.legend(loc='best', framealpha=0.7)
    save_fig(fig, save_dir, filename)


def plot_task_loss(runs, save_dir):
    plot_single_tag(runs, 'l_task', 'Task Loss (Cross-Entropy) vs Epoch',
                    'L_task', 'loss_task.png', save_dir, split='val')


def plot_recon_loss(runs, save_dir):
    plot_single_tag(runs, 'l_recon', 'Reconstruction Loss (MSE) vs Epoch',
                    'L_recon', 'loss_recon.png', save_dir, split='val')


def plot_rate_mean(runs, save_dir):
    fig, ax = make_fig('Feature Retention Rate vs Epoch',
                       'Epoch', 'Mean Selection Rate (fraction kept)')
    for i, run in enumerate(runs):
        snr   = run['snr']
        color = COLORS[i % len(COLORS)]
        ls    = LINESTY[i % len(LINESTY)]
        # Use train rate_mean — more stable than val
        key = 'train/rate_mean'
        if key in run['data']:
            steps, vals = run['data'][key]
            ax.plot(steps, smooth(vals), color=color, ls=ls,
                    label=f'SNR={snr:.0f} dB', linewidth=2.0)
    ax.axhline(y=1.0, color='gray', ls='--', alpha=0.4, label='No selection (rate=1)')
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc='best', framealpha=0.7)
    save_fig(fig, save_dir, 'rate_mean.png')


def plot_accuracy(runs, save_dir):
    fig, ax = make_fig('Val Top-1 Action Recognition Accuracy vs Epoch',
                       'Epoch', 'Top-1 Accuracy (%)')
    for i, run in enumerate(runs):
        snr   = run['snr']
        color = COLORS[i % len(COLORS)]
        ls    = LINESTY[i % len(LINESTY)]
        key   = 'val/top1_acc'
        if key in run['data']:
            steps, vals = run['data'][key]
            ax.plot(steps, smooth(vals * 100, window=7), color=color, ls=ls,
                    label=f'SNR={snr:.0f} dB', linewidth=2.0)
    ax.axhline(y=1.0, color='gray', ls=':', alpha=0.5, label='Random (1/101 ≈ 1%)')
    ax.legend(loc='upper left', framealpha=0.7)
    save_fig(fig, save_dir, 'val_accuracy.png')


def plot_temperature(runs, save_dir):
    fig, ax = make_fig('Gumbel Temperature Annealing',
                       'Epoch', 'Temperature τ')
    # All runs share the same schedule — just plot one
    if runs and 'tau' in runs[0]['data']:
        steps, vals = runs[0]['data']['tau']
        ax.plot(steps, vals, color=NAVY, linewidth=2.0)
        ax.set_ylim(0, 1.1)
    save_fig(fig, save_dir, 'temperature.png')


# ---------------------------------------------------------------------------
# Summary table plot
# ---------------------------------------------------------------------------

def plot_summary_table(runs, save_dir):
    """
    Bar chart: final val_acc and final rate_mean per SNR.
    Shows the key Week 5 result at a glance.
    """
    snrs      = []
    final_acc = []
    final_rate= []

    for run in runs:
        snrs.append(run['snr'])
        if 'val/top1_acc' in run['data']:
            _, vals = run['data']['val/top1_acc']
            final_acc.append(vals[-1] * 100)
        else:
            final_acc.append(0.0)
        if 'train/rate_mean' in run['data']:
            _, vals = run['data']['train/rate_mean']
            final_rate.append(vals[-1] * 100)
        else:
            final_rate.append(100.0)

    x     = np.arange(len(snrs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, final_acc,  width, color='#2196F3',
                    alpha=0.85, label='Val Top-1 Acc (%)')
    bars2 = ax2.bar(x + width/2, final_rate, width, color='#FF5722',
                    alpha=0.85, label='Feature Retention (%)')

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Top-1 Accuracy (%)', color='#2196F3')
    ax2.set_ylabel('Feature Retention (%)', color='#FF5722')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{s:.0f}' for s in snrs])
    ax1.set_title('TA-VideoJSCC: Final Accuracy & Feature Retention vs SNR',
                  fontweight='bold', color=NAVY)

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=9, color='#2196F3')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=9, color='#FF5722')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               framealpha=0.7)
    ax1.axhline(y=1.0, color='gray', ls='--', alpha=0.4, linewidth=1)
    ax1.text(len(snrs)-0.5, 1.2, 'random baseline', fontsize=8,
             color='gray', ha='right')

    fig.tight_layout()
    save_fig(fig, save_dir, 'summary_bar.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot TA-VideoJSCC Week 5 results')
    parser.add_argument('--out_dir',  default='./out_ta_video',
                        help='Training output directory')
    parser.add_argument('--save_dir', default='./plots_ta',
                        help='Where to save plots')
    parser.add_argument('--partial',  action='store_true',
                        help='Allow plotting even if training is still running')
    args = parser.parse_args()

    print(f"\nLoading runs from: {args.out_dir}")
    runs = load_all_runs(args.out_dir)

    if not runs:
        print("No completed runs found. Make sure training has started.")
        return

    print(f"\nFound {len(runs)} run(s): SNRs = {[r['snr'] for r in runs]}")
    print(f"Saving plots to: {args.save_dir}\n")

    print("Generating plots...")
    plot_loss_curves(runs, args.save_dir)
    plot_task_loss(runs, args.save_dir)
    plot_recon_loss(runs, args.save_dir)
    plot_rate_mean(runs, args.save_dir)
    plot_accuracy(runs, args.save_dir)
    plot_temperature(runs, args.save_dir)
    plot_summary_table(runs, args.save_dir)

    print(f"\nDone. All plots saved to {args.save_dir}/")
    print("Files:")
    for f in sorted(os.listdir(args.save_dir)):
        print(f"  {f}")


if __name__ == '__main__':
    main()