#!/usr/bin/env python3
"""
generate_gap_table.py
Generates the PSNR ≠ Task Accuracy motivating example table for the paper.

Reads gap_examples_AWGN_20260512_020649.json and produces:
  1. A formatted CSV table of the best examples
  2. A publication-quality matplotlib figure

Usage:
    conda activate action_eval
    python generate_gap_table.py \
        --json downstream/action_recognition/results/gap_examples_AWGN_20260512_020649.csv \
        --out plots/gap_examples/
"""

import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# ── Style ─────────────────────────────────────────────────────────────────────
NAVY  = '#1B2A4A'
GREEN = '#2E7D32'
RED   = '#C62828'
GRAY  = '#757575'

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 12,
    'figure.dpi'       : 150,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ── Best examples to highlight (selected for paper) ───────────────────────────
# Criteria: identical PSNR, clear action-type explanation, large confidence gap
SELECTED_SNRS = [7.0, 9.0, 16.0]   # The three most compelling pairs

# Human-readable explanations for why one survives and one fails
EXPLANATIONS = {
    7.0: {
        'correct_why'  : 'Finger motion on keyboard — distinctive, localised, survives compression',
        'incorrect_why': 'Rapid rotational body motion — fine spatial detail lost at low SNR',
    },
    9.0: {
        'correct_why'  : 'Large-scale body arc trajectory — high spatial energy, robust to noise',
        'incorrect_why': 'Subtle finger positions on flute — fine-grained motion lost in reconstruction',
    },
    16.0: {
        'correct_why'  : 'Uniform marching motion — strong periodic temporal pattern survives',
        'incorrect_why': 'Upper-body pull motion — background clutter dominates reconstructed frames',
    },
}


def load_examples(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {ex['snr_db']: ex for ex in data}


def format_action(name):
    """CamelCase → spaced: 'SalsaSpin' → 'Salsa Spin'"""
    import re
    return re.sub(r'([A-Z])', r' \1', name).strip()


def build_csv(examples, out_path):
    rows = []
    for snr in SELECTED_SNRS:
        ex = examples.get(snr)
        if not ex:
            continue
        rows.append({
            'SNR (dB)'              : snr,
            'PSNR (dB)'             : f"{ex['psnr_correct']:.2f}",
            'MS-SSIM (correct)'     : f"{ex['ms_ssim_correct']:.4f}",
            'MS-SSIM (incorrect)'   : f"{ex['ms_ssim_incorrect']:.4f}",
            'Video A (correct)'     : format_action(ex['class_correct']),
            'Conf A (%)'            : f"{ex['conf_correct_sf']*100:.1f}",
            'Classified A'          : 'Correct ✓',
            'Video B (incorrect)'   : format_action(ex['class_incorrect']),
            'Conf B (%)'            : f"{ex['conf_incorrect_sf']*100:.1f}",
            'Classified B'          : 'Wrong ✗',
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {out_path}")
    return rows


def build_figure(examples, out_path):
    """
    Horizontal bar chart showing confidence scores for correct vs incorrect
    video at identical PSNR — the key motivating result.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, len(SELECTED_SNRS), figsize=(14, 4.5),
                              sharey=False)
    fig.suptitle(
        'PSNR \u2260 Task Accuracy: Videos with Identical PSNR, Different Classification Outcome\n'
        '(VideoJSCC, CBR=1/6, AWGN Channel)',
        fontweight='bold', color=NAVY, fontsize=13, y=1.02
    )

    for ax, snr in zip(axes, SELECTED_SNRS):
        ex = examples.get(snr)
        if not ex:
            continue

        action_c = format_action(ex['class_correct'])
        action_i = format_action(ex['class_incorrect'])
        conf_c   = ex['conf_correct_sf'] * 100
        conf_i   = ex['conf_incorrect_sf'] * 100
        psnr     = ex['psnr_correct']
        ms_c     = ex['ms_ssim_correct']
        ms_i     = ex['ms_ssim_incorrect']

        y    = [1, 0]
        vals = [conf_c, conf_i]
        cols = [GREEN, RED]
        lbls = [f'{action_c}\n(correct ✓)', f'{action_i}\n(wrong ✗)']

        bars = ax.barh(y, vals, color=cols, alpha=0.85, height=0.5)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(0, 115)
        ax.set_yticks(y)
        ax.set_yticklabels(lbls, fontsize=10)
        ax.set_xlabel('Classifier Confidence (%)', fontsize=10)
        ax.set_title(
            f'SNR = {snr:.0f} dB\n'
            f'PSNR = {psnr:.2f} dB (identical)\n'
            f'MS-SSIM: {ms_c:.3f} vs {ms_i:.3f}',
            fontsize=10, color=NAVY
        )
        ax.axvline(x=50, color=GRAY, ls='--', alpha=0.4, lw=1)
        ax.text(51, -0.35, '50%\nthreshold', fontsize=8, color=GRAY)
        ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def build_paper_table_figure(examples, out_path):
    """
    Clean table-style figure for paper: each row is one pair.
    More suitable for embedding in the Word report.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    col_labels = [
        'SNR\n(dB)', 'PSNR\n(dB)',
        'Video A\n(Action class)', 'MS-SSIM A', 'Classified A',
        'Video B\n(Action class)', 'MS-SSIM B', 'Classified B',
        'PSNR gap\n(A vs B)'
    ]

    table_data = []
    for snr in SELECTED_SNRS:
        ex = examples.get(snr)
        if not ex:
            continue
        table_data.append([
            f'{snr:.0f}',
            f'{ex["psnr_correct"]:.2f}',
            format_action(ex['class_correct']),
            f'{ex["ms_ssim_correct"]:.4f}',
            'Correct ✓',
            format_action(ex['class_incorrect']),
            f'{ex["ms_ssim_incorrect"]:.4f}',
            'Wrong ✗',
            '0.00 dB',
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor(NAVY)
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style correct/incorrect columns and alternating rows
    for i, row in enumerate(table_data):
        shade = '#F5F5F5' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(shade)
        # Correct ✓ cell
        table[i+1, 4].set_facecolor('#E8F5E9')
        table[i+1, 4].set_text_props(color=GREEN, fontweight='bold')
        # Wrong ✗ cell
        table[i+1, 7].set_facecolor('#FFEBEE')
        table[i+1, 7].set_text_props(color=RED, fontweight='bold')
        # PSNR gap = 0
        table[i+1, 8].set_text_props(fontweight='bold', color='#1565C0')

    fig.suptitle(
        'Table: PSNR \u2260 Task Accuracy \u2014 Video Pairs with Identical PSNR but Different '
        'Classification Outcome\n'
        '(VideoJSCC, c=8, CBR=1/6, AWGN)',
        fontweight='bold', color=NAVY, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f"Paper table figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='downstream/action_recognition/results/gap_examples_AWGN_20260512_020649.json')
    parser.add_argument('--out',  default='plots/gap_examples')
    args = parser.parse_args()

    print(f"Loading: {args.json}")
    examples = load_examples(args.json)
    print(f"Found {len(examples)} gap examples at SNRs: {sorted(examples.keys())}")

    # CSV table
    build_csv(examples, os.path.join(args.out, 'gap_examples_table.csv'))

    # Bar chart figure (for presentation/report)
    build_figure(examples, os.path.join(args.out, 'gap_examples_bars.png'))

    # Clean table figure (for paper/Word report)
    build_paper_table_figure(examples, os.path.join(args.out, 'gap_examples_paper_table.png'))

    print("\nDone. Files generated:")
    for f in ['gap_examples_table.csv', 'gap_examples_bars.png', 'gap_examples_paper_table.png']:
        print(f"  {args.out}/{f}")

    print("\nKey findings:")
    for snr in SELECTED_SNRS:
        ex = examples.get(snr)
        if ex:
            print(f"\n  SNR={snr:.0f} dB | PSNR={ex['psnr_correct']:.2f} dB (identical)")
            print(f"    ✓ {format_action(ex['class_correct']):<20} conf={ex['conf_correct_sf']*100:.1f}%  MS-SSIM={ex['ms_ssim_correct']:.4f}")
            print(f"    ✗ {format_action(ex['class_incorrect']):<20} conf={ex['conf_incorrect_sf']*100:.1f}%  MS-SSIM={ex['ms_ssim_incorrect']:.4f}")


if __name__ == '__main__':
    main()