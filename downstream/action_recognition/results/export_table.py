# -*- coding: utf-8 -*-
"""
Export results to CSV and print formatted tables.
Handles multiple methods (DeepJSCC, H.264+LDPC) in the same table.
"""

import os
import csv
import json
import numpy as np
from typing import List, Dict
from datetime import datetime


def save_results_csv(results: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not results:
        return
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[export] Raw results   → {output_path}  ({len(results)} rows)")


def save_summary_csv(summary: Dict, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows = list(summary.values())
    if not rows:
        return
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[export] Summary table → {output_path}")


def save_gap_examples_json(gap_examples: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(gap_examples, f, indent=2)
    print(f"[export] Gap examples  → {output_path}")


def print_results_table(summary: Dict, recognizer_keys: List[str] = ['c3d', 'sf']):
    """
    Print the main paper table:
    Method | SNR | PSNR | MS-SSIM | C3D Orig | C3D Recon | SF Orig | SF Recon
    """
    avail = [k for k in recognizer_keys
             if any(f'acc_recon_{k}' in v for v in summary.values())]

    # Column widths
    method_w = max(len(str(v['method'])) for v in summary.values()) + 2
    method_w = max(method_w, 16)

    header = f"{'Method':<{method_w}} | {'SNR':>6} | {'PSNR (dB)':>10} | {'MS-SSIM':>8}"
    for k in avail:
        name = {'c3d': 'C3D', 'sf': 'SlowFast', 'tsn': 'TSN'}.get(k, k.upper())
        header += f" | {name+' Orig%':>12} | {name+' Recon%':>12}"

    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    # Group by method for readability
    methods = sorted(set(v['method'] for v in summary.values()))
    for method in methods:
        method_rows = {k: v for k, v in summary.items() if v['method'] == method}
        for (m, snr, c), row in sorted(method_rows.items(), key=lambda x: x[0][1]):
            line = (f"{method:<{method_w}} | {snr:>6.1f} | "
                    f"{row['psnr_mean']:>10.2f} | {row['ms_ssim_mean']:>8.4f}")
            for k in avail:
                orig  = row.get(f'acc_orig_{k}',  float('nan'))
                recon = row.get(f'acc_recon_{k}', float('nan'))
                line += f" | {orig:>12.1f} | {recon:>12.1f}"
            print(line)
        print("-" * len(header))   # separator between methods

    print(sep)
    print(f"  n_videos per row: {list(summary.values())[0]['n_videos']}\n")


def print_gap_examples_table(gap_examples: List[Dict], recognizer_key: str = 'c3d'):
    if not gap_examples:
        return

    name = {'c3d': 'C3D', 'sf': 'SlowFast', 'tsn': 'TSN'}.get(recognizer_key, recognizer_key.upper())
    sep  = "=" * 95
    print(f"\n{sep}")
    print(f"  KEY RESULT: PSNR ≠ Task Accuracy Gap Examples  [{name}]")
    print(sep)
    print(f"  {'Method':<20} | {'SNR':>5} | {'Class (correct)':>20} | "
          f"{'PSNR':>6} | {'Class (incorrect)':>20} | {'PSNR':>6} | {'ΔPSNR':>6}")
    print("-" * 95)

    for ex in gap_examples:
        print(
            f"  {ex['method']:<20} | {ex['snr_db']:>5.1f} | "
            f"{ex['class_correct']:>20} | {ex['psnr_correct']:>6.2f} | "
            f"{ex['class_incorrect']:>20} | {ex['psnr_incorrect']:>6.2f} | "
            f"{ex['psnr_diff_db']:>6.3f}"
        )

    print(sep)
    print("  Rows: video pairs with SIMILAR PSNR but DIFFERENT accuracy.")
    print("  Demonstrates PSNR alone does not predict task performance.\n")


def save_all(
    results:         List[Dict],
    summary:         Dict,
    gap_examples:    List[Dict],
    output_dir:      str       = "downstream/action_recognition/results",
    tag:             str       = "",
    recognizer_keys: List[str] = ['c3d', 'sf'],
):
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{tag}" if tag else ""
    os.makedirs(output_dir, exist_ok=True)

    save_results_csv(results,
                     os.path.join(output_dir, f"raw_results{tag}_{ts}.csv"))
    save_summary_csv(summary,
                     os.path.join(output_dir, f"summary_table{tag}_{ts}.csv"))
    save_gap_examples_json(gap_examples,
                           os.path.join(output_dir, f"gap_examples{tag}_{ts}.json"))

    print_results_table(summary, recognizer_keys=recognizer_keys)
    if gap_examples:
        print_gap_examples_table(gap_examples, recognizer_key=recognizer_keys[0])