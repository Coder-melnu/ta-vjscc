# -*- coding: utf-8 -*-
"""
Action recognition evaluation loop.
Runs C3D and/or SlowFast on original vs reconstructed GoPs.
Supports both DeepJSCC and H.264+LDPC pipelines.
Finds the PSNR ≠ accuracy gap examples — the key paper result.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from downstream.action_recognition.pipeline.video_utils import (
    list_ucf101_videos, load_frames_from_dir, gop_tensor_to_numpy,
)
from downstream.action_recognition.evaluate.metrics import compute_gop_metrics


# ---------------------------------------------------------------------------
# Single-method evaluation loop
# ---------------------------------------------------------------------------

def evaluate_single_method(
    pipeline,
    method_name:        str,
    videos:             List,
    snr_list:           List[float],
    c3d_recognizer      = None,
    slowfast_recognizer = None,
    tsn_recognizer      = None,
    n_frames:           int  = 5,
    image_size:         int  = 128,
) -> List[Dict]:
    """
    Evaluate one method across all videos and SNR points.
    Works with both DeepJSCCPipeline and H264LDPCPipeline.
    Supports C3D, SlowFast, and TSN recognizers.
    """
    results = []

    for snr in snr_list:
        pipeline.set_snr(snr)
        print(f"\n[{method_name}] SNR = {snr} dB")

        counts = {'c3d_orig': 0, 'c3d_recon': 0,
                  'sf_orig':  0, 'sf_recon':  0,
                  'tsn_orig': 0, 'tsn_recon': 0, 'total': 0}

        for frame_dir, class_idx, class_name in tqdm(videos, desc=f"{method_name} SNR={snr}"):
            gop = load_frames_from_dir(frame_dir, n_frames=n_frames, image_size=image_size)
            if gop is None:
                continue

            # Reconstruct — handle both pipeline return types
            recon_output = pipeline.reconstruct_gop(gop.unsqueeze(0))
            if isinstance(recon_output, tuple):
                # H264LDPCPipeline: returns (tensor, metrics_dict)
                gop_recon, baseline_metrics = recon_output
                gop_recon = gop_recon.squeeze(0) if gop_recon.dim() == 4 else gop_recon
            else:
                # DeepJSCCPipeline: returns tensor only
                gop_recon = recon_output.squeeze(0)
                baseline_metrics = {}

            metrics = compute_gop_metrics(gop_recon, gop)

            result = {
                'method'    : method_name,
                'video_dir' : frame_dir,
                'class_idx' : class_idx,
                'class_name': class_name,
                'snr_db'    : snr,
                'psnr_db'   : metrics['psnr_db'],
                'ms_ssim'   : metrics['ms_ssim'],
                'c'         : getattr(pipeline, 'c', 0),
                'cbr'       : getattr(pipeline, 'cbr', None),
                'ber'       : baseline_metrics.get('ber', None),
            }

            if c3d_recognizer is not None:
                orig_np  = gop_tensor_to_numpy(gop)
                recon_np = gop_tensor_to_numpy(gop_recon)
                io, _, co = c3d_recognizer.predict_from_frames(orig_np)
                ir, _, cr = c3d_recognizer.predict_from_frames(recon_np)
                result.update({
                    'pred_orig_c3d'    : io,  'pred_recon_c3d'   : ir,
                    'conf_orig_c3d'    : co,  'conf_recon_c3d'   : cr,
                    'correct_orig_c3d' : (io == class_idx),
                    'correct_recon_c3d': (ir == class_idx),
                })
                if io == class_idx: counts['c3d_orig']  += 1
                if ir == class_idx: counts['c3d_recon'] += 1

            if slowfast_recognizer is not None:
                io_sf, _, co_sf = slowfast_recognizer.predict_from_gop(gop)
                ir_sf, _, cr_sf = slowfast_recognizer.predict_from_gop(gop_recon)
                result.update({
                    'pred_orig_sf'    : io_sf, 'pred_recon_sf'   : ir_sf,
                    'conf_orig_sf'    : co_sf, 'conf_recon_sf'   : cr_sf,
                    'correct_orig_sf' : (io_sf == class_idx),
                    'correct_recon_sf': (ir_sf == class_idx),
                })
                if io_sf == class_idx: counts['sf_orig']  += 1
                if ir_sf == class_idx: counts['sf_recon'] += 1

            if tsn_recognizer is not None:
                io_tsn, _, co_tsn = tsn_recognizer.predict_from_gop(gop)
                ir_tsn, _, cr_tsn = tsn_recognizer.predict_from_gop(gop_recon)
                result.update({
                    'pred_orig_tsn'    : io_tsn, 'pred_recon_tsn'   : ir_tsn,
                    'conf_orig_tsn'    : co_tsn, 'conf_recon_tsn'   : cr_tsn,
                    'correct_orig_tsn' : (io_tsn == class_idx),
                    'correct_recon_tsn': (ir_tsn == class_idx),
                })
                if io_tsn == class_idx: counts['tsn_orig']  += 1
                if ir_tsn == class_idx: counts['tsn_recon'] += 1

            results.append(result)
            counts['total'] += 1

        n = counts['total']
        if n > 0:
            if c3d_recognizer:
                print(f"  C3D      orig={counts['c3d_orig']/n*100:.1f}%  "
                      f"recon={counts['c3d_recon']/n*100:.1f}%")
            if slowfast_recognizer:
                print(f"  SlowFast orig={counts['sf_orig']/n*100:.1f}%  "
                      f"recon={counts['sf_recon']/n*100:.1f}%")
            if tsn_recognizer:
                print(f"  TSN      orig={counts['tsn_orig']/n*100:.1f}%  "
                      f"recon={counts['tsn_recon']/n*100:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Full evaluation — DeepJSCC + H.264+LDPC
# ---------------------------------------------------------------------------

def evaluate_action_recognition(
    deepjscc_pipelines:  List,
    h264_pipelines:      List,
    frames_root:         str,
    annotation_path:     str,
    split_file:          str,
    snr_list:            List[float],
    c3d_recognizer       = None,
    slowfast_recognizer  = None,
    tsn_recognizer       = None,
    n_frames:            int  = 5,
    image_size:          int  = 128,
    max_videos:          int  = 200,
) -> List[Dict]:
    """
    Evaluate all methods across all videos and SNR points.

    Args:
        deepjscc_pipelines : list of DeepJSCCPipeline (one per CBR checkpoint)
        h264_pipelines     : list of H264LDPCPipeline (one per CBR, matching DeepJSCC CBRs)
    """
    classind = os.path.join(annotation_path, 'classInd.txt')
    split    = os.path.join(annotation_path, split_file)
    videos   = list_ucf101_videos(frames_root, classind, split, max_videos)

    print(f"\n[action_eval] {len(videos)} videos × {len(snr_list)} SNR points")
    print(f"[action_eval] {len(deepjscc_pipelines)} DeepJSCC + "
          f"{len(h264_pipelines)} H264+LDPC pipelines")

    all_results = []

    for pipeline in deepjscc_pipelines:
        method_name = f"DeepJSCC_c{pipeline.c}"
        print(f"\n{'='*55}\n  {method_name}\n{'='*55}")
        all_results.extend(evaluate_single_method(
            pipeline, method_name, videos, snr_list,
            c3d_recognizer, slowfast_recognizer, tsn_recognizer,
            n_frames, image_size,
        ))

    for pipeline in h264_pipelines:
        method_name = f"H264+LDPC_cbr{pipeline.cbr:.4f}"
        print(f"\n{'='*55}\n  {method_name}\n{'='*55}")
        all_results.extend(evaluate_single_method(
            pipeline, method_name, videos, snr_list,
            c3d_recognizer, slowfast_recognizer, tsn_recognizer,
            n_frames, image_size,
        ))

    return all_results


# ---------------------------------------------------------------------------
# PSNR ≠ accuracy gap finder
# ---------------------------------------------------------------------------

def find_psnr_accuracy_gap_examples(
    results:        List[Dict],
    recognizer_key: str   = 'c3d',
    psnr_tolerance: float = 1.5,
    top_k:          int   = 10,
) -> List[Dict]:
    """
    Find video pairs with similar PSNR but different accuracy within same (method, SNR).
    This is the key motivating result for the paper.
    """
    correct_key = f'correct_recon_{recognizer_key}'
    conf_key    = f'conf_recon_{recognizer_key}'

    if not results or correct_key not in results[0]:
        print(f"[gap_finder] Key '{correct_key}' not in results.")
        return []

    groups = {}
    for r in results:
        groups.setdefault((r['method'], r['snr_db']), []).append(r)

    gap_examples = []
    for (method, snr), group in groups.items():
        correct_vids   = [r for r in group if r[correct_key]]
        incorrect_vids = [r for r in group if not r[correct_key]]
        for rc in correct_vids:
            for ri in incorrect_vids:
                psnr_diff = abs(rc['psnr_db'] - ri['psnr_db'])
                if psnr_diff <= psnr_tolerance:
                    gap_examples.append({
                        'method'          : method,
                        'snr_db'          : snr,
                        'psnr_diff_db'    : psnr_diff,
                        'video_correct'   : rc['video_dir'],
                        'class_correct'   : rc['class_name'],
                        'psnr_correct'    : rc['psnr_db'],
                        'ms_ssim_correct' : rc['ms_ssim'],
                        f'conf_correct_{recognizer_key}': rc[conf_key],
                        'video_incorrect' : ri['video_dir'],
                        'class_incorrect' : ri['class_name'],
                        'psnr_incorrect'  : ri['psnr_db'],
                        'ms_ssim_incorrect': ri['ms_ssim'],
                        f'conf_incorrect_{recognizer_key}': ri[conf_key],
                    })

    gap_examples.sort(key=lambda x: x['psnr_diff_db'])

    if gap_examples:
        ex = gap_examples[0]
        print(f"\n[gap_finder] {len(gap_examples)} examples. Best:")
        print(f"  Method={ex['method']}, SNR={ex['snr_db']} dB, ΔPSNR={ex['psnr_diff_db']:.3f} dB")
        print(f"  Correct  : {ex['class_correct']:25s}  PSNR={ex['psnr_correct']:.2f} dB")
        print(f"  Incorrect: {ex['class_incorrect']:25s}  PSNR={ex['psnr_incorrect']:.2f} dB")

    return gap_examples[:top_k]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    results:         List[Dict],
    recognizer_keys: List[str] = ['c3d', 'sf'],
) -> Dict:
    """Aggregate into summary table keyed by (method, snr, c)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r['method'], r['snr_db'], r['c'])].append(r)

    summary = {}
    for (method, snr, c), group in sorted(groups.items()):
        entry = {
            'method'      : method,
            'snr_db'      : snr,
            'c'           : c,
            'n_videos'    : len(group),
            'psnr_mean'   : float(np.mean([r['psnr_db'] for r in group])),
            'ms_ssim_mean': float(np.mean([r['ms_ssim']  for r in group])),
        }
        for key in recognizer_keys:
            if f'correct_orig_{key}' in group[0]:
                entry[f'acc_orig_{key}']  = float(np.mean(
                    [r[f'correct_orig_{key}']  for r in group])) * 100
                entry[f'acc_recon_{key}'] = float(np.mean(
                    [r[f'correct_recon_{key}'] for r in group])) * 100
        summary[(method, snr, c)] = entry

    return summary