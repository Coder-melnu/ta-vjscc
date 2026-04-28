#!/usr/bin/env python3
"""
Run H.264/H.265 + LDPC baseline on frames pooled from multiple UCF101 clips.

Collects frames from N random clips across different classes into a single
folder, then runs the baseline on the combined set to produce a larger
bitstream that gives more realistic LDPC behavior.

Usage:
    # Bandwidth-matched to DeepJSCC CBR=1/6
    python baselines/run_baseline_multi.py \
        --test_root /path/to/UCF101Frames/test \
        --cbr 0.1667 \
        --out results/baseline_h264_awgn_cbr16.csv

    # Bandwidth-matched to DeepJSCC CBR=1/12
    python baselines/run_baseline_multi.py \
        --test_root /path/to/UCF101Frames/test \
        --cbr 0.0833 \
        --out results/baseline_h264_awgn_cbr112.csv

    # Unconstrained CRF (not for paper comparison)
    python baselines/run_baseline_multi.py \
        --test_root /path/to/UCF101Frames/test \
        --crf 23
"""

import os
import sys
import random
import shutil
import argparse
import tempfile
from pathlib import Path


def collect_frames(test_root, num_frames, image_size=256, seed=42):
    """
    Randomly sample num_frames frames from the entire UCF101 test set
    and copy into a single temporary directory as sequential PNGs.

    Returns:
        (tmp_dir_path, num_frames_collected)
    """
    random.seed(seed)

    exts = {'.png', '.jpg', '.jpeg'}
    all_frames = []
    for cls_dir in sorted(Path(test_root).iterdir()):
        if not cls_dir.is_dir():
            continue
        for clip_dir in sorted(cls_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            frames = [f for f in clip_dir.iterdir() if f.suffix.lower() in exts]
            all_frames.extend(frames)

    print(f"Total frames in test set: {len(all_frames)}")

    if len(all_frames) <= num_frames:
        selected = all_frames
    else:
        selected = random.sample(all_frames, num_frames)

    print(f"Sampled {len(selected)} frames")

    tmp_dir = tempfile.mkdtemp(prefix='ucf101_pooled_')
    from PIL import Image

    for i, f in enumerate(sorted(selected)):
        img = Image.open(f).convert('RGB').resize((image_size, image_size))
        img.save(os.path.join(tmp_dir, f'{i+1:04d}.png'))

    num_collected = len(selected)
    print(f"Frames saved to: {tmp_dir}")
    return tmp_dir, num_collected


def main():
    parser = argparse.ArgumentParser(
        description='Run H.264+LDPC baseline on pooled UCF101 clips')

    parser.add_argument('--test_root', required=True,
                        help='UCF101 test frames root (contains class folders)')
    parser.add_argument('--codec', default='h264', choices=['h264', 'h265'])
    parser.add_argument('--channel', default='AWGN', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr_list', nargs='+', type=float,
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 20, 21, 22, 23, 24, 25])
    parser.add_argument('--num_frames', type=int, default=500,
                        help='Number of frames to sample (default: 500)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ldpc_rate', type=float, default=0.6667,
                        help='LDPC code rate (default: 2/3)')
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--repeats', type=int, default=5,
                        help='Monte Carlo repeats per SNR point (default: 5)')
    parser.add_argument('--seed', type=int, default=42)

    # Bandwidth control — use ONE of these:
    parser.add_argument('--cbr', type=float, default=None,
                        help='DeepJSCC CBR to match (e.g., 0.1667 for 1/6)')
    parser.add_argument('--target_bpp', type=float, default=None,
                        help='Target source BPP for 2-pass encoding')
    parser.add_argument('--crf', type=int, default=23,
                        help='CRF quality (fallback if no cbr/target_bpp set)')

    parser.add_argument('--out', default=None)

    args = parser.parse_args()

    # Compute target_bpp from CBR if specified
    if args.cbr is not None:
        args.target_bpp = args.cbr * 2 * args.ldpc_rate
        print(f"CBR={args.cbr:.4f} → target_bpp={args.target_bpp:.4f} "
              f"(with LDPC R={args.ldpc_rate:.4f})")

    # Auto-generate output filename
    if args.out is None:
        bw_tag = f'cbr{args.cbr:.4f}' if args.cbr else (
            f'bpp{args.target_bpp:.3f}' if args.target_bpp else f'crf{args.crf}')
        args.out = f'./results/baseline_{args.codec}_{args.channel.lower()}_{bw_tag}.csv'

    # Determine encoding mode
    if args.target_bpp is not None:
        mode = f'2-pass target bitrate (source BPP={args.target_bpp:.4f})'
    else:
        mode = f'CRF={args.crf} (unconstrained — NOT bandwidth-matched)'

    # Step 1: Pool frames
    print(f"\n{'='*72}")
    print(f"Collecting {args.num_frames} frames from test set...")
    print(f"{'='*72}\n")

    pooled_dir, num_frames = collect_frames(
        args.test_root, args.num_frames,
        image_size=args.image_size, seed=args.seed)

    # Step 2: Run baseline on pooled frames
    sys.path.insert(0, os.path.dirname(__file__))
    from h264_ldpc_baseline import run_baseline

    print(f"\n{'='*72}")
    print(f"Baseline : {args.codec.upper()} intra + LDPC (R={args.ldpc_rate:.4f})")
    print(f"Encoding : {mode}")
    if args.target_bpp:
        target_kbps = int(args.target_bpp * args.image_size * args.image_size * args.fps / 1000)
        print(f"Target   : BPP={args.target_bpp:.4f} → {target_kbps} kbps")
    print(f"Channel  : {args.channel}")
    print(f"SNR list : {args.snr_list}")
    print(f"Frames   : {num_frames}")
    print(f"Repeats  : {args.repeats}")
    print(f"Output   : {args.out}")
    print(f"{'='*72}\n")

    results = []
    for snr in args.snr_list:
        print(f"\n--- SNR = {snr} dB ---")
        psnr_runs = []
        msssim_runs = []
        ber_runs = []

        for rep in range(args.repeats):
            r = run_baseline(
                frames_dir=pooled_dir,
                codec=args.codec,
                channel_type=args.channel,
                snr_db=snr,
                ldpc_rate=args.ldpc_rate,
                image_size=args.image_size,
                max_frames=num_frames,
                target_bpp=args.target_bpp,
                crf=args.crf,
                fps=args.fps,
            )
            psnr_runs.append(r['psnr'])
            msssim_runs.append(r['msssim'])
            ber_runs.append(r['ber'])
            print(f"    Run {rep+1}/{args.repeats}: "
                  f"PSNR={r['psnr']:.2f} MS-SSIM={r['msssim']:.4f} BER={r['ber']:.6f}")

        import numpy as np
        avg_result = {
            'snr_db': snr,
            'codec': args.codec,
            'channel': args.channel,
            'psnr': float(np.mean(psnr_runs)),
            'msssim': float(np.mean(msssim_runs)),
            'ber': float(np.mean(ber_runs)),
            'compressed_bytes': r['compressed_bytes'],
            'source_bpp': r['source_bpp'],
            'channel_bpp': r['channel_bpp'],
            'n_frames': r['n_frames'],
        }
        print(f"  Average: PSNR={avg_result['psnr']:.2f} "
              f"MS-SSIM={avg_result['msssim']:.4f} BER={avg_result['ber']:.6f}")
        results.append(avg_result)

    # Cleanup
    shutil.rmtree(pooled_dir, ignore_errors=True)

    if not results:
        print("\nNo results collected.")
        return

    # Print table
    print(f"\n{'='*72}")
    print(f"{'SNR (dB)':<10} {'PSNR (dB)':<11} {'MS-SSIM':<10} "
          f"{'BER':<12} {'Src BPP':<9} {'Ch BPP':<8}")
    print(f"{'-'*72}")
    for r in results:
        print(f"{r['snr_db']:<10.1f} {r['psnr']:<11.2f} {r['msssim']:<10.4f} "
              f"{r['ber']:<12.6f} {r['source_bpp']:<9.4f} {r['channel_bpp']:<8.4f}")

    # Save CSV
    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {args.out}")


if __name__ == '__main__':
    main()