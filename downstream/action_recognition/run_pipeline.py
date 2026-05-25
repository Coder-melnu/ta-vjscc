#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Master runner — downstream action recognition evaluation.
Evaluates both DeepJSCC and H.264+LDPC across SNR values.

Usage:
    # Smoke test (C3D only, 20 videos, 2 SNR points):
    python downstream/action_recognition/run_pipeline.py \
        --snr_list 13 7 \
        --max_videos 20 \
        --recognizers c3d \
        --device cuda:0

    # Full evaluation (both methods, both recognizers):
    python downstream/action_recognition/run_pipeline.py \
        --snr_list 19 13 7 4 1 \
        --max_videos 200 \
        --recognizers c3d slowfast \
        --slowfast_head downstream/action_recognition/weights/slowfast_ucf101_head.pth \
        --device cuda:0

    # Specific checkpoints:
    python downstream/action_recognition/run_pipeline.py \
        --ckpt_dirs out_video/checkpoints/VideoJSCC_AWGN_c8_snr13* \
        --snr_list 13 7 4 1 \
        --max_videos 100 \
        --recognizers c3d
"""

import os
import sys
import glob
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from downstream.action_recognition.pipeline.encode_decode import (
    build_pipeline_from_dir, find_checkpoint_file, parse_checkpoint_name,
)
from baselines.h264_baseline_wrapper import H264LDPCPipeline
from downstream.action_recognition.evaluate.action_eval import (
    evaluate_action_recognition,
    find_psnr_accuracy_gap_examples,
    aggregate_results,
)
from downstream.action_recognition.results.export_table import save_all


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description='Downstream Action Recognition Evaluation — DeepJSCC + H.264+LDPC'
    )

    # Checkpoints
    parser.add_argument('--ckpt_dirs', nargs='+', default=None,
        help='Checkpoint dirs. Default: auto-discover all in out_video/checkpoints/')

    # Dataset
    parser.add_argument('--frames_root',
        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--split_file',   default='testlist01.txt')
    parser.add_argument('--image_size',   type=int, default=128)
    parser.add_argument('--n_frames',     type=int, default=5)
    parser.add_argument('--max_videos',   type=int, default=100,
        help='Max videos to evaluate. Use 20 for smoke test, 200 for full eval.')

    # Channel sweep
    parser.add_argument('--snr_list', nargs='+', type=float,
        default=[19.0, 13.0, 7.0, 4.0, 1.0])
    parser.add_argument('--channel',  default='AWGN', choices=['AWGN', 'Rayleigh'])

    # Recognizers
    parser.add_argument('--recognizers', nargs='+',
        choices=['c3d', 'slowfast', 'tsn'], default=['tsn'])
    parser.add_argument('--slowfast_head', default=None)
    parser.add_argument('--tsn_head',      default=None)
    parser.add_argument('--c3d_config',    default=None)
    parser.add_argument('--c3d_ckpt',      default=None)

    # H.264+LDPC baseline
    parser.add_argument('--skip_h264', action='store_true',
        help='Skip H.264+LDPC baseline (useful for quick smoke tests)')
    parser.add_argument('--ldpc_rate', type=float, default=0.6667,
        help='LDPC code rate (default: 2/3)')
    parser.add_argument('--h264_codec', default='h264', choices=['h264', 'h265'])

    # Output
    parser.add_argument('--output_dir',
        default='downstream/action_recognition/results')
    parser.add_argument('--tag', default='')

    # Device
    parser.add_argument('--device', default='cuda:0')

    # Gap analysis
    parser.add_argument('--psnr_tolerance', type=float, default=1.5)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        print(f"[runner] GPU: {torch.cuda.get_device_name(device)}  "
              f"({torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB)")
    else:
        print("[runner] CPU only")

    # ------------------------------------------------------------------
    # 1. Discover DeepJSCC checkpoints
    # ------------------------------------------------------------------
    ckpt_dirs = []
    if args.ckpt_dirs:
        for pattern in args.ckpt_dirs:
            matches = glob.glob(pattern)
            ckpt_dirs.extend(matches if matches else
                            ([pattern] if os.path.isdir(pattern) else []))
    else:
        ckpt_dirs = sorted(glob.glob(
            os.path.join(PROJECT_ROOT, 'out_video/checkpoints/VideoJSCC_*')
        ))

    ckpt_dirs = [d for d in ckpt_dirs
                 if os.path.isdir(d) and find_checkpoint_file(d)]

    if not ckpt_dirs:
        print("[runner] ERROR: No valid checkpoints found.")
        return

    print(f"\n[runner] {len(ckpt_dirs)} DeepJSCC checkpoint(s):")
    for d in ckpt_dirs:
        print(f"  {os.path.basename(d)}")

    # ------------------------------------------------------------------
    # 2. Load recognizers
    # ------------------------------------------------------------------
    c3d_rec = slowfast_rec = tsn_rec = None

    if 'c3d' in args.recognizers:
        try:
            from downstream.action_recognition.models.c3d_recognizer import C3DRecognizer
            c3d_rec = C3DRecognizer(
                config_path=args.c3d_config,
                ckpt_path=args.c3d_ckpt,
                device=device,
                ucf_classind=os.path.join(args.annotation_path, 'classInd.txt'),
            )
            print("[runner] C3D loaded OK")
        except Exception as e:
            print(f"[runner] C3D failed: {e}")

    if 'slowfast' in args.recognizers:
        try:
            from downstream.action_recognition.models.slowfast_recognizer import SlowFastRecognizer
            slowfast_rec = SlowFastRecognizer(
                head_ckpt=args.slowfast_head,
                device=device,
                ucf_classind=os.path.join(args.annotation_path, 'classInd.txt'),
            )
            print("[runner] SlowFast loaded OK")
        except Exception as e:
            print(f"[runner] SlowFast failed: {e}")

    if 'tsn' in args.recognizers:
        try:
            from downstream.action_recognition.models.tsn_recognizer import TSNRecognizer
            tsn_rec = TSNRecognizer(
                head_ckpt=args.tsn_head,
                device=device,
                ucf_classind=os.path.join(args.annotation_path, 'classInd.txt'),
                image_size=args.image_size,
            )
            print("[runner] TSN loaded OK")
        except Exception as e:
            print(f"[runner] TSN failed: {e}")

    if c3d_rec is None and slowfast_rec is None and tsn_rec is None:
        print("[runner] ERROR: No recognizer available.")
        return

    recognizer_keys = (
        (['c3d'] if c3d_rec      else []) +
        (['sf']  if slowfast_rec else []) +
        (['tsn'] if tsn_rec      else [])
    )

    # ------------------------------------------------------------------
    # 3. Build DeepJSCC pipelines
    # ------------------------------------------------------------------
    deepjscc_pipelines = []
    cbr_set = set()

    for ckpt_dir in ckpt_dirs:
        try:
            pipeline = build_pipeline_from_dir(
                ckpt_dir=ckpt_dir,
                snr=args.snr_list[0],
                device=device,
            )
            deepjscc_pipelines.append(pipeline)
            info = parse_checkpoint_name(ckpt_dir)
            cbr_set.add(info.get('ratio', 1/6))
        except Exception as e:
            print(f"[runner] Failed to load {os.path.basename(ckpt_dir)}: {e}")

    # ------------------------------------------------------------------
    # 4. Build H.264+LDPC pipelines (one per unique CBR)
    # ------------------------------------------------------------------
    h264_pipelines = []

    if not args.skip_h264:
        print(f"\n[runner] Building H.264+LDPC pipelines for CBRs: {sorted(cbr_set)}")
        for cbr in sorted(cbr_set):
            try:
                h264_pipeline = H264LDPCPipeline(
                    cbr=cbr,
                    channel_type=args.channel,
                    snr=args.snr_list[0],
                    ldpc_rate=args.ldpc_rate,
                    codec=args.h264_codec,
                )
                h264_pipelines.append(h264_pipeline)
                print(f"[runner] H264+LDPC CBR={cbr:.4f} ready")
            except Exception as e:
                print(f"[runner] H264+LDPC CBR={cbr:.4f} failed: {e}")
    else:
        print("[runner] Skipping H.264+LDPC baseline (--skip_h264)")

    if not deepjscc_pipelines:
        print("[runner] ERROR: No DeepJSCC pipelines loaded.")
        return

    # ------------------------------------------------------------------
    # 5. Run evaluation
    # ------------------------------------------------------------------
    all_results = evaluate_action_recognition(
        deepjscc_pipelines=deepjscc_pipelines,
        h264_pipelines=h264_pipelines,
        frames_root=args.frames_root,
        annotation_path=args.annotation_path,
        split_file=args.split_file,
        snr_list=args.snr_list,
        c3d_recognizer=c3d_rec,
        slowfast_recognizer=slowfast_rec,
        tsn_recognizer=tsn_rec,
        n_frames=args.n_frames,
        image_size=args.image_size,
        max_videos=args.max_videos,
    )

    if not all_results:
        print("[runner] No results.")
        return

    # ------------------------------------------------------------------
    # 6. Aggregate + gap examples + save
    # ------------------------------------------------------------------
    summary      = aggregate_results(all_results, recognizer_keys=recognizer_keys)
    gap_examples = []
    for key in recognizer_keys:
        gap_examples.extend(
            find_psnr_accuracy_gap_examples(
                all_results, recognizer_key=key,
                psnr_tolerance=args.psnr_tolerance, top_k=10,
            )
        )

    save_all(
        results=all_results,
        summary=summary,
        gap_examples=gap_examples,
        output_dir=args.output_dir,
        tag=args.tag or args.channel,
        recognizer_keys=recognizer_keys,
    )
    print(f"\n[runner] Done. Results → {args.output_dir}/")


if __name__ == '__main__':
    main()