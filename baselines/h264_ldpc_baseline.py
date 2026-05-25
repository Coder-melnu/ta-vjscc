# -*- coding: utf-8 -*-
"""
H.264/H.265 + LDPC Separation Baseline for Deep-JSCC comparison.

Pipeline:
    Original frames
        → H.264/H.265 encode (FFmpeg, 2-pass target bitrate or CRF)
        → LDPC encode (Sionna 5G NR)
        → BPSK modulation
        → Channel (AWGN or Rayleigh)
        → BPSK demodulation + LLR
        → LDPC decode
        → H.264/H.265 decode
        → Measure PSNR / MS-SSIM

Bandwidth Matching:
    To fairly compare with DeepJSCC at a given CBR, the baseline must
    transmit the same number of channel symbols per pixel. The target
    bitrate for FFmpeg 2-pass encoding is computed as:

        source_bits_per_frame = (H × W × C) × CBR × 2 × ldpc_rate
        target_bitrate_kbps   = source_bits_per_frame × fps / 1000

    where the factor of 2 accounts for real/imaginary components of
    complex channel symbols, and ldpc_rate determines how much of the
    channel bandwidth is available for source bits after LDPC redundancy.

Usage:
    # Bandwidth-matched to DeepJSCC CBR=1/6
    python h264_ldpc_baseline.py --target_bpp 0.33

    # Bandwidth-matched to DeepJSCC CBR=1/12
    python h264_ldpc_baseline.py --target_bpp 0.17

    # Unconstrained CRF mode (legacy, not for paper comparison)
    python h264_ldpc_baseline.py --crf 23

Dependencies:
    conda activate sionna
    pip install sionna opencv-python Pillow scikit-image numpy pytorch-msssim
"""

import os
import subprocess
import tempfile
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

import torch
from pytorch_msssim import ms_ssim as torch_ms_ssim

import tensorflow as tf
import sionna
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


# ---------------------------------------------------------------------------
# FFmpeg Utilities
# ---------------------------------------------------------------------------

def encode_frames_to_video_crf(frames_dir: str, output_path: str,
                                codec: str = 'h264', crf: int = 23,
                                fps: int = 25) -> int:
    """
    Encode frames using CRF mode (unconstrained bitrate).
    NOT recommended for fair comparison — use 2-pass instead.
    """
    codec_lib = 'libx264' if codec == 'h264' else 'libx265'
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%04d.png'),
        '-c:v', codec_lib,
        '-crf', str(crf),
        '-g', '1',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg CRF encode failed:\n{result.stderr}")
    return os.path.getsize(output_path)


def encode_frames_to_video_2pass(frames_dir: str, output_path: str,
                                  codec: str = 'h264',
                                  target_bitrate_kbps: int = 500,
                                  fps: int = 25) -> int:
    """
    Encode frames using 2-pass target bitrate mode.
    This constrains the output to a specific bitrate, enabling
    fair bandwidth-matched comparison with DeepJSCC.

    Args:
        frames_dir          : folder containing %04d.png frames
        output_path         : output .mp4 path
        codec               : 'h264' or 'h265'
        target_bitrate_kbps : target bitrate in kbps
        fps                 : frames per second

    Returns:
        compressed file size in bytes
    """
    codec_lib = 'libx264' if codec == 'h264' else 'libx265'
    bitrate = f'{target_bitrate_kbps}k'
    input_pattern = os.path.join(frames_dir, '%04d.png')
    passlog = os.path.join(os.path.dirname(output_path), 'ffmpeg2pass')

    # Pass 1 — analyze
    cmd1 = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', codec_lib,
        '-b:v', bitrate,
        '-g', '1',
        '-pix_fmt', 'yuv420p',
        '-pass', '1',
        '-passlogfile', passlog,
        '-an', '-f', 'null', '/dev/null'
    ]
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode != 0:
        raise RuntimeError(f"FFmpeg 2-pass (pass 1) failed:\n{result1.stderr}")

    # Pass 2 — encode at target bitrate
    cmd2 = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', codec_lib,
        '-b:v', bitrate,
        '-g', '1',
        '-pix_fmt', 'yuv420p',
        '-pass', '2',
        '-passlogfile', passlog,
        '-an', output_path
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode != 0:
        raise RuntimeError(f"FFmpeg 2-pass (pass 2) failed:\n{result2.stderr}")

    # Clean up passlog files
    for f in Path(os.path.dirname(output_path)).glob('ffmpeg2pass*'):
        f.unlink()

    return os.path.getsize(output_path)


def decode_video_to_frames(video_path: str, output_dir: str,
                           image_size: int = 256):
    """Decode a compressed video back to PNG frames."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'scale={image_size}:{image_size}',
        os.path.join(output_dir, '%04d.png')
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg decode failed:\n{result.stderr}")


def frames_dir_to_png(input_dir: str, output_dir: str, image_size: int = 256,
                      max_frames: int = 50):
    """Copy/resize images into sequentially numbered PNGs for FFmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    exts = {'.png', '.jpg', '.jpeg'}
    files = sorted([f for f in Path(input_dir).iterdir() if f.suffix.lower() in exts])
    files = files[:max_frames]

    if len(files) == 0:
        raise ValueError(f"No image files found in {input_dir}")

    for i, f in enumerate(files):
        img = Image.open(f).convert('RGB').resize((image_size, image_size))
        img.save(os.path.join(output_dir, f'{i+1:04d}.png'))

    return len(files)


def compute_target_bitrate(image_size: int, cbr: float, ldpc_rate: float,
                           fps: int = 25) -> int:
    """
    Compute FFmpeg target bitrate (kbps) to match DeepJSCC's CBR.

    The total channel uses per pixel for DeepJSCC = CBR.
    For the separation baseline with BPSK:
        channel_bits_per_pixel = CBR × 2  (real + imaginary)
        source_bits_per_pixel  = channel_bits_per_pixel × ldpc_rate
        target_bitrate         = source_bpp × width × height × fps

    Args:
        image_size : frame width/height (square)
        cbr        : DeepJSCC channel bandwidth ratio (e.g., 1/6 or 1/12)
        ldpc_rate  : LDPC code rate (e.g., 2/3)
        fps        : frames per second

    Returns:
        target bitrate in kbps
    """
    source_bpp = cbr * 2 * ldpc_rate
    bits_per_frame = source_bpp * image_size * image_size
    target_kbps = int(bits_per_frame * fps / 1000)
    return target_kbps


# ---------------------------------------------------------------------------
# LDPC Channel Simulation
# ---------------------------------------------------------------------------

def bitstring_from_file(filepath: str) -> tf.Tensor:
    """Read a binary file and return as a float32 tensor of bits."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype(np.float32)
    return tf.constant(bits)


def bits_to_file(bits: tf.Tensor, filepath: str, original_size: int):
    """Write a bit tensor back to a binary file."""
    bits_np = tf.cast(tf.round(bits), tf.uint8).numpy()
    pad = (8 - len(bits_np) % 8) % 8
    bits_np = np.concatenate([bits_np, np.zeros(pad, dtype=np.uint8)])
    byte_array = np.packbits(bits_np)[:original_size]
    with open(filepath, 'wb') as f:
        f.write(byte_array.tobytes())


def simulate_ldpc_channel(input_file: str, output_file: str,
                           snr_db: float, channel_type: str = 'AWGN',
                           ldpc_rate: float = 0.5) -> float:
    """
    Simulate LDPC encoding -> channel -> LDPC decoding on a binary file.
    """
    bits = bitstring_from_file(input_file)
    total_bits = len(bits)
    original_bytes = os.path.getsize(input_file)

    k = 4096
    n = 6144

    pad_bits = (k - total_bits % k) % k
    bits_padded = tf.concat([bits, tf.zeros(pad_bits, dtype=tf.float32)], axis=0)
    num_codewords = len(bits_padded) // k
    bits_2d = tf.reshape(bits_padded, (num_codewords, k))

    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    codewords = encoder(bits_2d)

    # BPSK modulation: 0 -> +1, 1 -> -1
    bpsk = 1.0 - 2.0 * codewords

    snr_linear = 10.0 ** (snr_db / 10.0)
    noi_pwr = 1.0 / snr_linear
    noise_std = tf.sqrt(noi_pwr / 2.0)

    if channel_type == 'AWGN':
        noise = tf.random.normal(tf.shape(bpsk), stddev=noise_std)
        y = bpsk + noise
        llr = 2.0 * y / (noi_pwr + 1e-10)

    elif channel_type == 'Rayleigh':
        h_re = tf.random.normal([num_codewords, 1])
        h_im = tf.random.normal([num_codewords, 1])
        h_mag = tf.sqrt(h_re**2 + h_im**2)

        noise = tf.random.normal(tf.shape(bpsk), stddev=noise_std)
        y_rx = h_mag * bpsk + noise

        y_eq = h_mag * y_rx / (h_mag**2 + noi_pwr)
        var_eff = noi_pwr / (h_mag**2 + noi_pwr)
        llr = 2.0 * y_eq / (var_eff + 1e-10)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")

    bits_hat = decoder(-llr)
    ber = float(tf.reduce_mean(tf.cast(tf.not_equal(bits_2d, bits_hat), tf.float32)))

    bits_flat = tf.reshape(bits_hat, [-1])[:total_bits]
    bits_to_file(bits_flat, output_file, original_bytes)

    return ber


# ---------------------------------------------------------------------------
# Image Quality Metrics
# ---------------------------------------------------------------------------

def compute_ms_ssim_torch(orig_np: np.ndarray, recon_np: np.ndarray) -> float:
    """Compute MS-SSIM between two HWC uint8 images using pytorch-msssim."""
    orig_t  = torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    recon_t = torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    H = orig_t.shape[2]
    if H < 160:
        # 3-scale fallback for small images (128x128)
        weights = [0.3222, 0.3363, 0.3415]
        return float(torch_ms_ssim(orig_t, recon_t, data_range=1.0,
                                   win_size=7, weights=weights))
    else:
        return float(torch_ms_ssim(orig_t, recon_t, data_range=1.0))

def compute_psnr_msssim(original_dir: str, reconstructed_dir: str) -> dict:
    """Compute average PSNR and MS-SSIM between two folders of frames."""
    orig_files  = sorted(Path(original_dir).glob('*.png'))
    recon_files = sorted(Path(reconstructed_dir).glob('*.png'))

    n = min(len(orig_files), len(recon_files))
    if n == 0:
        raise ValueError("No matching PNG files found in directories")

    psnr_list   = []
    msssim_list = []

    for of, rf in zip(orig_files[:n], recon_files[:n]):
        orig  = np.array(Image.open(of).convert('RGB'))
        recon = np.array(Image.open(rf).convert('RGB'))

        if orig.shape != recon.shape:
            recon = np.array(Image.fromarray(recon).resize(
                (orig.shape[1], orig.shape[0])))

        psnr = ski_psnr(orig, recon, data_range=255)
        msssim = compute_ms_ssim_torch(orig, recon)

        psnr_list.append(psnr)
        msssim_list.append(msssim)

    return {
        'psnr':     float(np.mean(psnr_list)),
        'msssim':   float(np.mean(msssim_list)),
        'n_frames': n,
    }


# ---------------------------------------------------------------------------
# Full Baseline Pipeline
# ---------------------------------------------------------------------------

def run_baseline(frames_dir: str, codec: str, channel_type: str,
                 snr_db: float, ldpc_rate: float = 0.6667,
                 image_size: int = 256, max_frames: int = 50,
                 target_bpp: float = None, crf: int = 23,
                 fps: int = 25) -> dict:
    """
    Run the full H.264/H.265 + LDPC baseline for one SNR point.

    If target_bpp is set, uses 2-pass encoding matched to DeepJSCC CBR.
    Otherwise falls back to CRF mode (unconstrained).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir    = os.path.join(tmpdir, 'original')
        video_path  = os.path.join(tmpdir, 'compressed.mp4')
        recov_video = os.path.join(tmpdir, 'recovered.mp4')
        recon_dir   = os.path.join(tmpdir, 'reconstructed')

        # Step 1: Prepare original frames
        n_frames = frames_dir_to_png(frames_dir, orig_dir,
                                     image_size=image_size,
                                     max_frames=max_frames)
        print(f"  Prepared {n_frames} frames")

        # Step 2: H.264/H.265 encode
        if target_bpp is not None:
            # 2-pass: bandwidth-matched to DeepJSCC CBR
            # target_bpp is the source BPP (after LDPC overhead removed)
            target_kbps = int(target_bpp * image_size * image_size * fps / 1000)
            print(f"  2-pass encoding: target {target_kbps} kbps "
                  f"(source BPP={target_bpp:.3f})")
            compressed_bytes = encode_frames_to_video_2pass(
                orig_dir, video_path, codec=codec,
                target_bitrate_kbps=target_kbps, fps=fps)
        else:
            # CRF mode: unconstrained (not for fair comparison)
            compressed_bytes = encode_frames_to_video_crf(
                orig_dir, video_path, codec=codec, crf=crf, fps=fps)

        total_pixels = n_frames * image_size * image_size
        source_bpp = (compressed_bytes * 8) / total_pixels
        channel_bpp = source_bpp / ldpc_rate
        print(f"  Compressed: {compressed_bytes/1024:.1f} KB | "
              f"Source BPP: {source_bpp:.4f} | Channel BPP: {channel_bpp:.4f}")

        # Step 3: LDPC encode -> channel -> LDPC decode
        ber = simulate_ldpc_channel(video_path, recov_video,
                                    snr_db=snr_db, channel_type=channel_type,
                                    ldpc_rate=ldpc_rate)
        print(f"  BER after LDPC: {ber:.6f}")

        # Step 4: Decode recovered bitstream back to frames
        try:
            decode_video_to_frames(recov_video, recon_dir, image_size=image_size)
        except RuntimeError:
            print(f"  WARNING: Decode failed (high BER={ber:.4f}) — returning zeros")
            return {
                'snr_db': snr_db, 'codec': codec, 'channel': channel_type,
                'psnr': 0.0, 'msssim': 0.0, 'ber': ber,
                'compressed_bytes': compressed_bytes,
                'source_bpp': source_bpp, 'channel_bpp': channel_bpp,
                'n_frames': n_frames,
            }

        # Step 5: Compute metrics
        metrics = compute_psnr_msssim(orig_dir, recon_dir)
        print(f"  PSNR: {metrics['psnr']:.2f} dB | MS-SSIM: {metrics['msssim']:.4f}")

        return {
            'snr_db':           snr_db,
            'codec':            codec,
            'channel':          channel_type,
            'psnr':             metrics['psnr'],
            'msssim':           metrics['msssim'],
            'ber':              ber,
            'compressed_bytes': compressed_bytes,
            'source_bpp':       source_bpp,
            'channel_bpp':      channel_bpp,
            'n_frames':         metrics['n_frames'],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='H.264/H.265 + LDPC Baseline')

    parser.add_argument('--frames_dir',  default='/path/to/UCF101/frames',
                        help='Folder of input images (UCF101 frames)')
    parser.add_argument('--codec',       default='h264',
                        choices=['h264', 'h265'])
    parser.add_argument('--channel',     default='AWGN',
                        choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr_list',    nargs='+', type=float,
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                 20, 21, 22, 23, 24, 25])
    parser.add_argument('--ldpc_rate',   type=float, default=0.6667,
                        help='LDPC code rate (default: 2/3)')
    parser.add_argument('--image_size',  type=int,  default=256)
    parser.add_argument('--max_frames',  type=int,  default=50)
    parser.add_argument('--fps',         type=int,  default=25)

    # Bandwidth control — use ONE of these:
    parser.add_argument('--target_bpp',  type=float, default=None,
                        help='Target source BPP for 2-pass encoding. '
                             'Computed from DeepJSCC CBR: target_bpp = CBR × 2 × ldpc_rate. '
                             'For CBR=1/6 with R=2/3: target_bpp=0.222. '
                             'For CBR=1/12 with R=2/3: target_bpp=0.111.')
    parser.add_argument('--cbr',         type=float, default=None,
                        help='DeepJSCC CBR to match (e.g., 0.1667 for 1/6). '
                             'Automatically computes target_bpp from CBR and ldpc_rate.')
    parser.add_argument('--crf',         type=int,  default=23,
                        help='CRF quality (fallback if no target_bpp/cbr set)')

    parser.add_argument('--out',         default=None)

    args = parser.parse_args()

    # Compute target_bpp from CBR if specified
    if args.cbr is not None:
        args.target_bpp = args.cbr * 2 * args.ldpc_rate
        print(f"CBR={args.cbr:.4f} → target_bpp={args.target_bpp:.4f} "
              f"(with LDPC R={args.ldpc_rate:.4f})")

    # Auto-generate output filename
    if args.out is None:
        bw_tag = f'bpp{args.target_bpp:.3f}' if args.target_bpp else f'crf{args.crf}'
        args.out = f'./results/baseline_{args.codec}_{args.channel.lower()}_{bw_tag}.csv'

    # Determine encoding mode
    if args.target_bpp is not None:
        mode = '2-pass target bitrate'
    else:
        mode = f'CRF={args.crf} (unconstrained — NOT bandwidth-matched)'

    print(f"\n{'='*72}")
    print(f"Baseline : {args.codec.upper()} intra + LDPC (R={args.ldpc_rate:.4f})")
    print(f"Encoding : {mode}")
    if args.target_bpp:
        target_kbps = int(args.target_bpp * args.image_size * args.image_size * args.fps / 1000)
        print(f"Target   : BPP={args.target_bpp:.4f} → {target_kbps} kbps")
    print(f"Channel  : {args.channel}")
    print(f"SNR list : {args.snr_list}")
    print(f"Frames   : {args.frames_dir}")
    print(f"Output   : {args.out}")
    print(f"{'='*72}\n")

    results = []
    for snr in args.snr_list:
        print(f"\n--- SNR = {snr} dB ---")
        r = run_baseline(
            frames_dir=args.frames_dir,
            codec=args.codec,
            channel_type=args.channel,
            snr_db=snr,
            ldpc_rate=args.ldpc_rate,
            image_size=args.image_size,
            max_frames=args.max_frames,
            target_bpp=args.target_bpp,
            crf=args.crf,
            fps=args.fps,
        )
        results.append(r)

    if not results:
        print("\nNo results collected.")
        return

    # Print comparison table
    print(f"\n{'='*72}")
    print(f"{'SNR (dB)':<10} {'PSNR (dB)':<11} {'MS-SSIM':<10} "
          f"{'BER':<12} {'Src BPP':<9} {'Ch BPP':<8}")
    print(f"{'-'*72}")
    for r in results:
        print(f"{r['snr_db']:<10.1f} {r['psnr']:<11.2f} {r['msssim']:<10.4f} "
              f"{r['ber']:<12.6f} {r['source_bpp']:<9.4f} {r['channel_bpp']:<8.4f}")

    # Save to CSV
    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {args.out}")


if __name__ == '__main__':
    main()