# -*- coding: utf-8 -*-
"""
H.264/H.265 + LDPC Separation Baseline for Deep-JSCC comparison.

Pipeline:
    Original frames
        → H.264/H.265 encode (FFmpeg)       [source coding]
        → LDPC encode (Sionna)              [channel coding]
        → Channel (AWGN or Rayleigh)        [transmission]
        → LDPC decode (Sionna)              [channel decoding]
        → H.264/H.265 decode (FFmpeg)       [source decoding]
        → Measure PSNR / MS-SSIM

Usage:
    # Use defaults (edit defaults in main() below)
    python h264_ldpc_baseline.py

    # Override specific arguments
    python h264_ldpc_baseline.py --codec h265 --channel Rayleigh
    python h264_ldpc_baseline.py --snr_list 0 10 20 --max_frames 10

Dependencies:
    conda activate sionna
    pip install sionna opencv-python Pillow scikit-image numpy
    brew install ffmpeg  (macOS)
"""

import os
import subprocess
import tempfile
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from skimage.metrics import structural_similarity as ski_ssim

import tensorflow as tf
import sionna
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.channel import AWGN


# ---------------------------------------------------------------------------
# FFmpeg Utilities
# ---------------------------------------------------------------------------

def encode_frames_to_video(frames_dir: str, output_path: str,
                            codec: str = 'h264', crf: int = 23,
                            fps: int = 25) -> int:
    """
    Encode a folder of PNG frames to a compressed video.

    Args:
        frames_dir  : folder containing %04d.png frames
        output_path : output .mp4 path
        codec       : 'h264' or 'h265'
        crf         : quality factor (lower = better quality, larger file)
                      typical range: 18 (high quality) - 28 (low quality)
        fps         : frames per second

    Returns:
        compressed file size in bytes
    """
    codec_lib = 'libx264' if codec == 'h264' else 'libx265'
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%04d.png'),
        '-c:v', codec_lib,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg encode failed:\n{result.stderr}")

    return os.path.getsize(output_path)


def decode_video_to_frames(video_path: str, output_dir: str,
                           image_size: int = 256):
    """
    Decode a compressed video back to PNG frames.

    Args:
        video_path  : input .mp4 path
        output_dir  : folder to write decoded %04d.png frames
        image_size  : resize frames to (image_size x image_size)
    """
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
    """
    Copy/resize images from input_dir into output_dir as sequentially
    numbered PNGs (%04d.png) for FFmpeg input.

    Args:
        input_dir  : folder of images (any format)
        output_dir : output folder for numbered PNGs
        image_size : resize to square
        max_frames : max number of frames to process

    Returns:
        number of frames prepared
    """
    os.makedirs(output_dir, exist_ok=True)
    exts = {'.png', '.jpg', '.jpeg'}
    files = sorted([f for f in Path(input_dir).iterdir() if f.suffix.lower() in exts])
    files = files[:max_frames]

    for i, f in enumerate(files):
        img = Image.open(f).convert('RGB').resize((image_size, image_size))
        img.save(os.path.join(output_dir, f'{i+1:04d}.png'))

    return len(files)


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
    """
    Write a bit tensor back to a binary file.

    Args:
        bits          : float32 tensor of bits (0.0 or 1.0)
        filepath      : output file path
        original_size : original file size in bytes (to trim padding)
    """
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

    Args:
        input_file   : path to compressed video file (bitstream)
        output_file  : path to write recovered bitstream
        snr_db       : channel SNR in dB
        channel_type : 'AWGN' or 'Rayleigh'
        ldpc_rate    : code rate (k/n). 0.5 = rate-1/2

    Returns:
        bit error rate (BER) after decoding
    """
    bits = bitstring_from_file(input_file)
    total_bits = len(bits)
    original_bytes = os.path.getsize(input_file)

    k = 1000                    # info bits per codeword
    n = int(k / ldpc_rate)      # codeword length (n=2000 for rate-1/2)

    # Pad bits to multiple of k
    pad_bits = (k - total_bits % k) % k
    bits_padded = tf.concat([bits, tf.zeros(pad_bits, dtype=tf.float32)], axis=0)
    num_codewords = len(bits_padded) // k
    bits_2d = tf.reshape(bits_padded, (num_codewords, k))

    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, hard_out=True)
    channel = AWGN()

    codewords = encoder(bits_2d)    # (num_codewords, n)
    no = tf.cast(10 ** (-snr_db / 10.0), dtype=tf.float32)

    if channel_type == 'AWGN':
        y = channel([codewords, no])
    elif channel_type == 'Rayleigh':
        # Flat Rayleigh fading with perfect CSI equalization
        h = tf.sqrt(tf.random.normal([num_codewords, 1])**2 +
                    tf.random.normal([num_codewords, 1])**2) / tf.sqrt(2.0)
        y = channel([h * codewords, no])
        y = y / (h + 1e-8)
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")

    bits_hat = decoder([y, no])     # (num_codewords, k)
    ber = float(tf.reduce_mean(tf.cast(tf.not_equal(bits_2d, bits_hat), tf.float32)))

    bits_flat = tf.reshape(bits_hat, [-1])[:total_bits]
    bits_to_file(bits_flat, output_file, original_bytes)

    return ber


# ---------------------------------------------------------------------------
# Image Quality Metrics
# ---------------------------------------------------------------------------

def compute_psnr_msssim(original_dir: str, reconstructed_dir: str) -> dict:
    """
    Compute average PSNR and MS-SSIM between two folders of frames.

    Returns:
        dict with 'psnr', 'msssim', 'n_frames' keys
    """
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
        ssim = ski_ssim(orig, recon, data_range=255, channel_axis=2, win_size=7)

        psnr_list.append(psnr)
        msssim_list.append(ssim)

    return {
        'psnr':     float(np.mean(psnr_list)),
        'msssim':   float(np.mean(msssim_list)),
        'n_frames': n,
    }


# ---------------------------------------------------------------------------
# Full Baseline Pipeline
# ---------------------------------------------------------------------------

def run_baseline(frames_dir: str, codec: str, channel_type: str,
                 snr_db: float, crf: int = 23,
                 image_size: int = 256, max_frames: int = 50) -> dict:
    """
    Run the full H.264/H.265 + LDPC baseline for one SNR point.

    Returns:
        dict with psnr, msssim, ber, compressed_bytes, bpp
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
        compressed_bytes = encode_frames_to_video(orig_dir, video_path,
                                                  codec=codec, crf=crf)
        total_pixels = n_frames * image_size * image_size
        bpp = (compressed_bytes * 8) / total_pixels
        print(f"  Compressed: {compressed_bytes/1024:.1f} KB | BPP: {bpp:.4f}")

        # Step 3: LDPC encode -> channel -> LDPC decode
        ber = simulate_ldpc_channel(video_path, recov_video,
                                    snr_db=snr_db, channel_type=channel_type)
        print(f"  BER after LDPC: {ber:.6f}")

        # Step 4: Decode recovered bitstream back to frames
        try:
            decode_video_to_frames(recov_video, recon_dir, image_size=image_size)
        except RuntimeError:
            print(f"  WARNING: Decode failed (high BER={ber:.4f}) — returning zeros")
            return {
                'snr_db': snr_db, 'codec': codec, 'channel': channel_type,
                'psnr': 0.0, 'msssim': 0.0, 'ber': ber,
                'compressed_bytes': compressed_bytes, 'bpp': bpp,
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
            'bpp':              bpp,
            'n_frames':         metrics['n_frames'],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='H.264/H.265 + LDPC Baseline')

    # ---- Edit these defaults to avoid typing arguments every time ----
    parser.add_argument('--frames_dir',  default='/path/to/UCF101/frames',
                        help='Folder of input images (UCF101 frames)')
    parser.add_argument('--codec',       default='h264',
                        choices=['h264', 'h265'])
    parser.add_argument('--channel',     default='AWGN',
                        choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr_list',    nargs='+', type=float,
                        default=[0, 5, 10, 15, 20])
    parser.add_argument('--crf',         type=int,  default=23,
                        help='FFmpeg quality (18=high quality, 28=low quality)')
    parser.add_argument('--image_size',  type=int,  default=256)
    parser.add_argument('--max_frames',  type=int,  default=50)
    parser.add_argument('--out',         default='./results/baseline_h264_awgn.csv')
    # ------------------------------------------------------------------

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Baseline : {args.codec.upper()} + LDPC")
    print(f"Channel  : {args.channel}")
    print(f"SNR list : {args.snr_list}")
    print(f"Frames   : {args.frames_dir}")
    print(f"{'='*60}\n")

    results = []
    for snr in args.snr_list:
        print(f"\n--- SNR = {snr} dB ---")
        r = run_baseline(
            frames_dir=args.frames_dir,
            codec=args.codec,
            channel_type=args.channel,
            snr_db=snr,
            crf=args.crf,
            image_size=args.image_size,
            max_frames=args.max_frames,
        )
        results.append(r)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"{'SNR (dB)':<12} {'PSNR (dB)':<12} {'MS-SSIM':<12} {'BER':<12} {'BPP':<8}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['snr_db']:<12.1f} {r['psnr']:<12.2f} {r['msssim']:<12.4f} "
              f"{r['ber']:<12.6f} {r['bpp']:<8.4f}")

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