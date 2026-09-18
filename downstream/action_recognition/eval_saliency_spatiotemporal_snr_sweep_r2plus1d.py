# -*- coding: utf-8 -*-
"""Paired 0--25 dB sweep for GoP-level spatiotemporal allocation.

This is evaluation-only.  It compares (1) the separately trained uniform
checkpoint, (2) the learned spatiotemporal allocation, and (3) that same
learned checkpoint with its allocation map replaced by ones.  The same clips
and AWGN draws are used at every SNR for all three paths.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_dataloaders
from model import ratio2filtersize
from model.ta_video_jscc_saliency_spatiotemporal_r2plus1d import (
    TAVideoJSCCSaliencySpatiotemporalR2Plus1D,
)
from utils import set_seed


PATHS = ("uniform_trained", "learned_allocation", "learned_weights_uniform")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform_ckpt", required=True)
    parser.add_argument("--learned_ckpt", required=True)
    parser.add_argument("--r2plus1d_ckpt", required=True)
    parser.add_argument("--frames_root", default="datasets/UCF101Frames")
    parser.add_argument(
        "--annotation_path",
        default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist",
    )
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--gop_size", type=int, default=5)
    parser.add_argument("--gops_per_clip", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--ratio", default="1/6")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--scorer_hidden", type=int, default=16)
    parser.add_argument("--allocation_temperature", type=float, default=1.0)
    parser.add_argument("--min_relative_power", type=float, default=0.1)
    parser.add_argument("--snr_start", type=int, default=0)
    parser.add_argument("--snr_end", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_clips", type=int, default=None,
        help="Optional smoke-test limit; omit for the complete validation split.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Keep zero for exact paired clip sampling across repeated passes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel_seed", type=int, default=1042)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="./out_saliency_spatiotemporal_snr_sweep")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def paired_rng(device, seed):
    """Reset and restore Python, NumPy, CPU and CUDA RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def checkpoint_artifact(path):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "model_state" not in artifact:
        raise ValueError(f"Expected training checkpoint with model_state: {path}")
    return artifact


def make_model(args, c, device, checkpoint_path, expected_mode, manifest_hash):
    artifact = checkpoint_artifact(checkpoint_path)
    if artifact.get("allocation_mode") != expected_mode:
        raise RuntimeError(
            f"{checkpoint_path} says allocation_mode={artifact.get('allocation_mode')!r}; "
            f"expected {expected_mode!r}"
        )
    if artifact.get("split_manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Checkpoint/evaluation split mismatch: {checkpoint_path}")
    model = TAVideoJSCCSaliencySpatiotemporalR2Plus1D(
        c=c,
        snr=13.0,
        n_frames=args.gop_size,
        hidden_dim=args.hidden_dim,
        scorer_hidden=args.scorer_hidden,
        r2plus1d_ckpt=args.r2plus1d_ckpt,
        allocation_temperature=args.allocation_temperature,
        min_relative_power=args.min_relative_power,
    )
    model.load_state_dict(artifact["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    evaluator_manifest = model.task_model.checkpoint_metadata.get("split_manifest_sha256")
    if evaluator_manifest != manifest_hash:
        raise RuntimeError("R(2+1)D/evaluation split mismatch")
    return model, {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "epoch": artifact.get("epoch"),
        "allocation_mode": artifact.get("allocation_mode"),
        "validation_metrics": artifact.get("validation_metrics"),
    }


def clip_quality(reference, reconstruction):
    batch, frames = reference.shape[:2]
    flat_reference = reference.flatten(0, 1)
    flat_reconstruction = reconstruction.flatten(0, 1)
    frame_mse = (flat_reconstruction - flat_reference).square().mean((1, 2, 3))
    frame_psnr = -10.0 * torch.log10(frame_mse.clamp_min(1e-12))
    frame_ssim = ssim(
        flat_reconstruction, flat_reference, data_range=1.0, size_average=False
    )
    frame_ms_ssim = ms_ssim(
        flat_reconstruction,
        flat_reference,
        data_range=1.0,
        size_average=False,
        win_size=7,
        weights=(0.3, 0.3, 0.4),
    )
    return (
        frame_psnr.reshape(batch, frames).mean(1),
        frame_ssim.reshape(batch, frames).mean(1),
        frame_ms_ssim.reshape(batch, frames).mean(1),
    )


@torch.no_grad()
def evaluate_channel(model, loader, device, path_name, allocation_mode, snr, seed, quiet, max_clips=None):
    model.eval(); model.task_model.eval()
    model.set_allocation_mode(allocation_mode)
    model.change_snr(snr)
    rows = []
    clip_id = 0
    weighted_frame_share = torch.zeros(model.n_frames)
    total_for_share = 0
    energy_min, energy_max = float("inf"), float("-inf")
    cv_weighted = 0.0
    with paired_rng(device, seed):
        progress = tqdm(
            loader, desc=f"{path_name}: {snr:g} dB", disable=quiet, leave=False
        )
        for clips, labels in progress:
            if max_clips is not None:
                remaining = max_clips - clip_id
                if remaining <= 0:
                    break
                clips, labels = clips[:remaining], labels[:remaining]
            clips, labels = clips.to(device), labels.to(device)
            reconstruction, _, _ = model(clips)
            reconstruction = reconstruction.clamp(0, 1)
            logits = model.task_model(reconstruction)
            top5 = logits.topk(5, dim=1).indices
            psnr, ssim_value, ms_ssim_value = clip_quality(clips, reconstruction)
            mse = (reconstruction - clips).square().mean((1, 2, 3, 4))
            audit = model.last_allocation_audit
            batch = labels.numel()
            weighted_frame_share += torch.tensor(audit["frame_energy_share"]) * batch
            cv_weighted += audit["relative_power_cv"] * batch
            total_for_share += batch
            energy_min = min(energy_min, audit["energy_ratio_min"])
            energy_max = max(energy_max, audit["energy_ratio_max"])
            for item in range(batch):
                rows.append({
                    "path": path_name,
                    "snr_db": float(snr),
                    "clip_id": clip_id,
                    "label": int(labels[item]),
                    "predicted_label": int(top5[item, 0]),
                    "top1_correct": int(top5[item, 0] == labels[item]),
                    "top5_correct": int(top5[item].eq(labels[item]).any()),
                    "mse": float(mse[item]),
                    "psnr_db": float(psnr[item]),
                    "ssim": float(ssim_value[item]),
                    "ms_ssim_3scale": float(ms_ssim_value[item]),
                })
                clip_id += 1
    return rows, {
        "relative_power_cv": cv_weighted / total_for_share,
        "frame_energy_share": (weighted_frame_share / total_for_share).tolist(),
        "energy_ratio_min": energy_min,
        "energy_ratio_max": energy_max,
        "noise_variance": model.gop_channel.last_audit["noise_variance"],
    }


@torch.no_grad()
def evaluate_bypass(model, loader, device, seed, quiet, max_clips=None):
    """Evaluate each trained codec with the GoP channel bypassed."""
    model.eval(); model.task_model.eval()
    rows = []
    clip_id = 0
    with paired_rng(device, seed):
        for clips, labels in tqdm(loader, desc="channel bypass", disable=quiet, leave=False):
            if max_clips is not None:
                remaining = max_clips - clip_id
                if remaining <= 0:
                    break
                clips, labels = clips[:remaining], labels[:remaining]
            clips, labels = clips.to(device), labels.to(device)
            reconstruction = model.decode(model.encode(clips), clips.shape).clamp(0, 1)
            logits = model.task_model(reconstruction)
            top5 = logits.topk(5, dim=1).indices
            psnr, ssim_value, ms_ssim_value = clip_quality(clips, reconstruction)
            for item in range(labels.numel()):
                rows.append({
                    "clip_id": clip_id,
                    "label": int(labels[item]),
                    "top1_correct": int(top5[item, 0] == labels[item]),
                    "top5_correct": int(top5[item].eq(labels[item]).any()),
                    "psnr_db": float(psnr[item]),
                    "ssim": float(ssim_value[item]),
                    "ms_ssim_3scale": float(ms_ssim_value[item]),
                })
                clip_id += 1
    return rows


def average(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def aggregate(rows, diagnostics, bypass):
    top1 = sum(row["top1_correct"] for row in rows)
    top5 = sum(row["top5_correct"] for row in rows)
    bypass_top1 = sum(row["top1_correct"] for row in bypass)
    count = len(rows)
    return {
        "path": rows[0]["path"],
        "snr_db": rows[0]["snr_db"],
        "clips": count,
        "top1_percent": 100.0 * top1 / count,
        "top5_percent": 100.0 * top5 / count,
        "psnr_db": average(rows, "psnr_db"),
        "ssim": average(rows, "ssim"),
        "ms_ssim_3scale": average(rows, "ms_ssim_3scale"),
        "bypass_top1_percent": 100.0 * bypass_top1 / len(bypass),
        "top1_retention_percent": 100.0 * top1 / bypass_top1 if bypass_top1 else float("nan"),
        "relative_power_cv": diagnostics["relative_power_cv"],
        "frame_energy_share": json.dumps(diagnostics["frame_energy_share"]),
        "energy_ratio_min": diagnostics["energy_ratio_min"],
        "energy_ratio_max": diagnostics["energy_ratio_max"],
        "noise_variance": diagnostics["noise_variance"],
    }


def mcnemar(left, right):
    if len(left) != len(right):
        raise RuntimeError("Paired paths have different clip counts")
    left_win = right_win = 0
    for a, b in zip(left, right):
        if a["clip_id"] != b["clip_id"] or a["label"] != b["label"]:
            raise RuntimeError("Paired paths are not clip-aligned")
        left_win += int(a["top1_correct"] and not b["top1_correct"])
        right_win += int(b["top1_correct"] and not a["top1_correct"])
    discordant = left_win + right_win
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(left_win, right_win)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return left_win, right_win, discordant, p_value


def add_holm(rows, family_key):
    """Holm-adjust p-values separately within each named comparison."""
    for family in sorted({row[family_key] for row in rows}):
        members = [row for row in rows if row[family_key] == family]
        ordered = sorted(members, key=lambda row: row["p_exact"])
        running = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * row["p_exact"])
            running = max(running, adjusted)
            row["p_holm_26_snr"] = running


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plots(output, summaries):
    colors = {
        "uniform_trained": "#4C78A8",
        "learned_allocation": "#E45756",
        "learned_weights_uniform": "#72B7B2",
    }
    labels = {
        "uniform_trained": "Uniform-trained",
        "learned_allocation": "Learned allocation",
        "learned_weights_uniform": "Learned weights, uniform allocation",
    }
    for metric, ylabel, filename in (
        ("top1_percent", "Top-1 accuracy (%)", "top1_vs_snr.png"),
        ("top5_percent", "Top-5 accuracy (%)", "top5_vs_snr.png"),
        ("psnr_db", "PSNR (dB)", "psnr_vs_snr.png"),
        ("ms_ssim_3scale", "3-scale MS-SSIM", "ms_ssim_vs_snr.png"),
        ("top1_retention_percent", "Top-1 retention vs bypass (%)", "retention_vs_snr.png"),
    ):
        plt.figure(figsize=(7.2, 4.6))
        for name in PATHS:
            subset = sorted((row for row in summaries if row["path"] == name), key=lambda x: x["snr_db"])
            plt.plot(
                [row["snr_db"] for row in subset],
                [row[metric] for row in subset],
                marker="o", markersize=3, linewidth=1.5,
                color=colors[name], label=labels[name],
            )
        plt.xlabel("Channel SNR (dB)"); plt.ylabel(ylabel)
        plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
        plt.savefig(output / filename, dpi=180); plt.close()

    indexed = {(row["path"], row["snr_db"]): row for row in summaries}
    snrs = sorted({row["snr_db"] for row in summaries})
    plt.figure(figsize=(7.2, 4.6))
    for other, label, color in (
        ("uniform_trained", "Learned allocation - uniform-trained", "#E45756"),
        ("learned_weights_uniform", "Learned allocation - same weights uniform", "#7A5195"),
    ):
        gains = [
            indexed[("learned_allocation", snr)]["top1_percent"]
            - indexed[(other, snr)]["top1_percent"] for snr in snrs
        ]
        plt.plot(snrs, gains, marker="o", markersize=3, linewidth=1.5, label=label, color=color)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Channel SNR (dB)"); plt.ylabel("Top-1 difference (percentage points)")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(output / "top1_gain_vs_snr.png", dpi=180); plt.close()


def main():
    args = build_parser().parse_args()
    if args.snr_end < args.snr_start:
        raise ValueError("snr_end must be >= snr_start")
    if args.num_workers != 0:
        raise ValueError("Use --num_workers 0 to guarantee paired repeated clip sampling")
    set_seed(args.seed)
    ratio = float(Fraction(args.ratio))
    _, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=args.frames_root,
        annotation_path=args.annotation_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    c = ratio2filtersize(torch.empty(3, args.image_size, args.image_size), ratio)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = list(range(args.snr_start, args.snr_end + 1))
    run_name = (
        f"SaliencyGoPSweep_c{c}_{args.snr_start}to{args.snr_end}dB_"
        f"seed{args.channel_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output = Path(args.out) / run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "split_manifest.sha256").write_text(manifest_hash + "\n")

    all_rows = []
    summaries = []
    rows_by_key = {}
    checkpoint_info = {}
    bypass_by_checkpoint = {}

    plans = (
        ("uniform_trained", args.uniform_ckpt, "uniform", "uniform"),
        ("learned_allocation", args.learned_ckpt, "spatiotemporal", "spatiotemporal"),
        ("learned_weights_uniform", args.learned_ckpt, "spatiotemporal", "uniform"),
    )
    active_checkpoint = None
    model = None
    for path_name, checkpoint_path, expected_mode, allocation_mode in plans:
        if checkpoint_path != active_checkpoint:
            if model is not None:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            model, info = make_model(
                args, c, device, checkpoint_path, expected_mode, manifest_hash
            )
            checkpoint_info[expected_mode] = info
            bypass_by_checkpoint[expected_mode] = evaluate_bypass(
                model, val_loader, device, args.seed, args.disable_tqdm, args.max_clips
            )
            active_checkpoint = checkpoint_path
        bypass = bypass_by_checkpoint[expected_mode]
        for snr in snrs:
            rows, diagnostics = evaluate_channel(
                model, val_loader, device, path_name, allocation_mode,
                snr, args.channel_seed, args.disable_tqdm, args.max_clips,
            )
            rows_by_key[(path_name, snr)] = rows
            all_rows.extend(rows)
            summaries.append(aggregate(rows, diagnostics, bypass))
            print(
                f"{path_name:25s} SNR={snr:2d} dB | "
                f"Top1={summaries[-1]['top1_percent']:.2f}% | "
                f"PSNR={summaries[-1]['psnr_db']:.2f} dB"
            )

    comparisons = (
        ("learned_allocation", "uniform_trained"),
        ("learned_allocation", "learned_weights_uniform"),
        ("learned_weights_uniform", "uniform_trained"),
    )
    tests = []
    summary_index = {(row["path"], row["snr_db"]): row for row in summaries}
    for left, right in comparisons:
        comparison = f"{left}_vs_{right}"
        for snr in snrs:
            left_win, right_win, discordant, p_value = mcnemar(
                rows_by_key[(left, snr)], rows_by_key[(right, snr)]
            )
            tests.append({
                "comparison": comparison,
                "snr_db": snr,
                "top1_difference_pp": (
                    summary_index[(left, snr)]["top1_percent"]
                    - summary_index[(right, snr)]["top1_percent"]
                ),
                "left_correct_right_wrong": left_win,
                "left_wrong_right_correct": right_win,
                "discordant_pairs": discordant,
                "p_exact": p_value,
            })
    add_holm(tests, "comparison")

    auc = []
    for path_name in PATHS:
        subset = sorted((row for row in summaries if row["path"] == path_name), key=lambda x: x["snr_db"])
        x = np.asarray([row["snr_db"] for row in subset], dtype=float)
        for metric in ("top1_percent", "top5_percent", "psnr_db", "ms_ssim_3scale"):
            y = np.asarray([row[metric] for row in subset], dtype=float)
            auc.append({
                "path": path_name,
                "metric": metric,
                "auc_0_25": float(np.trapz(y, x)),
                "mean_over_snr_range": float(np.trapz(y, x) / (x[-1] - x[0])) if len(x) > 1 else float(y[0]),
            })

    write_csv(output / "summary_by_snr.csv", summaries)
    write_csv(output / "per_clip_predictions.csv", all_rows)
    write_csv(output / "pairwise_mcnemar.csv", tests)
    write_csv(output / "auc_summary.csv", auc)
    make_plots(output, summaries)
    config = {
        **vars(args),
        "ratio_float": ratio,
        "c": c,
        "snr_values": snrs,
        "paths": list(PATHS),
        "actual_device": str(device),
        "evaluation_split_manifest_sha256": manifest_hash,
        "checkpoint_info": checkpoint_info,
        "paired_noise": True,
        "parameter_updates": 0,
        "checkpoint_selected_at_training_snr_db": 13,
        "official_test_used": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }
    (output / "evaluation_config.json").write_text(json.dumps(config, indent=2, default=str))
    result = {
        "status": "completed",
        "output": str(output),
        "snr_points": len(snrs),
        "clips_per_path_per_snr": len(next(iter(rows_by_key.values()))),
        "parameter_updates": 0,
        "official_test_used": False,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
