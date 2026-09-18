# -*- coding: utf-8 -*-
"""Matched validation/test evaluator for continuous spatial UEP.

Evaluates learned, random, all-ones, and bypass modes with identical clips,
GoPs, channel noise, total energy, transmitted symbol count, CBR, and frozen
TSN.  Emits per-clip metrics, power diagnostics, Wilson intervals, and exact
paired McNemar tests.  No optimizer or parameter update is permitted.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm

from downstream.action_recognition.eval_videojscc_tsn_export import (
    build_loader,
    deterministic_channel,
)
from model.ta_video_jscc_uep_continuous import TAVideoJSCCContinuousUEP


MODES = ("learned", "random", "all_ones", "bypass")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uep_ckpt", required=True)
    p.add_argument("--tsn_ckpt", required=True)
    p.add_argument("--frames_root", default="datasets/UCF101Frames")
    p.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--gop_size", type=int, default=5)
    p.add_argument("--gops_per_clip", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--test_gop_seed", type=int, default=44)
    p.add_argument("--channel_seed", type=int, default=1042)
    p.add_argument("--channel", choices=("AWGN", "Rayleigh"), default="AWGN")
    p.add_argument("--snr", type=float, default=13.0)
    p.add_argument("--c", type=int, default=8)
    p.add_argument("--ratio", type=float, default=1.0 / 6.0)
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=1729)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="./out_continuous_uep_matched")
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inferred_config_dir(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.parent.parent.name != "checkpoints":
        raise ValueError("Expected <root>/checkpoints/<run_tag>/<checkpoint>.pt")
    return checkpoint.parent.parent.parent / "configs" / checkpoint.parent.name


def load_model(args, device, evaluation_manifest_hash):
    artifact = torch.load(args.uep_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or "model_state" not in artifact:
        raise KeyError("Expected continuous-UEP checkpoint with model_state")
    config_dir = inferred_config_dir(args.uep_ckpt)
    manifest_file = config_dir / "split_manifest.sha256"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing training manifest: {manifest_file}")
    training_manifest_hash = manifest_file.read_text().strip()
    embedded_manifest = artifact.get("split_manifest_sha256")
    if embedded_manifest != training_manifest_hash:
        raise RuntimeError("Checkpoint/config manifest hashes disagree")
    if args.split == "validation" and training_manifest_hash != evaluation_manifest_hash:
        raise RuntimeError("UEP training/evaluation manifest mismatch")

    if artifact.get("method") != "continuous_bounded_spatial_uep":
        raise RuntimeError(
            f"Wrong checkpoint method: {artifact.get('method')}"
        )
    checkpoint_alpha = artifact.get("alpha")
    if checkpoint_alpha is None:
        raise KeyError("Continuous checkpoint does not contain alpha")
    if abs(float(checkpoint_alpha) - args.alpha) > 1e-12:
        raise RuntimeError(
            f"Checkpoint alpha={checkpoint_alpha} "
            f"but evaluator alpha={args.alpha}"
        )

    model = TAVideoJSCCContinuousUEP(
        c=args.c, channel_type=args.channel, snr=args.snr,
        n_frames=args.gop_size, hidden_dim=args.hidden_dim,
        tsn_head_ckpt=args.tsn_ckpt, lambda_task=0.001,
        lambda_recon=1.0, alpha=args.alpha,
        random_seed=args.random_seed,
    )
    model.load_state_dict(artifact["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tsn_hash = model.tsn.checkpoint_metadata.get("split_manifest_sha256")
    if args.split == "validation" and tsn_hash != evaluation_manifest_hash:
        raise RuntimeError("TSN/evaluation manifest mismatch")
    return model, {
        "checkpoint_epoch": artifact.get("epoch"),
        "checkpoint_validation_metrics": artifact.get("validation_metrics"),
        "checkpoint_sha256": sha256_file(args.uep_ckpt),
        "training_config_dir": str(config_dir),
        "training_manifest_sha256": training_manifest_hash,
        "tsn_manifest_sha256": tsn_hash,
    }


def normalised_entropy(values):
    probability = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / math.log(values.shape[1])


def top_share(values, fraction=0.10):
    probability = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    count = max(1, math.ceil(values.shape[1] * fraction))
    return probability.topk(count, dim=1).values.sum(dim=1)


@torch.no_grad()
def evaluate_mode(model, loader, device, mode, channel_seed):
    model.eval(); model.set_allocation_mode(mode); model.reset_random_sequence()
    rows, energy_ratios = [], []
    clip_id = 0
    with deterministic_channel(channel_seed, device):
        for gops, labels in tqdm(loader, desc=f"evaluation:{mode}"):
            gops, labels = gops.to(device), labels.to(device)
            reconstructed_raw, relative_power, _ = model(gops)
            reconstructed = reconstructed_raw.clamp(0, 1)
            logits_recon = model.tsn(reconstructed)
            logits_clean = model.tsn(gops)
            top5_recon = logits_recon.topk(5, dim=1).indices
            top5_clean = logits_clean.topk(5, dim=1).indices
            batch, frames = gops.shape[:2]
            flat_gt, flat_pred = gops.flatten(0, 1), reconstructed.flatten(0, 1)
            frame_mse = (flat_pred - flat_gt).square().mean(dim=(1, 2, 3))
            frame_psnr = -10 * torch.log10(frame_mse.clamp_min(1e-12))
            frame_ssim = ssim(flat_pred, flat_gt, data_range=1.0, size_average=False)
            frame_ms = ms_ssim(
                flat_pred, flat_gt, data_range=1.0, size_average=False,
                win_size=7, weights=(0.3, 0.3, 0.4),
            )
            raw_mse = (reconstructed_raw - gops).square().mean(dim=(1, 2, 3, 4))

            power_values = relative_power.reshape(batch, frames, -1).reshape(batch, -1)
            relative_mean = power_values.mean(dim=1)
            relative_std = power_values.std(dim=1, unbiased=False)
            relative_cv = relative_std / relative_mean.clamp_min(1e-12)
            relative_entropy = normalised_entropy(power_values)
            relative_top10 = top_share(power_values)

            actual = model.last_spatial_power.reshape(batch, frames, -1).reshape(batch, -1)
            actual = actual / actual.sum(dim=1, keepdim=True).clamp_min(1e-12)
            actual_mean = actual.mean(dim=1)
            actual_std = actual.std(dim=1, unbiased=False)
            actual_cv = actual_std / actual_mean.clamp_min(1e-12)
            actual_entropy = normalised_entropy(actual)
            actual_top10 = top_share(actual)
            energy_ratios.append(model.last_power_audit["energy_ratio_mean"])

            for item in range(batch):
                start, end = item * frames, (item + 1) * frames
                rows.append({
                    "clip_id": clip_id, "label": int(labels[item]),
                    "raw_reconstruction_mse": float(raw_mse[item]),
                    "psnr_db": float(frame_psnr[start:end].mean()),
                    "ssim": float(frame_ssim[start:end].mean()),
                    "ms_ssim_3scale": float(frame_ms[start:end].mean()),
                    "reconstructed_top1_correct": int(top5_recon[item, 0] == labels[item]),
                    "reconstructed_top5_correct": int(top5_recon[item].eq(labels[item]).any()),
                    "clean_top1_correct": int(top5_clean[item, 0] == labels[item]),
                    "clean_top5_correct": int(top5_clean[item].eq(labels[item]).any()),
                    "reconstructed_predicted_label": int(top5_recon[item, 0]),
                    "clean_predicted_label": int(top5_clean[item, 0]),
                    "relative_power_mean": float(relative_mean[item]),
                    "relative_power_std": float(relative_std[item]),
                    "relative_power_cv": float(relative_cv[item]),
                    "relative_power_entropy_normalised": float(relative_entropy[item]),
                    "relative_power_top10_share": float(relative_top10[item]),
                    "actual_power_cv": float(actual_cv[item]),
                    "actual_power_entropy_normalised": float(actual_entropy[item]),
                    "actual_power_top10_share": float(actual_top10[item]),
                })
                clip_id += 1
    return rows, energy_ratios


def average(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def wilson(successes, total, z=1.959963984540054):
    p = successes / total; denominator = 1 + z*z/total
    centre = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return centre-radius, centre+radius


def aggregate(rows, energy_ratios, ratio):
    count = len(rows)
    top1 = sum(row["reconstructed_top1_correct"] for row in rows)
    top5 = sum(row["reconstructed_top5_correct"] for row in rows)
    clean1 = sum(row["clean_top1_correct"] for row in rows)
    clean5 = sum(row["clean_top5_correct"] for row in rows)
    low, high = wilson(top1, count)
    return {
        "clips": count,
        "psnr_db": average(rows, "psnr_db"),
        "ssim": average(rows, "ssim"),
        "ms_ssim_3scale": average(rows, "ms_ssim_3scale"),
        "top1_correct": top1,
        "top1_percent": 100*top1/count,
        "top1_wilson_95_percent": [100*low, 100*high],
        "top5_percent": 100*top5/count,
        "clean_top1_percent": 100*clean1/count,
        "clean_top5_percent": 100*clean5/count,
        "accuracy_retention_percent": 100*top1/clean1,
        "actual_cbr": ratio,
        "all_symbols_transmitted": True,
        "relative_power_mean": average(rows, "relative_power_mean"),
        "relative_power_std": average(rows, "relative_power_std"),
        "relative_power_cv": average(rows, "relative_power_cv"),
        "relative_power_entropy_normalised": average(rows, "relative_power_entropy_normalised"),
        "relative_power_top10_share": average(rows, "relative_power_top10_share"),
        "actual_power_cv": average(rows, "actual_power_cv"),
        "actual_power_entropy_normalised": average(rows, "actual_power_entropy_normalised"),
        "actual_power_top10_share": average(rows, "actual_power_top10_share"),
        "energy_ratio_mean": sum(energy_ratios)/len(energy_ratios),
        "energy_ratio_min": min(energy_ratios), "energy_ratio_max": max(energy_ratios),
    }


def mcnemar(left, right):
    left_wins = right_wins = 0
    for a, b in zip(left, right):
        if a["clip_id"] != b["clip_id"] or a["label"] != b["label"]:
            raise RuntimeError("Mode rows are not paired")
        x, y = a["reconstructed_top1_correct"], b["reconstructed_top1_correct"]
        left_wins += int(x == 1 and y == 0)
        right_wins += int(x == 0 and y == 1)
    discordant = left_wins + right_wins
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(left_wins, right_wins)+1)) / 2**discordant
        p_value = min(1.0, 2*tail)
    return {
        "left_correct_right_wrong": left_wins,
        "left_wrong_right_correct": right_wins,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def main():
    args = parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader, manifest, manifest_hash = build_loader(args)
    model, checkpoint_metadata = load_model(args, device, manifest_hash)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mode_rows, summaries = {}, {}
    for mode in MODES:
        rows, energy = evaluate_mode(model, loader, device, mode, args.channel_seed)
        mode_rows[mode] = rows
        summaries[mode] = aggregate(rows, energy, args.ratio)
    changed = [k for k,v in model.state_dict().items() if not torch.equal(before[k], v.detach().cpu())]
    if changed:
        raise RuntimeError(f"Model changed during evaluation: {changed[:5]}")

    pairwise = {}
    for left, right in itertools.combinations(MODES, 2):
        pairwise[f"{left}_vs_{right}"] = {
            "left": left, "right": right, **mcnemar(mode_rows[left], mode_rows[right])
        }
    uniform_audit = {
        "max_abs_psnr_difference_db": max(
            abs(a["psnr_db"]-b["psnr_db"])
            for a,b in zip(mode_rows["all_ones"], mode_rows["bypass"])
        ),
        "top1_disagreements": sum(
            a["reconstructed_top1_correct"] != b["reconstructed_top1_correct"]
            for a,b in zip(mode_rows["all_ones"], mode_rows["bypass"])
        ),
        "predicted_label_disagreements": sum(
            a["reconstructed_predicted_label"] != b["reconstructed_predicted_label"]
            for a,b in zip(mode_rows["all_ones"], mode_rows["bypass"])
        ),
    }

    name = (
        f"ContinuousUEP_{args.split}_{args.channel}"
        f"_c{args.c}_snr{args.snr:g}"
        f"_alpha{args.alpha:.2f}_seed{args.channel_seed}"
    )
    output = Path(args.out) / name
    output.mkdir(parents=True, exist_ok=False)
    for mode, rows in mode_rows.items():
        with (output/f"per_clip_{mode}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/"split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output/"split_manifest.sha256").write_text(manifest_hash+"\n")
    (output/"mode_summaries.json").write_text(json.dumps(summaries, indent=2))
    (output/"pairwise_mcnemar.json").write_text(json.dumps(pairwise, indent=2))
    config = {
        **vars(args), "actual_device": str(device), "modes": list(MODES),
        "evaluation_manifest_sha256": manifest_hash,
        "checkpoint_metadata": checkpoint_metadata,
        "parameter_updates": 0, "all_symbols_transmitted": True,
        "temporal_power_transfer_supported": False,
        "allocation_scope": "spatial_within_each_encoded_frame",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "pytorch": torch.__version__,
    }
    (output/"evaluation_config.json").write_text(json.dumps(config, indent=2, default=str))
    result = {
        "summaries": summaries, "pairwise_mcnemar": pairwise,
        "all_ones_vs_bypass_audit": uniform_audit,
        "parameter_updates": 0, "official_test_used": args.split == "test",
    }
    (output/"summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); print(f"Saved: {output}")


if __name__ == "__main__":
    main()
