# -*- coding: utf-8 -*-
"""Matched-mode evaluation of task-aware unequal power allocation.

Runs learned, random-permutation, all-ones, and bypass modes on the identical
locked split.  Every mode resets the channel RNG to the same seed.  No
optimizer is created and all model parameters are frozen.  The evaluator emits
per-clip reconstruction/task metrics, power-distribution diagnostics, accuracy
retention, Wilson confidence intervals, and exact paired McNemar tests.

The current allocator normalises and permutes each encoded frame separately;
therefore it is a spatial UEP evaluator, not a temporal power-transfer test.
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
from model.ta_video_jscc_selector_joint import TAVideoJSCC


MODES = ("learned", "random", "all_ones", "bypass")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector_ckpt", required=True)
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
    p.add_argument(
        "--tau", type=float, required=True,
        help="Temperature belonging to this selected checkpoint; it is not stored in old raw state_dict checkpoints",
    )
    p.add_argument("--random_mask_seed", type=int, default=1729)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="./out_ta_power_allocation_matched")
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inferred_training_config_dir(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    # <root>/checkpoints/<run_tag>/<selected>.pt -> <root>/configs/<run_tag>
    if checkpoint_path.parent.parent.name != "checkpoints":
        raise ValueError(
            "Expected selector checkpoint under <root>/checkpoints/<run_tag>/"
        )
    return checkpoint_path.parent.parent.parent / "configs" / checkpoint_path.parent.name


def load_selector_model(args, device, evaluation_manifest_hash):
    checkpoint_path = Path(args.selector_ckpt)
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and "model_state" in artifact:
        state = artifact["model_state"]
        checkpoint_epoch = artifact.get("epoch")
        embedded_manifest = artifact.get("split_manifest_sha256")
    else:
        state = artifact
        checkpoint_epoch = None
        embedded_manifest = None

    config_dir = inferred_training_config_dir(checkpoint_path)
    manifest_file = config_dir / "split_manifest.sha256"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing selector training manifest hash: {manifest_file}")
    training_manifest_hash = manifest_file.read_text().strip()
    if embedded_manifest is not None and embedded_manifest != training_manifest_hash:
        raise RuntimeError("Embedded and run-directory selector manifest hashes disagree")
    if args.split == "validation" and training_manifest_hash != evaluation_manifest_hash:
        raise RuntimeError("Selector training/evaluation validation manifest mismatch")

    model = TAVideoJSCC(
        c=args.c, channel_type=args.channel, snr=args.snr,
        n_frames=args.gop_size, hidden_dim=args.hidden_dim,
        tsn_head_ckpt=args.tsn_ckpt, lambda_task=0.001,
        lambda_recon=1.0, lambda_rate=0.0, tau=args.tau,
        device=str(device),
    )
    model.load_state_dict(state, strict=True)
    model.set_temperature(args.tau)
    model.random_mask_seed = args.random_mask_seed
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tsn_manifest_hash = model.tsn.checkpoint_metadata.get("split_manifest_sha256")
    if args.split == "validation" and tsn_manifest_hash != evaluation_manifest_hash:
        raise RuntimeError("TSN/evaluation validation manifest mismatch")
    return model, {
        "checkpoint_epoch": checkpoint_epoch,
        "training_config_dir": str(config_dir),
        "selector_training_manifest_sha256": training_manifest_hash,
        "tsn_manifest_sha256": tsn_manifest_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def normalised_entropy(values):
    values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(values.clamp_min(1e-12) * values.clamp_min(1e-12).log()).sum(dim=1)
    return entropy / math.log(values.shape[1])


def top_fraction_share(values, fraction=0.10):
    values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    count = max(1, math.ceil(values.shape[1] * fraction))
    return values.topk(count, dim=1).values.sum(dim=1)


@torch.no_grad()
def evaluate_mode(model, loader, device, mode, channel_seed):
    model.eval()
    model.set_selector_mode(mode)
    rows = []
    energy_ratios = []
    clip_id = 0
    with deterministic_channel(channel_seed, device):
        for gops, labels in tqdm(loader, desc=f"evaluation:{mode}"):
            gops, labels = gops.to(device), labels.to(device)
            reconstructed_raw, mask, _ = model(gops)
            reconstructed = reconstructed_raw.clamp(0, 1)
            logits_recon = model.tsn(reconstructed)
            logits_clean = model.tsn(gops)
            top5_recon = logits_recon.topk(5, dim=1).indices
            top5_clean = logits_clean.topk(5, dim=1).indices
            batch, frames = gops.shape[:2]
            flat_gt = gops.flatten(0, 1)
            flat_pred = reconstructed.flatten(0, 1)
            frame_mse = (flat_pred - flat_gt).square().mean(dim=(1, 2, 3))
            frame_psnr = -10.0 * torch.log10(frame_mse.clamp_min(1e-12))
            frame_ssim = ssim(flat_pred, flat_gt, data_range=1.0, size_average=False)
            frame_ms = ms_ssim(
                flat_pred, flat_gt, data_range=1.0, size_average=False,
                win_size=7, weights=(0.3, 0.3, 0.4),
            )
            raw_mse = (reconstructed_raw - gops).square().mean(dim=(1, 2, 3, 4))

            # Mask has one spatial allocation value broadcast across latent channels.
            mask_by_clip = mask.reshape(batch, frames, -1).reshape(batch, -1)
            mask_mean = mask_by_clip.mean(dim=1)
            mask_std = mask_by_clip.std(dim=1, unbiased=False)
            mask_cv = mask_std / mask_mean.abs().clamp_min(1e-12)
            mask_entropy = normalised_entropy(mask_by_clip)

            # last_spatial_power sums to one separately for every encoded frame.
            # Divide the concatenated five-frame vector by its total before stats.
            power = model.last_spatial_power.reshape(batch, frames, -1).reshape(batch, -1)
            power = power / power.sum(dim=1, keepdim=True).clamp_min(1e-12)
            power_mean = power.mean(dim=1)
            power_std = power.std(dim=1, unbiased=False)
            power_cv = power_std / power_mean.clamp_min(1e-12)
            power_entropy = normalised_entropy(power)
            power_top10_share = top_fraction_share(power, 0.10)
            energy_ratios.append(float(model.last_power_audit["energy_ratio"]))

            for item in range(batch):
                start, end = item * frames, (item + 1) * frames
                rows.append({
                    "clip_id": clip_id,
                    "label": int(labels[item]),
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
                    "mask_mean": float(mask_mean[item]),
                    "mask_std": float(mask_std[item]),
                    "mask_cv": float(mask_cv[item]),
                    "mask_entropy_normalised": float(mask_entropy[item]),
                    "post_power_cv": float(power_cv[item]),
                    "post_power_entropy_normalised": float(power_entropy[item]),
                    "post_power_top10_share": float(power_top10_share[item]),
                })
                clip_id += 1
    return rows, energy_ratios


def mean(rows, key):
    return sum(row[key] for row in rows) / len(rows)


def wilson_interval(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return centre - radius, centre + radius


def aggregate(rows, energy_ratios, ratio):
    count = len(rows)
    top1 = sum(row["reconstructed_top1_correct"] for row in rows)
    top5 = sum(row["reconstructed_top5_correct"] for row in rows)
    clean_top1 = sum(row["clean_top1_correct"] for row in rows)
    clean_top5 = sum(row["clean_top5_correct"] for row in rows)
    low, high = wilson_interval(top1, count)
    return {
        "clips": count,
        "psnr_db": mean(rows, "psnr_db"),
        "ssim": mean(rows, "ssim"),
        "ms_ssim_3scale": mean(rows, "ms_ssim_3scale"),
        "top1_correct": top1,
        "top1_percent": 100 * top1 / count,
        "top1_wilson_95_percent": [100 * low, 100 * high],
        "top5_percent": 100 * top5 / count,
        "clean_top1_percent": 100 * clean_top1 / count,
        "clean_top5_percent": 100 * clean_top5 / count,
        "accuracy_retention_percent": 100 * top1 / clean_top1,
        "actual_cbr": ratio,
        "all_symbols_transmitted": True,
        "mask_mean": mean(rows, "mask_mean"),
        "mask_std": mean(rows, "mask_std"),
        "mask_cv": mean(rows, "mask_cv"),
        "mask_entropy_normalised": mean(rows, "mask_entropy_normalised"),
        "post_power_cv": mean(rows, "post_power_cv"),
        "post_power_entropy_normalised": mean(rows, "post_power_entropy_normalised"),
        "post_power_top10_share": mean(rows, "post_power_top10_share"),
        "energy_ratio_mean": sum(energy_ratios) / len(energy_ratios),
        "energy_ratio_min": min(energy_ratios),
        "energy_ratio_max": max(energy_ratios),
    }


def mcnemar(left_rows, right_rows):
    if len(left_rows) != len(right_rows):
        raise ValueError("Paired modes have different sample counts")
    left_win = right_win = 0
    for left, right in zip(left_rows, right_rows):
        if left["clip_id"] != right["clip_id"] or left["label"] != right["label"]:
            raise RuntimeError("Paired modes are not clip-aligned")
        a = left["reconstructed_top1_correct"]
        b = right["reconstructed_top1_correct"]
        left_win += int(a == 1 and b == 0)
        right_win += int(a == 0 and b == 1)
    discordant = left_win + right_win
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(left_win, right_win) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_correct_right_wrong": left_win,
        "left_wrong_right_correct": right_win,
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p": p_value,
    }


def state_snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_unchanged(model, before):
    changed = [
        key for key, value in model.state_dict().items()
        if not torch.equal(before[key], value.detach().cpu())
    ]
    if changed:
        raise RuntimeError(f"Parameters/buffers changed during evaluation: {changed[:5]}")


def main():
    args = parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader, manifest, manifest_hash = build_loader(args)
    model, checkpoint_metadata = load_selector_model(args, device, manifest_hash)
    before = state_snapshot(model)

    mode_rows = {}
    summaries = {}
    for mode in MODES:
        rows, energy_ratios = evaluate_mode(
            model, loader, device, mode, args.channel_seed
        )
        mode_rows[mode] = rows
        summaries[mode] = aggregate(rows, energy_ratios, args.ratio)
    assert_unchanged(model, before)

    # Clean predictions must be identical in every matched mode.
    reference = mode_rows[MODES[0]]
    for mode in MODES[1:]:
        for left, right in zip(reference, mode_rows[mode]):
            if (
                left["clip_id"] != right["clip_id"]
                or left["label"] != right["label"]
                or left["clean_top1_correct"] != right["clean_top1_correct"]
                or left["clean_top5_correct"] != right["clean_top5_correct"]
            ):
                raise RuntimeError(f"Clean paired alignment failed for mode {mode}")

    pairwise = {}
    for left, right in itertools.combinations(MODES, 2):
        pairwise[f"{left}_vs_{right}"] = {
            "left": left,
            "right": right,
            **mcnemar(mode_rows[left], mode_rows[right]),
        }

    bypass_rows = mode_rows["bypass"]
    ones_rows = mode_rows["all_ones"]
    uniform_check = {
        "max_abs_psnr_difference_db": max(
            abs(a["psnr_db"] - b["psnr_db"])
            for a, b in zip(bypass_rows, ones_rows)
        ),
        "top1_disagreements": sum(
            a["reconstructed_top1_correct"] != b["reconstructed_top1_correct"]
            for a, b in zip(bypass_rows, ones_rows)
        ),
        "predicted_label_disagreements": sum(
            a["reconstructed_predicted_label"] != b["reconstructed_predicted_label"]
            for a, b in zip(bypass_rows, ones_rows)
        ),
    }

    run_name = (
        f"TAPowerMatched_{args.split}_{args.channel}_c{args.c}"
        f"_snr{args.snr:g}_seed{args.channel_seed}"
    )
    output = Path(args.out) / run_name
    output.mkdir(parents=True, exist_ok=False)
    for mode, rows in mode_rows.items():
        with (output / f"per_clip_{mode}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "split_manifest.sha256").write_text(manifest_hash + "\n")
    (output / "mode_summaries.json").write_text(json.dumps(summaries, indent=2))
    (output / "pairwise_mcnemar.json").write_text(json.dumps(pairwise, indent=2))
    evaluation_config = {
        **vars(args),
        "actual_device": str(device),
        "modes": list(MODES),
        "evaluation_split_manifest_sha256": manifest_hash,
        "checkpoint_metadata": checkpoint_metadata,
        "parameter_updates": 0,
        "all_symbols_transmitted": True,
        "temporal_power_transfer_supported": False,
        "allocation_scope": "spatial_within_each_encoded_frame",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, default=str)
    )
    result = {
        "summaries": summaries,
        "pairwise_mcnemar": pairwise,
        "all_ones_vs_bypass_audit": uniform_check,
        "parameter_updates": 0,
        "official_test_used": args.split == "test",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
