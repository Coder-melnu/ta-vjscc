# -*- coding: utf-8 -*-
"""Train matched uniform or saliency-supervised GoP power allocation.

Existing channel/model/training files are not used or modified. Both modes
transmit every c=8 (or c=4) latent symbol with identical total GoP energy and
fixed AWGN variance. Official UCF101 test data is never used for selection.
"""

import argparse
import csv
import json
import os
import time
from fractions import Fraction

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_dataloaders
from model import ratio2filtersize
from model.channel import Channel
from model.ta_video_jscc_saliency_spatiotemporal_r2plus1d import (
    TAVideoJSCCSaliencySpatiotemporalR2Plus1D,
)
from train_videojscc_no_selector_joint import fixed_rng, load_videojscc_initialization
from utils import set_seed


def set_codec_trainable(model, trainable):
    for module in (model.jscc, model.temporal):
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)


def train_epoch(model, optimizer, device, loader, mode, codec_trainable, max_iters=None):
    model.train(); model.task_model.eval(); model.set_allocation_mode(mode)
    set_codec_trainable(model, codec_trainable)
    totals = {
        "loss": 0.0, "l_recon": 0.0, "l_task": 0.0,
        "l_importance": 0.0, "psnr": 0.0,
        "relative_power_mean": 0.0, "relative_power_cv": 0.0,
    }
    iterations = 0
    for clips, labels in loader:
        if max_iters is not None and iterations >= max_iters:
            break
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, info, _, _ = model.joint_loss(
            clips, labels, use_importance_teacher=(mode == "spatiotemporal")
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        for key in totals:
            totals[key] += info[key]
        iterations += 1
    if iterations == 0:
        raise RuntimeError("Training loader produced zero iterations")
    return {key: value / iterations for key, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(model, device, loader, mode, channel_seed):
    model.eval(); model.task_model.eval(); model.set_allocation_mode(mode)
    totals = {
        "loss": 0.0, "l_recon": 0.0, "l_task": 0.0,
        "l_importance": 0.0, "psnr": 0.0,
        "relative_power_mean": 0.0, "relative_power_cv": 0.0,
    }
    correct1 = correct5 = samples = iterations = 0
    frame_energy_share = None
    energy_min, energy_max = float("inf"), float("-inf")
    with fixed_rng(device, channel_seed):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            _, info, _, logits = model.joint_loss(
                clips, labels, use_importance_teacher=False
            )
            for key in totals:
                totals[key] += info[key]
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
            samples += labels.numel(); iterations += 1
            audit = model.last_allocation_audit
            share = torch.tensor(audit["frame_energy_share"])
            frame_energy_share = share if frame_energy_share is None else frame_energy_share + share
            energy_min = min(energy_min, audit["energy_ratio_min"])
            energy_max = max(energy_max, audit["energy_ratio_max"])
    metrics = {key: value / iterations for key, value in totals.items()}
    metrics.update({
        "top1_acc": correct1 / samples,
        "top5_acc": correct5 / samples,
        "samples": samples,
        "frame_energy_share": (frame_energy_share / iterations).tolist(),
        "energy_ratio_min": energy_min,
        "energy_ratio_max": energy_max,
        "noise_variance": model.gop_channel.last_audit["noise_variance"],
        "all_symbols_transmitted": True,
    })
    return metrics


def structural_audit(model, loader, device, channel_seed):
    """Prove legacy equivalence, fixed noise/energy, shapes and gradients."""
    clips, labels = next(iter(loader))
    clips, labels = clips[:1].to(device), labels[:1].to(device)
    model.eval(); model.task_model.eval()
    latent_gop = model.encode(clips)
    flat_latent = latent_gop.reshape(-1, *latent_gop.shape[2:])
    legacy_channel = Channel("AWGN", model.gop_channel.snr).to(device)

    with fixed_rng(device, channel_seed):
        legacy_received = legacy_channel(flat_latent).reshape_as(latent_gop)
    with fixed_rng(device, channel_seed):
        gop_received = model.gop_channel(latent_gop)
    channel_equivalence = (legacy_received - gop_received).abs().max().item()
    if channel_equivalence > 2e-5:
        raise RuntimeError(
            f"Uniform GoP channel does not reproduce legacy AWGN: {channel_equivalence}"
        )

    model.set_allocation_mode("uniform")
    transmitted, allocation, _, uniform_gain = model.allocate(latent_gop)
    uniform_transmit_difference = (transmitted - latent_gop).abs().max().item()
    if uniform_transmit_difference > 2e-5:
        raise RuntimeError("Uniform allocation changed the transmitted latent")
    if allocation.min().item() != 1.0 or allocation.max().item() != 1.0:
        raise RuntimeError("Uniform allocation map is not exactly one")
    uniform_gain_difference = (uniform_gain - 1.0).abs().max().item()
    if uniform_gain_difference > 2e-6:
        raise RuntimeError("Uniform receiver gain is not exactly one")

    reference_energy = latent_gop.square().flatten(1).sum(1)
    transmit_energy = transmitted.square().flatten(1).sum(1)
    energy_ratio = transmit_energy / reference_energy
    if not torch.allclose(energy_ratio, torch.ones_like(energy_ratio), atol=2e-6):
        raise RuntimeError("Uniform GoP energy constraint failed")

    # For a non-uniform map, prove that allocation changes only effective
    # noise protection, not the clean latent values presented to the decoder.
    model.set_allocation_mode("spatiotemporal")
    learned_transmitted, _, _, learned_gain = model.allocate(latent_gop)
    clean_equalized = learned_transmitted / learned_gain.clamp_min(1e-6)
    clean_equalization_difference = (
        clean_equalized - latent_gop
    ).abs().max().item()
    if clean_equalization_difference > 2e-5:
        raise RuntimeError("Receiver equalization failed to restore clean latent scale")

    # Test the learned path and explicit importance-supervision gradient.
    set_codec_trainable(model, False)
    model.zero_grad(set_to_none=True)
    with fixed_rng(device, channel_seed):
        loss, info, reconstructed, _ = model.joint_loss(
            clips, labels, use_importance_teacher=True
        )
    loss.backward()
    scorer_gradient_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.scorer.parameters()
        if parameter.grad is not None
    )
    evaluator_gradient_tensors = sum(
        parameter.grad is not None for parameter in model.task_model.parameters()
    )
    audit = dict(model.last_allocation_audit)
    model.zero_grad(set_to_none=True); model.eval()
    if scorer_gradient_l1 <= 0:
        raise RuntimeError("Importance/task losses did not reach the scorer")
    if evaluator_gradient_tensors:
        raise RuntimeError("Frozen R(2+1)D accumulated gradients")
    if reconstructed.shape != clips.shape:
        raise RuntimeError("Existing decoder did not preserve the clip shape")
    if not audit["all_symbols_transmitted"]:
        raise RuntimeError("At least one latent symbol was not transmitted")
    if not audit["noise_variance_is_frame_independent"]:
        raise RuntimeError("GoP noise variance is not frame independent")
    result = {
        "legacy_uniform_channel_max_abs_diff": channel_equivalence,
        "uniform_transmit_max_abs_diff": uniform_transmit_difference,
        "uniform_receiver_gain_max_abs_diff": uniform_gain_difference,
        "uniform_energy_ratio": energy_ratio.detach().cpu().tolist(),
        "learned_clean_equalization_max_abs_diff": clean_equalization_difference,
        "decoder_input_shape": list(flat_latent.shape),
        "decoder_output_shape": list(reconstructed.shape),
        "all_symbols_transmitted": audit["all_symbols_transmitted"],
        "learned_energy_ratio_min": audit["energy_ratio_min"],
        "learned_energy_ratio_max": audit["energy_ratio_max"],
        "fixed_noise_variance": audit["noise_variance"],
        "noise_variance_is_frame_independent": audit["noise_variance_is_frame_independent"],
        "receiver_gain_min": audit["receiver_gain_min"],
        "receiver_gain_max": audit["receiver_gain_max"],
        "allocation_map_available_at_receiver": audit["allocation_map_available_at_receiver"],
        "importance_loss": info["l_importance"],
        "scorer_gradient_l1": scorer_gradient_l1,
        "evaluator_gradient_tensors": evaluator_gradient_tensors,
        "status": "PASS",
    }
    print("\nGoP channel and saliency-selector audit")
    print(json.dumps(result, indent=2))
    return result


def save_checkpoint(path, model, epoch, metrics, metadata):
    torch.save({
        "model_state": model.state_dict(), "epoch": epoch,
        "validation_metrics": metrics, **metadata,
    }, path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Uniform or saliency-supervised GoP allocation"
    )
    parser.add_argument("--allocation_mode", choices=("uniform", "spatiotemporal"), required=True)
    parser.add_argument("--frames_root", default="datasets/UCF101Frames")
    parser.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--gop_size", type=int, default=5)
    parser.add_argument("--gops_per_clip", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--snr_list", nargs="+", default=["13"])
    parser.add_argument("--ratio_list", nargs="+", default=["1/6"])
    parser.add_argument("--videojscc_ckpt", required=True)
    parser.add_argument("--r2plus1d_ckpt", required=True)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--scorer_hidden", type=int, default=16)
    parser.add_argument("--lambda_task", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_importance", type=float, default=0.01)
    parser.add_argument("--allocation_temperature", type=float, default=1.0)
    parser.add_argument("--min_relative_power", type=float, default=0.1)
    parser.add_argument("--scorer_only_epochs", type=int, default=2)
    parser.add_argument("--collapse_psnr_floor", type=float, default=15.0)
    parser.add_argument("--collapse_max_drop", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_channel_seed", type=int, default=1042)
    parser.add_argument("--max_iters_per_epoch", type=int, default=None)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--out", default="./out_ta_saliency_spatiotemporal_r2plus1d")
    parser.add_argument("--device", default="cuda:0")
    return parser


def train_pipeline(params):
    train_loader, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=params["frames_root"], annotation_path=params["annotation_path"],
        image_size=params["image_size"], batch_size=params["batch_size"],
        num_workers=params["num_workers"], gop_size=params["gop_size"],
        gops_per_clip=params["gops_per_clip"], val_fraction=params["val_fraction"],
        seed=params["seed"],
    )
    c = ratio2filtersize(
        torch.empty(3, params["image_size"], params["image_size"]), params["ratio"]
    )
    device_string = params["device"] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    model = TAVideoJSCCSaliencySpatiotemporalR2Plus1D(
        c=c, snr=params["snr"], n_frames=params["gop_size"],
        hidden_dim=params["hidden_dim"], scorer_hidden=params["scorer_hidden"],
        r2plus1d_ckpt=params["r2plus1d_ckpt"],
        lambda_task=params["lambda_task"], lambda_recon=params["lambda_recon"],
        lambda_importance=params["lambda_importance"],
        allocation_temperature=params["allocation_temperature"],
        min_relative_power=params["min_relative_power"],
    ).to(device)
    baseline = load_videojscc_initialization(model, params["videojscc_ckpt"])
    if baseline["split_manifest_sha256"] != manifest_hash:
        raise RuntimeError("Baseline/training split-manifest mismatch")
    evaluator_hash = model.task_model.checkpoint_metadata.get("split_manifest_sha256")
    if evaluator_hash != manifest_hash:
        raise RuntimeError("R(2+1)D/training split-manifest mismatch")
    print(f"[Split audit] PASS: {manifest_hash}")
    audit = structural_audit(model, val_loader, device, params["val_channel_seed"])
    if params["audit_only"]:
        print("Audit-only run completed; no optimizer step was taken.")
        return

    uniform_metrics = evaluate_epoch(
        model, device, val_loader, "uniform", params["val_channel_seed"]
    )
    collapse_threshold = max(
        params["collapse_psnr_floor"],
        uniform_metrics["psnr"] - params["collapse_max_drop"],
    )
    print(
        f"Collapse guard: uniform={uniform_metrics['psnr']:.2f} dB | "
        f"stop below {collapse_threshold:.2f} dB"
    )
    optimized = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("task_model.")
    ]
    optimizer = optim.Adam(
        optimized, lr=params["init_lr"], weight_decay=params["weight_decay"]
    )
    tag = (
        f"TASaliency_{params['allocation_mode']}_GoPAWGN_c{c}_snr{params['snr']}"
        f"_ratio{params['ratio']:.4f}_{params['epochs']}ep_"
        f"{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}"
    )
    ckpt_dir = os.path.join(params["out"], "checkpoints", tag)
    config_dir = os.path.join(params["out"], "configs", tag)
    log_dir = os.path.join(params["out"], "logs", tag)
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "split_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(config_dir, "split_manifest.sha256"), "w") as handle:
        handle.write(manifest_hash + "\n")
    fields = [
        "epoch", "codec_trainable", "train_loss", "train_recon_loss",
        "train_task_loss", "train_importance_loss", "train_psnr",
        "val_loss", "val_recon_loss", "val_task_loss", "val_psnr",
        "val_top1", "val_top5", "relative_power_mean", "relative_power_cv",
        "frame_energy_share",
    ]
    metrics_handle = open(os.path.join(config_dir, "metrics.csv"), "w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    metrics_writer.writeheader()
    writer = SummaryWriter(log_dir=log_dir)
    metadata = {
        "method": "saliency_supervised_spatiotemporal_power_allocation",
        "allocation_mode": params["allocation_mode"],
        "split_manifest_sha256": manifest_hash,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "r2plus1d_split_manifest_sha256": evaluator_hash,
        "all_symbols_transmitted": True,
        "gop_fixed_noise_variance": True,
        "official_test_used": False,
    }
    best_top1 = float("-inf"); best_epoch = -1
    collapse = None; completed_epochs = 0; started = time.time()
    try:
        with tqdm(range(1, params["epochs"] + 1), disable=params["disable_tqdm"]) as progress:
            for epoch in progress:
                codec_trainable = (
                    params["allocation_mode"] == "uniform"
                    or epoch > params["scorer_only_epochs"]
                )
                train = train_epoch(
                    model, optimizer, device, train_loader,
                    params["allocation_mode"], codec_trainable,
                    params["max_iters_per_epoch"],
                )
                val = evaluate_epoch(
                    model, device, val_loader, params["allocation_mode"],
                    params["val_channel_seed"],
                )
                completed_epochs = epoch
                metrics_writer.writerow({
                    "epoch": epoch, "codec_trainable": codec_trainable,
                    "train_loss": train["loss"],
                    "train_recon_loss": train["l_recon"],
                    "train_task_loss": train["l_task"],
                    "train_importance_loss": train["l_importance"],
                    "train_psnr": train["psnr"], "val_loss": val["loss"],
                    "val_recon_loss": val["l_recon"],
                    "val_task_loss": val["l_task"], "val_psnr": val["psnr"],
                    "val_top1": val["top1_acc"], "val_top5": val["top5_acc"],
                    "relative_power_mean": val["relative_power_mean"],
                    "relative_power_cv": val["relative_power_cv"],
                    "frame_energy_share": json.dumps(val["frame_energy_share"]),
                }); metrics_handle.flush()
                for key, value in train.items(): writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val.items():
                    if isinstance(value, (int, float)): writer.add_scalar(f"val/{key}", value, epoch)
                progress.set_description(f"Epoch {epoch}")
                progress.set_postfix(
                    psnr=f"{val['psnr']:.2f}", acc=f"{100*val['top1_acc']:.1f}%",
                    cv=f"{val['relative_power_cv']:.3f}",
                )
                save_checkpoint(
                    os.path.join(ckpt_dir, "latest.pt"), model, epoch, val, metadata
                )
                if val["psnr"] < collapse_threshold:
                    collapse = {
                        "epoch": epoch, "val_psnr": val["psnr"],
                        "threshold": collapse_threshold,
                        "val_top1": val["top1_acc"],
                    }
                    save_checkpoint(
                        os.path.join(ckpt_dir, "collapse_guard.pt"),
                        model, epoch, val, metadata,
                    )
                    print("\nCOLLAPSE GUARD TRIGGERED: " + json.dumps(collapse))
                    break
                if val["top1_acc"] > best_top1:
                    best_top1, best_epoch = val["top1_acc"], epoch
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best_top1.pt"),
                        model, epoch, val, metadata,
                    )
    finally:
        metrics_handle.close(); writer.close()
    selected_metrics = None
    uniform_at_selected = None
    if best_epoch > 0:
        artifact = torch.load(
            os.path.join(ckpt_dir, "best_top1.pt"), map_location=device,
            weights_only=False,
        )
        model.load_state_dict(artifact["model_state"], strict=True)
        selected_metrics = evaluate_epoch(
            model, device, val_loader, params["allocation_mode"],
            params["val_channel_seed"],
        )
        uniform_at_selected = evaluate_epoch(
            model, device, val_loader, "uniform", params["val_channel_seed"]
        )
    summary = {
        "status": "collapse_guard_triggered" if collapse else "completed",
        "epochs_completed": completed_epochs,
        "best_top1_epoch": best_epoch,
        "best_top1_metrics": selected_metrics,
        "same_weights_uniform_diagnostic": uniform_at_selected,
        "initial_uniform_metrics": uniform_metrics,
        "collapse_guard": collapse,
        "structural_audit": audit,
        "split_manifest_sha256": manifest_hash,
        "official_test_used": False,
        "total_hours": (time.time() - started) / 3600,
    }
    with open(os.path.join(config_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(config_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump({**params, **metadata, "c": c, "actual_device": device_string}, handle)
    print(json.dumps(summary, indent=2))
    print(f"Saved checkpoints: {ckpt_dir}")


def main():
    args = build_parser().parse_args(); set_seed(args.seed)
    params = vars(args)
    params["snr_list"] = list(map(float, args.snr_list))
    params["ratio_list"] = [float(Fraction(value)) for value in args.ratio_list]
    for ratio in params["ratio_list"]:
        for snr in params["snr_list"]:
            run = dict(params); run["ratio"] = ratio; run["snr"] = snr
            train_pipeline(run)


if __name__ == "__main__":
    main()
