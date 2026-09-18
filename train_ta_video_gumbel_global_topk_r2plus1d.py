# -*- coding: utf-8 -*-
"""Train spatial-only or spatiotemporal Gumbel global-Top-K VideoJSCC.

This is a guarded SNR-13 pilot. It uses an exact GoP-level hard budget,
Gumbel exploration only in training, deterministic Top-K in validation, and a
frozen five-frame R(2+1)D evaluator. The official UCF101 test split is unused.
"""

import argparse
import csv
import json
import math
import os
import time
from fractions import Fraction

import torch
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_dataloaders
from model import ratio2filtersize
from model.ta_video_jscc_gumbel_global_topk_r2plus1d import (
    TAVideoJSCCGumbelGlobalTopKR2Plus1D,
)
from train_videojscc_no_selector_joint import fixed_rng, load_videojscc_initialization
from utils import set_seed


def scheduled_value(start, end, epoch, hold_epochs, ramp_epochs):
    if epoch <= hold_epochs:
        return start
    progress = min(1.0, (epoch - hold_epochs) / max(1, ramp_epochs))
    return start + progress * (end - start)


def scheduled_temperature(start, minimum, epoch, total_epochs):
    if total_epochs <= 1:
        return minimum
    progress = (epoch - 1) / (total_epochs - 1)
    return max(minimum, start * ((minimum / start) ** progress))


def set_train_state(model, codec_trainable):
    model.train()
    model.task_model.eval()
    model.set_codec_trainable(codec_trainable)


def train_epoch(model, optimizer, device, loader, scope, codec_trainable, max_iters=None):
    set_train_state(model, codec_trainable)
    model.set_selector_mode(scope)
    totals = {
        "loss": 0.0, "l_task": 0.0, "l_recon": 0.0, "psnr": 0.0,
        "hard_fraction": 0.0, "score_mean": 0.0,
    }
    iterations = 0
    for clips, labels in loader:
        if max_iters is not None and iterations >= max_iters:
            break
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, info, _, _ = model.joint_loss(clips, labels)
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
def evaluate_epoch(model, device, loader, channel_seed, mode):
    model.eval()
    model.task_model.eval()
    model.set_selector_mode(mode)
    model.reset_random_sequence()
    totals = {
        "loss": 0.0, "l_task": 0.0, "l_recon": 0.0, "psnr": 0.0,
        "hard_fraction": 0.0, "score_mean": 0.0,
    }
    correct1 = correct5 = samples = iterations = 0
    frame_counts = None
    with fixed_rng(device, channel_seed):
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            _, info, _, logits = model.joint_loss(clips, labels)
            for key in totals:
                totals[key] += info[key]
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
            samples += labels.numel()
            iterations += 1
            counts = torch.tensor(model.last_selection_audit["frame_keep_counts"])
            frame_counts = counts if frame_counts is None else frame_counts + counts
    if iterations == 0:
        raise RuntimeError("Validation loader produced zero iterations")
    metrics = {key: value / iterations for key, value in totals.items()}
    metrics.update({
        "top1_acc": correct1 / samples,
        "top5_acc": correct5 / samples,
        "samples": samples,
        "frame_keep_counts": (frame_counts / iterations).tolist(),
        "keep_count": model.last_selection_audit["keep_count"],
        "locations": model.last_selection_audit["locations"],
    })
    return metrics


def structural_audit(model, loader, device, channel_seed, scope):
    """Check exact K, spatial sharing, 100%-retention neutrality and gradients."""
    clips, labels = next(iter(loader))
    clips, labels = clips.to(device), labels.to(device)
    original_keep = model.selector.keep_fraction

    model.eval()
    model.set_keep_fraction(1.0)
    with fixed_rng(device, channel_seed):
        model.set_selector_mode("bypass")
        bypass, _, _ = model(clips)
    with fixed_rng(device, channel_seed):
        model.set_selector_mode(scope)
        full_keep, _, _ = model(clips)
    full_keep_max_abs_diff = (bypass - full_keep).abs().max().item()
    if full_keep_max_abs_diff > 2e-5:
        raise RuntimeError(
            f"100%-retention path differs from bypass: {full_keep_max_abs_diff}"
        )

    model.set_keep_fraction(original_keep)
    model.eval(); model.set_selector_mode(scope)
    with fixed_rng(device, channel_seed):
        model(clips)
    audit = dict(model.last_selection_audit)
    expected = audit["keep_count"]
    if audit["hard_count_min"] != expected or audit["hard_count_max"] != expected:
        raise RuntimeError("Hard mask did not contain exactly K selections per GoP")
    if scope == "spatial_only" and len(set(audit["frame_keep_counts"])) != 1:
        raise RuntimeError("Spatial-only mode did not allocate equally across frames")

    set_train_state(model, codec_trainable=False)
    model.set_selector_mode(scope)
    model.zero_grad(set_to_none=True)
    with fixed_rng(device, channel_seed):
        loss, _, _, _ = model.joint_loss(clips, labels)
    loss.backward()
    selector_gradient_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.selector.scorer.parameters()
        if parameter.grad is not None
    )
    evaluator_gradients = sum(
        parameter.grad is not None for parameter in model.task_model.parameters()
    )
    model.zero_grad(set_to_none=True); model.eval()
    if selector_gradient_l1 <= 0:
        raise RuntimeError("Joint loss gradient did not reach the selector")
    if evaluator_gradients:
        raise RuntimeError("Frozen R(2+1)D accumulated gradients")
    result = {
        **audit,
        "full_keep_max_abs_diff": full_keep_max_abs_diff,
        "selector_gradient_l1": selector_gradient_l1,
        "evaluator_gradient_tensors": evaluator_gradients,
        "status": "PASS",
    }
    print("\nStructural audit")
    print(json.dumps(result, indent=2))
    return result


def save_checkpoint(path, model, epoch, metrics, metadata):
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "validation_metrics": metrics,
        **metadata,
    }, path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Guarded Gumbel global-Top-K pilot with frozen R(2+1)D"
    )
    parser.add_argument("--frames_root", default="datasets/UCF101Frames")
    parser.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--gop_size", type=int, default=5)
    parser.add_argument("--gops_per_clip", type=int, default=1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--channel", choices=("AWGN",), default="AWGN")
    parser.add_argument("--snr_list", nargs="+", default=["13"])
    parser.add_argument("--ratio_list", nargs="+", default=["1/6"])
    parser.add_argument("--allocation_scope", choices=("spatial_only", "spatiotemporal"), required=True)
    parser.add_argument("--videojscc_ckpt", required=True)
    parser.add_argument("--r2plus1d_ckpt", required=True)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--lambda_task", type=float, default=0.001)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--start_keep_fraction", type=float, default=0.9)
    parser.add_argument("--target_keep_fraction", type=float, default=0.5)
    parser.add_argument("--keep_hold_epochs", type=int, default=3)
    parser.add_argument("--keep_ramp_epochs", type=int, default=10)
    parser.add_argument("--selector_only_epochs", type=int, default=3)
    parser.add_argument("--tau_start", type=float, default=1.0)
    parser.add_argument("--tau_min", type=float, default=0.3)
    parser.add_argument("--collapse_psnr_floor", type=float, default=15.0)
    parser.add_argument("--collapse_max_drop", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--init_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_seed", type=int, default=1729)
    parser.add_argument("--val_channel_seed", type=int, default=1042)
    parser.add_argument("--max_iters_per_epoch", type=int, default=None)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--out", default="./out_ta_gumbel_global_topk_r2plus1d")
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
    model = TAVideoJSCCGumbelGlobalTopKR2Plus1D(
        c=c, channel_type=params["channel"], snr=params["snr"],
        n_frames=params["gop_size"], hidden_dim=params["hidden_dim"],
        r2plus1d_ckpt=params["r2plus1d_ckpt"],
        lambda_task=params["lambda_task"], lambda_recon=params["lambda_recon"],
        keep_fraction=params["start_keep_fraction"], tau=params["tau_start"],
        random_seed=params["random_seed"],
    ).to(device)
    baseline = load_videojscc_initialization(model, params["videojscc_ckpt"])
    if baseline["split_manifest_sha256"] != manifest_hash:
        raise RuntimeError("Baseline/training split-manifest mismatch")
    evaluator_hash = model.task_model.checkpoint_metadata.get("split_manifest_sha256")
    if evaluator_hash != manifest_hash:
        raise RuntimeError("R(2+1)D/training split-manifest mismatch")
    print(f"[Split audit] PASS: {manifest_hash}")
    audit = structural_audit(
        model, val_loader, device, params["val_channel_seed"], params["allocation_scope"]
    )
    if params["audit_only"]:
        print("Audit-only run completed; no optimizer step was taken.")
        return

    model.set_keep_fraction(1.0)
    bypass_metrics = evaluate_epoch(
        model, device, val_loader, params["val_channel_seed"], "bypass"
    )
    collapse_threshold = max(
        params["collapse_psnr_floor"],
        bypass_metrics["psnr"] - params["collapse_max_drop"],
    )
    print(
        f"Collapse guard: bypass={bypass_metrics['psnr']:.2f} dB | "
        f"stop below {collapse_threshold:.2f} dB"
    )
    model.set_keep_fraction(params["start_keep_fraction"])

    # Include codec parameters in the optimizer now; requires_grad is toggled
    # during selector-only warm-up without losing optimizer state later.
    optimized_parameters = [
        parameter for name, parameter in model.named_parameters()
        if not name.startswith("task_model.")
    ]
    optimizer = optim.Adam(
        optimized_parameters, lr=params["init_lr"], weight_decay=params["weight_decay"]
    )

    tag = (
        f"TAGumbelGlobalTopK_{params['allocation_scope']}_{params['channel']}"
        f"_c{c}_snr{params['snr']}_ratio{params['ratio']:.4f}"
        f"_keep{params['target_keep_fraction']:.2f}_{params['epochs']}ep_"
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
        "epoch", "codec_trainable", "keep_fraction", "tau", "train_loss",
        "train_task_loss", "train_recon_loss", "train_psnr", "val_loss",
        "val_task_loss", "val_recon_loss", "val_psnr", "val_top1", "val_top5",
        "hard_fraction", "score_mean", "frame_keep_counts",
    ]
    metrics_handle = open(os.path.join(config_dir, "metrics.csv"), "w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    metrics_writer.writeheader()
    writer = SummaryWriter(log_dir=log_dir)
    metadata = {
        "method": "gumbel_straight_through_global_topk_r2plus1d",
        "allocation_scope": params["allocation_scope"],
        "split_manifest_sha256": manifest_hash,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "baseline_split_manifest_sha256": baseline["split_manifest_sha256"],
        "r2plus1d_split_manifest_sha256": evaluator_hash,
        "target_keep_fraction": params["target_keep_fraction"],
        "official_test_used": False,
    }
    best_top1 = float("-inf"); best_epoch = -1
    completed_epochs = 0; collapse = None; started = time.time()
    try:
        with tqdm(range(1, params["epochs"] + 1), disable=params["disable_tqdm"]) as progress:
            for epoch in progress:
                keep = scheduled_value(
                    params["start_keep_fraction"], params["target_keep_fraction"],
                    epoch, params["keep_hold_epochs"], params["keep_ramp_epochs"],
                )
                tau = scheduled_temperature(
                    params["tau_start"], params["tau_min"], epoch, params["epochs"]
                )
                codec_trainable = epoch > params["selector_only_epochs"]
                model.set_keep_fraction(keep); model.set_temperature(tau)
                train = train_epoch(
                    model, optimizer, device, train_loader, params["allocation_scope"],
                    codec_trainable, params["max_iters_per_epoch"],
                )
                val = evaluate_epoch(
                    model, device, val_loader, params["val_channel_seed"],
                    params["allocation_scope"],
                )
                completed_epochs = epoch
                row = {
                    "epoch": epoch, "codec_trainable": codec_trainable,
                    "keep_fraction": keep, "tau": tau,
                    "train_loss": train["loss"], "train_task_loss": train["l_task"],
                    "train_recon_loss": train["l_recon"], "train_psnr": train["psnr"],
                    "val_loss": val["loss"], "val_task_loss": val["l_task"],
                    "val_recon_loss": val["l_recon"], "val_psnr": val["psnr"],
                    "val_top1": val["top1_acc"], "val_top5": val["top5_acc"],
                    "hard_fraction": val["hard_fraction"],
                    "score_mean": val["score_mean"],
                    "frame_keep_counts": json.dumps(val["frame_keep_counts"]),
                }
                metrics_writer.writerow(row); metrics_handle.flush()
                for key, value in train.items(): writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val.items():
                    if isinstance(value, (int, float)): writer.add_scalar(f"val/{key}", value, epoch)
                progress.set_description(f"Epoch {epoch}")
                progress.set_postfix(
                    keep=f"{keep:.2f}", tau=f"{tau:.2f}",
                    psnr=f"{val['psnr']:.2f}", acc=f"{100*val['top1_acc']:.1f}%",
                )
                save_checkpoint(
                    os.path.join(ckpt_dir, "latest.pt"), model, epoch, val, metadata
                )
                if val["psnr"] < collapse_threshold:
                    collapse = {
                        "epoch": epoch, "val_psnr": val["psnr"],
                        "threshold": collapse_threshold, "val_top1": val["top1_acc"],
                    }
                    save_checkpoint(
                        os.path.join(ckpt_dir, "collapse_guard.pt"),
                        model, epoch, val, metadata,
                    )
                    print("\nCOLLAPSE GUARD TRIGGERED: " + json.dumps(collapse))
                    break
                at_target_budget = abs(
                    keep - params["target_keep_fraction"]
                ) < 1e-12
                if at_target_budget and val["top1_acc"] > best_top1:
                    best_top1, best_epoch = val["top1_acc"], epoch
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best_top1.pt"), model, epoch, val, metadata
                    )
    finally:
        metrics_handle.close(); writer.close()

    control_metrics = {}
    if best_epoch > 0:
        artifact = torch.load(
            os.path.join(ckpt_dir, "best_top1.pt"), map_location=device,
            weights_only=False,
        )
        model.load_state_dict(artifact["model_state"], strict=True)
        model.set_keep_fraction(artifact["validation_metrics"]["hard_fraction"])
        control_metrics["learned"] = evaluate_epoch(
            model, device, val_loader, params["val_channel_seed"], params["allocation_scope"]
        )
        random_mode = "random_" + params["allocation_scope"]
        control_metrics["random"] = evaluate_epoch(
            model, device, val_loader, params["val_channel_seed"], random_mode
        )
    summary = {
        "status": "collapse_guard_triggered" if collapse else "completed",
        "epochs_completed": completed_epochs,
        "best_top1_epoch": best_epoch,
        "best_top1": best_top1 if best_epoch > 0 else None,
        "bypass_metrics": bypass_metrics,
        "selected_checkpoint_controls": control_metrics,
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
    args = build_parser().parse_args()
    if args.target_keep_fraction > args.start_keep_fraction:
        raise ValueError("target_keep_fraction cannot exceed start_keep_fraction")
    if args.tau_min > args.tau_start:
        raise ValueError("tau_min cannot exceed tau_start")
    set_seed(args.seed)
    params = vars(args)
    params["snr_list"] = list(map(float, args.snr_list))
    params["ratio_list"] = [float(Fraction(value)) for value in args.ratio_list]
    for ratio in params["ratio_list"]:
        for snr in params["snr_list"]:
            run = dict(params); run["ratio"] = ratio; run["snr"] = snr
            train_pipeline(run)


if __name__ == "__main__":
    main()
