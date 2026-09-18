# -*- coding: utf-8 -*-
"""Train staged variance-normalised continuous spatial UEP.

All symbols remain transmitted. Learned spatial scores produce continuous
relative power values using a bounded tanh mapping. Per-frame unit-mean
normalisation and exact sample-wise energy renormalisation preserve the
communication budget. Model selection uses only locked internal validation.
"""

import argparse
import csv
import json
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
from model.ta_video_jscc_uep_continuous_staged import TAVideoJSCCStagedContinuousUEP
from train_videojscc_no_selector_joint import fixed_rng, load_videojscc_initialization
from utils import set_seed


def configure_stage(model, stage):
    """Set trainable parameters for allocator-only or joint training."""
    if stage not in {"A", "B"}:
        raise ValueError("stage must be A or B")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.allocator.scorer.parameters():
        parameter.requires_grad_(True)

    if stage == "B":
        for parameter in model.jscc.parameters():
            parameter.requires_grad_(True)
        for parameter in model.temporal.parameters():
            parameter.requires_grad_(True)

    for parameter in model.tsn.parameters():
        parameter.requires_grad_(False)


def set_training_modes(model, stage):
    """Prevent state changes in modules frozen during each stage."""
    model.train()
    model.tsn.eval()

    if stage == "A":
        model.jscc.eval()
        model.temporal.eval()
        model.allocator.train()
        model.allocator.scorer.train()


def make_optimizer(model, stage, params):
    configure_stage(model, stage)

    scorer = list(model.allocator.scorer.parameters())

    if stage == "A":
        return optim.Adam(
            [{
                "params": scorer,
                "lr": params["scorer_lr_stage_a"],
                "weight_decay": 0.0,
                "group_name": "scorer",
            }]
        )

    codec = (
        list(model.jscc.parameters())
        + list(model.temporal.parameters())
    )
    return optim.Adam([
        {
            "params": scorer,
            "lr": params["scorer_lr_stage_b"],
            "weight_decay": 0.0,
            "group_name": "scorer",
        },
        {
            "params": codec,
            "lr": params["codec_lr_stage_b"],
            "weight_decay": params["codec_weight_decay"],
            "group_name": "codec",
        },
    ])


def optimizer_lrs(optimizer):
    values = {
        group.get("group_name", f"group_{index}"): group["lr"]
        for index, group in enumerate(optimizer.param_groups)
    }
    return values.get("scorer", 0.0), values.get("codec", 0.0)


def train_epoch(
    model, optimizer, device, loader, stage, max_iters=None
):
    set_training_modes(model, stage)
    model.set_allocation_mode("learned")
    totals = {
        "loss": 0.0, "l_task": 0.0, "l_recon": 0.0, "psnr": 0.0,
        "relative_power_mean": 0.0, "relative_power_std": 0.0,
        "relative_power_cv": 0.0, "relative_power_min": 0.0, "relative_power_max": 0.0,
    }
    iterations = 0
    for gops, labels in loader:
        if max_iters is not None and iterations >= max_iters:
            break
        gops, labels = gops.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, info, _, _ = model.joint_loss(gops, labels)
        loss.backward()
        optimizer.step()
        for key in totals:
            totals[key] += info[key]
        iterations += 1
    if iterations == 0:
        raise RuntimeError("Training loader produced zero iterations")
    return {key: value / iterations for key, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(model, device, loader, channel_seed, mode="learned"):
    model.eval()
    model.set_allocation_mode(mode)
    model.reset_random_sequence()
    totals = {
        "loss": 0.0, "l_task": 0.0, "l_recon": 0.0, "psnr": 0.0,
        "relative_power_mean": 0.0, "relative_power_std": 0.0,
        "relative_power_cv": 0.0, "relative_power_min": 0.0, "relative_power_max": 0.0,
    }
    correct1 = correct5 = samples = iterations = 0
    energy_min, energy_max = float("inf"), float("-inf")
    with fixed_rng(device, channel_seed):
        for gops, labels in loader:
            gops, labels = gops.to(device), labels.to(device)
            _, info, _, logits = model.joint_loss(gops, labels)
            for key in totals:
                totals[key] += info[key]
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
            samples += labels.numel(); iterations += 1
            energy_min = min(energy_min, model.last_power_audit["energy_ratio_min"])
            energy_max = max(energy_max, model.last_power_audit["energy_ratio_max"])
    metrics = {key: value / iterations for key, value in totals.items()}
    metrics.update({
        "top1_acc": correct1 / samples,
        "top5_acc": correct5 / samples,
        "samples": samples,
        "energy_ratio_min": energy_min,
        "energy_ratio_max": energy_max,
        "all_symbols_transmitted": True,
    })
    return metrics


def audit_staged(
    model, loader, device, channel_seed, params
):
    """Audit allocation, Stage A, Stage B, and frozen TSN."""
    model.eval()
    gops, labels = next(iter(loader))
    gops, labels = gops.to(device), labels.to(device)

    def run(mode):
        model.eval()
        model.set_allocation_mode(mode)
        model.reset_random_sequence()
        with fixed_rng(device, channel_seed):
            reconstructed, power, scores = model(gops)
        return (
            reconstructed,
            power,
            scores,
            dict(model.last_power_audit),
        )

    bypass, _, _, _ = run("bypass")
    all_ones, _, _, _ = run("all_ones")
    _, learned_power, learned_scores, learned_audit = run(
        "learned"
    )

    uniform_max_abs_diff = (
        bypass - all_ones
    ).abs().max().item()
    if uniform_max_abs_diff > 1e-5:
        raise RuntimeError(
            "All-ones allocation does not reproduce bypass"
        )

    constant_scores = torch.full_like(learned_scores, 3.0)
    constant_power = model.allocator._scores_to_power(
        constant_scores
    )
    constant_score_error = (
        constant_power - torch.ones_like(constant_power)
    ).abs().max().item()
    if constant_score_error > 1e-7:
        raise RuntimeError(
            "Constant scores do not produce uniform power"
        )

    power_small = model.allocator._scores_to_power(
        learned_scores * 0.1
    )
    power_large = model.allocator._scores_to_power(
        learned_scores * 10.0
    )
    small_scale_error = (
        learned_power - power_small
    ).abs().max().item()
    large_scale_error = (
        learned_power - power_large
    ).abs().max().item()

    if small_scale_error > 2e-6:
        raise RuntimeError(
            "Power changed when scores were scaled by 0.1"
        )
    if large_scale_error > 2e-6:
        raise RuntimeError(
            "Power changed when scores were scaled by 10"
        )

    per_frame_means = learned_power.flatten(1).mean(dim=1)
    unit_mean_error = (
        per_frame_means - torch.ones_like(per_frame_means)
    ).abs().max().item()
    if unit_mean_error > 1e-6:
        raise RuntimeError(
            "Relative power is not unit mean per frame"
        )

    if not learned_audit["all_symbols_transmitted"]:
        raise RuntimeError("At least one symbol was dropped")

    if (
        abs(learned_audit["energy_ratio_min"] - 1.0) > 2e-6
        or abs(
            learned_audit["energy_ratio_max"] - 1.0
        ) > 2e-6
    ):
        raise RuntimeError(
            "Total channel-input energy was not preserved"
        )

    codec_state_before = {
        name: value.detach().cpu().clone()
        for name, value in {
            **{
                f"jscc.{name}": value
                for name, value in model.jscc.state_dict().items()
            },
            **{
                f"temporal.{name}": value
                for name, value
                in model.temporal.state_dict().items()
            },
        }.items()
    }

    # Stage A: scorer only.
    configure_stage(model, "A")
    set_training_modes(model, "A")
    model.set_allocation_mode("learned")
    model.zero_grad(set_to_none=True)

    _, _, _, logits = model.joint_loss(gops, labels)
    task_only = torch.nn.functional.cross_entropy(
        logits, labels
    )
    task_only.backward()

    stage_a_scorer_gradient_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.allocator.scorer.parameters()
        if parameter.grad is not None
    )
    stage_a_codec_gradient_tensors = sum(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith(("jscc.", "temporal."))
    )
    stage_a_tsn_gradient_tensors = sum(
        parameter.grad is not None
        for parameter in model.tsn.parameters()
    )

    codec_state_after = {
        name: value.detach().cpu().clone()
        for name, value in {
            **{
                f"jscc.{name}": value
                for name, value in model.jscc.state_dict().items()
            },
            **{
                f"temporal.{name}": value
                for name, value
                in model.temporal.state_dict().items()
            },
        }.items()
    }
    stage_a_codec_state_changes = [
        name
        for name in codec_state_before
        if not torch.equal(
            codec_state_before[name],
            codec_state_after[name],
        )
    ]

    if stage_a_scorer_gradient_l1 <= 0:
        raise RuntimeError(
            "Stage A task gradient did not reach scorer"
        )
    if stage_a_codec_gradient_tensors != 0:
        raise RuntimeError(
            "Stage A produced codec/TFM parameter gradients"
        )
    if stage_a_tsn_gradient_tensors != 0:
        raise RuntimeError(
            "Stage A produced TSN parameter gradients"
        )
    if stage_a_codec_state_changes:
        raise RuntimeError(
            "Stage A changed frozen codec/TFM state: "
            f"{stage_a_codec_state_changes[:5]}"
        )
    if model.jscc.training or model.temporal.training:
        raise RuntimeError(
            "Stage A codec/TFM did not remain in eval mode"
        )

    # Stage B: scorer and codec/TFM.
    configure_stage(model, "B")
    set_training_modes(model, "B")
    model.set_allocation_mode("learned")
    model.zero_grad(set_to_none=True)

    _, _, _, logits = model.joint_loss(gops, labels)
    task_only = torch.nn.functional.cross_entropy(
        logits, labels
    )
    task_only.backward()

    stage_b_scorer_gradient_l1 = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.allocator.scorer.parameters()
        if parameter.grad is not None
    )
    stage_b_codec_gradient_l1 = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if name.startswith(("jscc.", "temporal."))
        and parameter.grad is not None
    )
    stage_b_tsn_gradient_tensors = sum(
        parameter.grad is not None
        for parameter in model.tsn.parameters()
    )

    if (
        stage_b_scorer_gradient_l1 <= 0
        or stage_b_codec_gradient_l1 <= 0
    ):
        raise RuntimeError(
            "Stage B task gradient did not reach scorer and codec"
        )
    if stage_b_tsn_gradient_tensors != 0:
        raise RuntimeError(
            "Stage B produced TSN parameter gradients"
        )

    optimizer_a = make_optimizer(model, "A", params)
    groups_a = [
        {
            "name": group["group_name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(
                parameter.numel()
                for parameter in group["params"]
            ),
        }
        for group in optimizer_a.param_groups
    ]

    optimizer_b = make_optimizer(model, "B", params)
    groups_b = [
        {
            "name": group["group_name"],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "parameters": sum(
                parameter.numel()
                for parameter in group["params"]
            ),
        }
        for group in optimizer_b.param_groups
    ]

    if len(groups_a) != 1:
        raise RuntimeError(
            "Stage A must have exactly one optimiser group"
        )
    if groups_a[0]["name"] != "scorer":
        raise RuntimeError(
            "Stage A optimiser contains a non-scorer group"
        )
    if groups_a[0]["weight_decay"] != 0.0:
        raise RuntimeError(
            "Stage A scorer weight decay is not zero"
        )

    if [group["name"] for group in groups_b] != [
        "scorer", "codec"
    ]:
        raise RuntimeError(
            "Stage B optimiser groups are incorrect"
        )
    if groups_b[0]["weight_decay"] != 0.0:
        raise RuntimeError(
            "Stage B scorer weight decay is not zero"
        )

    model.zero_grad(set_to_none=True)
    model.eval()
    configure_stage(model, "A")

    result = {
        "uniform_path_max_abs_diff": uniform_max_abs_diff,
        "constant_score_power_max_abs_diff": (
            constant_score_error
        ),
        "small_score_scale_power_max_abs_diff": (
            small_scale_error
        ),
        "large_score_scale_power_max_abs_diff": (
            large_scale_error
        ),
        "unit_mean_max_abs_error": unit_mean_error,
        "alpha": model.allocator.alpha,
        "score_eps": model.allocator.score_eps,
        "relative_power": {
            "mean": learned_power.mean().item(),
            "std": learned_power.std(
                unbiased=False
            ).item(),
            "cv": (
                learned_power.std(unbiased=False)
                / learned_power.mean().clamp_min(1e-12)
            ).item(),
            "min": learned_power.min().item(),
            "max": learned_power.max().item(),
        },
        "all_symbols_transmitted": learned_audit[
            "all_symbols_transmitted"
        ],
        "energy_ratio_min": learned_audit[
            "energy_ratio_min"
        ],
        "energy_ratio_max": learned_audit[
            "energy_ratio_max"
        ],
        "stage_a": {
            "scorer_task_gradient_l1": (
                stage_a_scorer_gradient_l1
            ),
            "codec_gradient_tensors": (
                stage_a_codec_gradient_tensors
            ),
            "tsn_gradient_tensors": (
                stage_a_tsn_gradient_tensors
            ),
            "codec_state_changes": (
                stage_a_codec_state_changes
            ),
            "codec_training": model.jscc.training,
            "temporal_training": model.temporal.training,
            "optimizer_groups": groups_a,
        },
        "stage_b": {
            "scorer_task_gradient_l1": (
                stage_b_scorer_gradient_l1
            ),
            "codec_task_gradient_l1": (
                stage_b_codec_gradient_l1
            ),
            "tsn_gradient_tensors": (
                stage_b_tsn_gradient_tensors
            ),
            "optimizer_groups": groups_b,
        },
        "allocation_scope": (
            "spatial_within_each_encoded_frame"
        ),
    }

    print("\nStaged variance-normalised UEP audit")
    print(json.dumps(result, indent=2))
    print(
        "PASS: Stage A scorer-only; Stage B controlled joint; "
        "TSN frozen; exact energy"
    )
    return result


def save_checkpoint(path, model, epoch, metrics, metadata):
    torch.save({
        "model_state": model.state_dict(),
        "epoch": epoch,
        "validation_metrics": metrics,
        **metadata,
    }, path)


def parser():
    p = argparse.ArgumentParser(
        description="Train continuous bounded spatial UEP"
    )
    p.add_argument("--frames_root", default="datasets/UCF101Frames")
    p.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--gop_size", type=int, default=5)
    p.add_argument("--gops_per_clip", type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--channel", choices=("AWGN", "Rayleigh"), default="AWGN")
    p.add_argument("--snr_list", nargs="+", default=["13"])
    p.add_argument("--ratio_list", nargs="+", default=["1/6"])
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--tsn_head_ckpt", required=True)
    p.add_argument("--videojscc_ckpt", required=True)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--random_seed", type=int, default=1729)
    p.add_argument("--audit_only", action="store_true")
    p.add_argument("--max_iters_per_epoch", type=int, default=None)
    p.add_argument("--lambda_task", type=float, default=0.001)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--stage_a_epochs", type=int, default=10)
    p.add_argument("--scorer_lr_stage_a", type=float, default=1e-4)
    p.add_argument("--scorer_lr_stage_b", type=float, default=1e-5)
    p.add_argument("--codec_lr_stage_b", type=float, default=1e-6)
    p.add_argument("--codec_weight_decay", type=float, default=5e-4)
    p.add_argument("--score_eps", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_time", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_channel_seed", type=int, default=1042)
    p.add_argument("--out", default="./out_ta_uep_continuous_staged")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--disable_tqdm", action="store_true")
    return p.parse_args()


def train_pipeline(params):
    print(f"\nLoading UCF101 GoP dataset (N={params['gop_size']})...")
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
    model = TAVideoJSCCStagedContinuousUEP(
        c=c, channel_type=params["channel"], snr=params["snr"],
        n_frames=params["gop_size"], hidden_dim=params["hidden_dim"],
        tsn_head_ckpt=params["tsn_head_ckpt"],
        lambda_task=params["lambda_task"], lambda_recon=params["lambda_recon"],
        alpha=params["alpha"], score_eps=params["score_eps"],
        random_seed=params["random_seed"],
    ).to(device)
    baseline = load_videojscc_initialization(model, params["videojscc_ckpt"])
    if baseline["split_manifest_sha256"] != manifest_hash:
        raise RuntimeError("Baseline/training split-manifest mismatch")
    tsn_hash = model.tsn.checkpoint_metadata.get("split_manifest_sha256")
    if tsn_hash != manifest_hash:
        raise RuntimeError("TSN/training split-manifest mismatch")
    print(f"[Split audit] PASS: {manifest_hash}")
    if not 0 < params["stage_a_epochs"] < params["epochs"]:
        raise ValueError(
            "stage_a_epochs must be between 1 and epochs-1"
        )
    print(
        f"Staged variance-normalised UEP: "
        f"alpha={model.allocator.alpha:.3f} | "
        f"Stage A={params['stage_a_epochs']} epochs | "
        f"Stage B={params['epochs']-params['stage_a_epochs']} epochs"
    )
    audit = audit_staged(
        model, val_loader, device,
        params["val_channel_seed"], params,
    )
    if params["audit_only"]:
        print("Audit-only run completed; no optimizer step was taken.")
        return

    scorer_params = sum(
        p.numel() for p in model.allocator.scorer.parameters()
    )
    codec_params = sum(
        p.numel() for p in model.jscc.parameters()
    ) + sum(
        p.numel() for p in model.temporal.parameters()
    )
    frozen_tsn_params = sum(
        p.numel() for p in model.tsn.parameters()
    )
    print(
        f"Scorer params: {scorer_params:,} | "
        f"Codec/TFM params: {codec_params:,} | "
        f"Frozen TSN: {frozen_tsn_params:,}"
    )
    tag = (
        f"TAUEPStagedVarNorm_{params['channel']}_c{c}_snr{params['snr']}"
        f"_ratio{params['ratio']:.4f}_alpha{params['alpha']:.2f}"
        f"_stageA{params['stage_a_epochs']}"
        f"_{params['epochs']}ep_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}"
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
        "epoch", "stage", "scorer_lr", "codec_lr",
        "train_loss", "train_task_loss", "train_recon_loss",
        "val_loss", "val_task_loss", "val_recon_loss", "val_top1", "val_top5",
        "val_psnr", "power_mean", "power_std", "power_cv", "power_min", "power_max",
    ]
    metrics_handle = open(os.path.join(config_dir, "metrics.csv"), "w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    metrics_writer.writeheader()
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = make_optimizer(model, "A", params)

    metadata = {
        "method": "staged_variance_normalised_continuous_spatial_uep",
        "split_manifest_sha256": manifest_hash,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "baseline_split_manifest_sha256": baseline["split_manifest_sha256"],
        "tsn_split_manifest_sha256": tsn_hash,
        "all_symbols_transmitted": True,
        "actual_cbr": params["ratio"],
        "allocation_scope": "spatial_within_each_encoded_frame",
        "alpha": model.allocator.alpha,
        "power_mapping": (
            "z=(s-mean(s))/sqrt(mean((s-mean(s))^2)); "
            "p=(1+alpha*tanh(z))/mean(1+alpha*tanh(z))"
        ),
        "score_eps": model.allocator.score_eps,
        "stage_a_epochs": params["stage_a_epochs"],
        "stage_b_epochs": (
            params["epochs"] - params["stage_a_epochs"]
        ),
        "scorer_lr_stage_a": params["scorer_lr_stage_a"],
        "scorer_lr_stage_b": params["scorer_lr_stage_b"],
        "codec_lr_stage_b": params["codec_lr_stage_b"],
        "scorer_weight_decay": 0.0,
        "codec_weight_decay": params["codec_weight_decay"],
    }
    best_loss = float("inf"); best_loss_epoch = -1
    best_top1 = float("-inf"); best_top1_epoch = -1
    started = time.time(); completed_epochs = 0
    try:
        with tqdm(range(params["epochs"]), disable=params["disable_tqdm"]) as progress:
            for index in progress:
                epoch = index + 1
                stage = (
                    "A"
                    if epoch <= params["stage_a_epochs"]
                    else "B"
                )

                if epoch == params["stage_a_epochs"] + 1:
                    optimizer = make_optimizer(model, "B", params)
                    print(
                        "\nTransition to Stage B: codec/TFM unfrozen"
                    )

                scorer_lr, codec_lr = optimizer_lrs(optimizer)

                progress.set_description(
                    f"Epoch {epoch} Stage {stage}"
                )

                train = train_epoch(
                    model, optimizer, device, train_loader, stage,
                    params["max_iters_per_epoch"],
                )
                val = evaluate_epoch(
                    model, device, val_loader, params["val_channel_seed"], "learned"
                )
                completed_epochs = epoch
                metrics_writer.writerow({
                    "epoch": epoch, "stage": stage,
                    "scorer_lr": scorer_lr,
                    "codec_lr": codec_lr,
                    "train_loss": train["loss"], "train_task_loss": train["l_task"],
                    "train_recon_loss": train["l_recon"], "val_loss": val["loss"],
                    "val_task_loss": val["l_task"], "val_recon_loss": val["l_recon"],
                    "val_top1": val["top1_acc"], "val_top5": val["top5_acc"],
                    "val_psnr": val["psnr"], "power_mean": val["relative_power_mean"],
                    "power_std": val["relative_power_std"],
                    "power_cv": val["relative_power_cv"],
                    "power_min": val["relative_power_min"],
                    "power_max": val["relative_power_max"],
                }); metrics_handle.flush()
                for key, value in train.items(): writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val.items():
                    if isinstance(value, (int, float)): writer.add_scalar(f"val/{key}", value, epoch)
                progress.set_postfix(
                    loss=f"{train['loss']:.4f}", task=f"{train['l_task']:.3f}",
                    psnr=f"{val['psnr']:.2f}dB", val_acc=f"{val['top1_acc']*100:.1f}%",
                    power_cv=f"{val['relative_power_cv']:.3f}",
                )
                if epoch == params["stage_a_epochs"]:
                    save_checkpoint(
                        os.path.join(ckpt_dir, "stage_a_final.pt"),
                        model, epoch, val,
                        {**metadata, "selected_stage": "A"},
                    )
                    print(
                        f"\n  Stage-A endpoint saved: epoch={epoch} | "
                        f"Top-1={val['top1_acc']*100:.1f}%"
                    )

                if stage == "B" and val["loss"] < best_loss:
                    best_loss, best_loss_epoch = val["loss"], epoch
                    save_checkpoint(
                        os.path.join(
                            ckpt_dir, "best_joint_loss.pt"
                        ),
                        model, epoch, val,
                        {**metadata, "selected_stage": "B"},
                    )
                    print(
                        f"\n  Best Stage-B joint-loss saved: "
                        f"epoch={epoch} | loss={best_loss:.4f} | "
                        f"Top-1={val['top1_acc']*100:.1f}%"
                    )

                if stage == "B" and val["top1_acc"] > best_top1:
                    best_top1, best_top1_epoch = (
                        val["top1_acc"], epoch
                    )
                    save_checkpoint(
                        os.path.join(ckpt_dir, "best_top1.pt"),
                        model, epoch, val,
                        {**metadata, "selected_stage": "B"},
                    )
                    print(
                        f"\n  Best Stage-B Top-1 saved: epoch={epoch} | "
                        f"Top-1={best_top1*100:.1f}% | "
                        f"loss={val['loss']:.4f}"
                    )

                save_checkpoint(
                    os.path.join(ckpt_dir, "latest.pt"),
                    model, epoch, val,
                    {**metadata, "selected_stage": stage},
                )
                if params["max_time"] > 0 and (time.time() - started) / 3600 >= params["max_time"]:
                    print(f"max_time={params['max_time']} h reached; stopping")
                    break
    finally:
        metrics_handle.close(); writer.close()

    def selected(filename):
        artifact = torch.load(os.path.join(ckpt_dir, filename), map_location=device, weights_only=False)
        model.load_state_dict(artifact["model_state"], strict=True)
        return evaluate_epoch(model, device, val_loader, params["val_channel_seed"], "learned")

    stage_a_metrics = selected("stage_a_final.pt")
    joint_metrics = selected("best_joint_loss.pt")
    top1_metrics = selected("best_top1.pt")
    summary = {
        "status": "completed" if completed_epochs == params["epochs"] else "stopped",
        "method": "staged_variance_normalised_continuous_spatial_uep",
        "epochs_completed": completed_epochs,
        "stage_a_final_epoch": params["stage_a_epochs"],
        "stage_a_final_metrics": stage_a_metrics,
        "best_joint_loss_epoch": best_loss_epoch,
        "best_top1_epoch": best_top1_epoch,
        "best_joint_loss_metrics": joint_metrics,
        "best_top1_metrics": top1_metrics,
        "scorer_params": scorer_params,
        "codec_tfm_params": codec_params,
        "frozen_tsn_params": frozen_tsn_params,
        "split_manifest_sha256": manifest_hash,
        "official_test_used": False,
        "audit": audit,
        "total_hours": (time.time() - started) / 3600,
    }
    with open(os.path.join(config_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(config_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump({**params, **metadata, "c": c, "actual_device": device_string}, handle)
    print(json.dumps(summary, indent=2))
    print(f"Saved checkpoints: {ckpt_dir}")


def main():
    args = parser(); set_seed(args.seed)
    params = vars(args)
    params["snr_list"] = list(map(float, args.snr_list))
    params["ratio_list"] = [float(Fraction(value)) for value in args.ratio_list]
    for ratio in params["ratio_list"]:
        for snr in params["snr_list"]:
            run = dict(params); run["ratio"] = ratio; run["snr"] = snr
            train_pipeline(run)


if __name__ == "__main__":
    main()
