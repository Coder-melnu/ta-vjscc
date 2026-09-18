# -*- coding: utf-8 -*-
"""No-selector attribution control for task-supervised VideoJSCC fine-tuning.

The clean VideoJSCC+TFM checkpoint is warm-started and fine-tuned for the full
run with L = lambda_recon*MSE + lambda_task*cross_entropy.  The corrected TSN
is frozen and held in eval mode, but gradients pass through it to the codec.
No selector implementation is imported or instantiated by the model.

The trainer uses only the group-aware internal validation split for model
selection and keeps independent best-joint-loss and best-Top-1 checkpoints.
The official UCF101 test set is intentionally unavailable here.
"""

import argparse
import csv
import hashlib
import json
import os
import time
from contextlib import contextmanager
from fractions import Fraction

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_dataloaders
from model import ratio2filtersize
from model.videojscc_no_selector_task import NoSelectorVideoJSCC
from utils import set_seed


@contextmanager
def fixed_rng(device, seed):
    """Fix validation channel noise without perturbing the training RNG."""
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_videojscc_initialization(model, checkpoint_path):
    """Strictly load jscc.* and temporal.* from a trusted clean checkpoint."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"VideoJSCC checkpoint not found: {checkpoint_path}")
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest_hash = None
    if isinstance(artifact, dict):
        manifest_hash = artifact.get("split_manifest_sha256")
        if manifest_hash is None and isinstance(artifact.get("config"), dict):
            manifest_hash = artifact["config"].get("split_manifest_sha256")
        state = artifact.get("model_state", artifact.get("model_state_dict", artifact))
    else:
        state = artifact
    jscc_state = {key[5:]: value for key, value in state.items() if key.startswith("jscc.")}
    temporal_state = {
        key[9:]: value for key, value in state.items() if key.startswith("temporal.")
    }
    if not jscc_state or not temporal_state:
        raise KeyError("Checkpoint must contain both jscc.* and temporal.* weights")
    model.jscc.load_state_dict(jscc_state, strict=True)
    model.temporal.load_state_dict(temporal_state, strict=True)
    print(f"[VideoJSCC init] Loaded: {checkpoint_path}")
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split_manifest_sha256": manifest_hash,
    }


def train_epoch(model, optimizer, device, loader, max_iters=None):
    model.train()
    model.tsn.eval()
    totals = {"loss": 0.0, "l_task": 0.0, "l_recon": 0.0, "psnr": 0.0}
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
def evaluate_epoch(model, device, loader, channel_seed):
    model.eval()
    totals = {"loss": 0.0, "l_task": 0.0, "l_recon": 0.0, "psnr": 0.0}
    correct1 = correct5 = samples = iterations = 0
    with fixed_rng(device, channel_seed):
        for gops, labels in loader:
            gops, labels = gops.to(device), labels.to(device)
            _, info, _, logits = model.joint_loss(gops, labels)
            for key in totals:
                totals[key] += info[key]
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum().item()
            correct5 += top5.eq(labels[:, None]).any(dim=1).sum().item()
            samples += labels.numel()
            iterations += 1
    if samples == 0 or iterations == 0:
        raise RuntimeError("Validation loader produced zero samples")
    metrics = {key: value / iterations for key, value in totals.items()}
    metrics.update({
        "top1_acc": correct1 / samples,
        "top5_acc": correct5 / samples,
        "samples": samples,
    })
    return metrics


def audit_no_selector(model, loader, device, channel_seed):
    """Prove absence of selector path and correctness of frozen-task gradients."""
    forbidden_modules = [name for name, _ in model.named_modules() if "selector" in name.lower()]
    forbidden_params = [name for name, _ in model.named_parameters() if "selector" in name.lower()]
    if forbidden_modules or forbidden_params:
        raise RuntimeError(
            f"Selector contamination: modules={forbidden_modules}, params={forbidden_params}"
        )
    if any(parameter.requires_grad for parameter in model.tsn.parameters()):
        raise RuntimeError("TSN contains trainable parameters")

    model.eval()
    gops, labels = next(iter(loader))
    gops, labels = gops.to(device), labels.to(device)
    with fixed_rng(device, channel_seed):
        reconstructed = model(gops)
    with fixed_rng(device, channel_seed):
        batch, frames, channels, height, width = gops.shape
        direct = model.jscc(gops.reshape(batch * frames, channels, height, width))
        direct = model.temporal(direct.reshape(batch, frames, channels, height, width))
    path_max_abs_diff = (reconstructed - direct).abs().max().item()
    if path_max_abs_diff != 0.0:
        raise RuntimeError(f"Plain-path audit failed: max abs diff={path_max_abs_diff}")

    model.zero_grad(set_to_none=True)
    logits = model.tsn(reconstructed)
    task_loss = F.cross_entropy(logits, labels)
    task_loss.backward()
    codec_task_grad = sum(
        float(parameter.grad.abs().sum())
        for name, parameter in model.named_parameters()
        if not name.startswith("tsn.") and parameter.grad is not None
    )
    tsn_grad_tensors = sum(
        parameter.grad is not None for parameter in model.tsn.parameters()
    )
    model.zero_grad(set_to_none=True)
    if codec_task_grad <= 0:
        raise RuntimeError("Task loss did not produce a codec/TFM gradient")
    if tsn_grad_tensors != 0:
        raise RuntimeError("Frozen TSN accumulated gradients")

    result = {
        "selector_modules": 0,
        "selector_parameters": 0,
        "plain_path_max_abs_diff": path_max_abs_diff,
        "codec_task_gradient_l1": codec_task_grad,
        "tsn_gradient_tensors": tsn_grad_tensors,
        "tsn_training": model.tsn.training,
    }
    print("\nNo-selector control audit")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("  PASS: plain VideoJSCC path; task gradient reaches codec; TSN frozen")
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
        description="Fine-tune plain VideoJSCC with frozen task loss; no selector"
    )
    p.add_argument("--frames_root", default="datasets/UCF101Frames")
    p.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--gop_size", type=int, default=5)
    p.add_argument("--gops_per_clip", type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--channel", choices=["AWGN", "Rayleigh"], default="AWGN")
    p.add_argument("--snr_list", nargs="+", default=["13"])
    p.add_argument("--ratio_list", nargs="+", default=["1/6"])
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--tsn_head_ckpt", default="downstream/action_recognition/weights/tsn_ucf101_head_locked_split_best.pt")
    p.add_argument("--videojscc_ckpt", required=True)
    p.add_argument("--audit_only", action="store_true")
    p.add_argument("--max_iters_per_epoch", type=int, default=None)
    p.add_argument("--lambda_task", type=float, default=0.001)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--init_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--step_size", type=int, default=25)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--max_time", type=float, default=0.0,
                   help="Hours; 0 disables the wall-clock limit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_channel_seed", type=int, default=1042)
    p.add_argument("--out", default="./out_videojscc_no_selector")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--disable_tqdm", action="store_true")
    return p.parse_args()


def train_pipeline(params):
    print(f"\nLoading UCF101 GoP dataset (N={params['gop_size']})...")
    train_loader, val_loader, split_manifest, split_hash = build_train_val_dataloaders(
        frames_root=params["frames_root"], annotation_path=params["annotation_path"],
        image_size=params["image_size"], batch_size=params["batch_size"],
        num_workers=params["num_workers"], gop_size=params["gop_size"],
        gops_per_clip=params["gops_per_clip"], val_fraction=params["val_fraction"],
       # seed=params["seed"],
        seed=params["split_seed"], enforce_locked_counts=True,
    )
    c = ratio2filtersize(
        torch.empty(3, params["image_size"], params["image_size"]), params["ratio"]
    )
    device_string = params["device"] if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    print(
        f"SNR={params['snr']} dB | ratio={params['ratio']:.4f} | c={c} | "
        f"channel={params['channel']}"
    )
    print(
        f"No selector | lambda_task={params['lambda_task']} | "
        f"lambda_recon={params['lambda_recon']} | lr={params['init_lr']} | "
        f"epochs={params['epochs']}"
    )
    model = NoSelectorVideoJSCC(
        c=c, channel_type=params["channel"], snr=params["snr"],
        n_frames=params["gop_size"], hidden_dim=params["hidden_dim"],
        tsn_head_ckpt=params["tsn_head_ckpt"],
        lambda_task=params["lambda_task"], lambda_recon=params["lambda_recon"],
        device=device_string,
    ).to(device)
    baseline = load_videojscc_initialization(model, params["videojscc_ckpt"])
    if baseline["split_manifest_sha256"] is None:
        raise RuntimeError("Baseline checkpoint lacks split_manifest_sha256")
    if baseline["split_manifest_sha256"] != split_hash:
        raise RuntimeError("Baseline/training split-manifest mismatch")
    tsn_hash = model.tsn.checkpoint_metadata.get("split_manifest_sha256")
    if tsn_hash != split_hash:
        raise RuntimeError("TSN/training split-manifest mismatch")
    print(f"[Split audit] PASS: {split_hash}")
    audit = audit_no_selector(model, val_loader, device, params["val_channel_seed"])
    if params["audit_only"]:
        print("\nAudit-only run completed; no optimizer step was taken.")
        return

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params (codec+TFM): {trainable:,} | Frozen TSN: {frozen:,}")
    tag = (
        f"VideoJSCCNoSelector_{params['channel']}_c{c}_snr{params['snr']}"
        f"_ratio{params['ratio']:.4f}_joint{params['epochs']}ep"
        f"_lt{params['lambda_task']}_lr{params['lambda_recon']}"
        f"_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}"
    )
    checkpoint_dir = os.path.join(params["out"], "checkpoints", tag)
    config_dir = os.path.join(params["out"], "configs", tag)
    log_dir = os.path.join(params["out"], "logs", tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "split_manifest.json"), "w") as handle:
        json.dump(split_manifest, handle, indent=2)
    with open(os.path.join(config_dir, "split_manifest.sha256"), "w") as handle:
        handle.write(split_hash + "\n")

    fields = [
        "epoch", "lr", "train_loss", "train_task_loss", "train_recon_loss",
        "val_loss", "val_task_loss", "val_recon_loss", "val_top1", "val_top5",
        "val_psnr",
    ]
    metrics_handle = open(os.path.join(config_dir, "metrics.csv"), "w", newline="")
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=fields)
    metrics_writer.writeheader()
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=params["init_lr"], weight_decay=params["weight_decay"],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params["step_size"], gamma=params["gamma"]
    )
    selection_metadata = {
        "control": "no_selector",
        "split_manifest_sha256": split_hash,
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "baseline_split_manifest_sha256": baseline["split_manifest_sha256"],
        "tsn_split_manifest_sha256": tsn_hash,
    }
    best_loss = float("inf")
    best_loss_epoch = -1
    best_top1 = float("-inf")
    best_top1_epoch = -1
    started = time.time()
    completed_epochs = 0

    try:
        with tqdm(range(params["epochs"]), disable=params["disable_tqdm"]) as progress:
            for zero_epoch in progress:
                epoch = zero_epoch + 1
                progress.set_description(f"Epoch {epoch}")
                train = train_epoch(
                    model, optimizer, device, train_loader,
                    params["max_iters_per_epoch"],
                )
                val = evaluate_epoch(model, device, val_loader, params["val_channel_seed"])
                completed_epochs = epoch
                row = {
                    "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": train["loss"], "train_task_loss": train["l_task"],
                    "train_recon_loss": train["l_recon"], "val_loss": val["loss"],
                    "val_task_loss": val["l_task"], "val_recon_loss": val["l_recon"],
                    "val_top1": val["top1_acc"], "val_top5": val["top5_acc"],
                    "val_psnr": val["psnr"],
                }
                metrics_writer.writerow(row); metrics_handle.flush()
                for key, value in train.items(): writer.add_scalar(f"train/{key}", value, epoch)
                for key, value in val.items(): writer.add_scalar(f"val/{key}", value, epoch)
                progress.set_postfix(
                    loss=f"{train['loss']:.4f}", l_task=f"{train['l_task']:.4f}",
                    l_recon=f"{train['l_recon']:.4f}", psnr=f"{val['psnr']:.2f}dB",
                    val_acc=f"{val['top1_acc']*100:.1f}%",
                    val_top5=f"{val['top5_acc']*100:.1f}%",
                )
                if val["loss"] < best_loss:
                    best_loss, best_loss_epoch = val["loss"], epoch
                    save_checkpoint(
                        os.path.join(checkpoint_dir, "best_joint_loss.pt"),
                        model, epoch, val, selection_metadata,
                    )
                    print(
                        f"\n  Best joint-loss saved: epoch={epoch} | "
                        f"loss={val['loss']:.4f} | Top-1={val['top1_acc']*100:.1f}%"
                    )
                if val["top1_acc"] > best_top1:
                    best_top1, best_top1_epoch = val["top1_acc"], epoch
                    save_checkpoint(
                        os.path.join(checkpoint_dir, "best_top1.pt"),
                        model, epoch, val, selection_metadata,
                    )
                    print(
                        f"\n  Best Top-1 saved: epoch={epoch} | "
                        f"Top-1={val['top1_acc']*100:.1f}% | loss={val['loss']:.4f}"
                    )
                save_checkpoint(
                    os.path.join(checkpoint_dir, "latest.pt"),
                    model, epoch, val, selection_metadata,
                )
                scheduler.step()
                if params["max_time"] > 0 and (time.time() - started) / 3600 >= params["max_time"]:
                    print(f"\nmax_time={params['max_time']} h reached; stopping.")
                    break
    finally:
        metrics_handle.close()
        writer.close()

    def load_selected(filename):
        artifact = torch.load(
            os.path.join(checkpoint_dir, filename), map_location=device,
            weights_only=False,
        )
        model.load_state_dict(artifact["model_state"], strict=True)
        return evaluate_epoch(model, device, val_loader, params["val_channel_seed"])

    best_loss_metrics = load_selected("best_joint_loss.pt")
    best_top1_metrics = load_selected("best_top1.pt")
    summary = {
        "status": "completed" if completed_epochs == params["epochs"] else "stopped",
        "control": "no_selector",
        "epochs_completed": completed_epochs,
        "best_joint_loss_epoch": best_loss_epoch,
        "best_top1_epoch": best_top1_epoch,
        "best_joint_loss_metrics": best_loss_metrics,
        "best_top1_metrics": best_top1_metrics,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "split_manifest_sha256": split_hash,
        "official_test_used": False,
        "audit": audit,
        "total_hours": (time.time() - started) / 3600,
    }
    with open(os.path.join(config_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(config_dir, "config.yaml"), "w") as handle:
        yaml.safe_dump({**params, "c": c, "actual_device": device_string}, handle)
    print(json.dumps(summary, indent=2))
    print(f"Saved checkpoints: {checkpoint_dir}")


def main():
    args = parser()
    set_seed(args.seed)
    params = vars(args)
    params["snr_list"] = list(map(float, args.snr_list))
    params["ratio_list"] = [float(Fraction(value)) for value in args.ratio_list]
    for ratio in params["ratio_list"]:
        for snr in params["snr_list"]:
            run = dict(params)
            run["ratio"] = ratio
            run["snr"] = snr
            train_pipeline(run)


if __name__ == "__main__":
    main()
