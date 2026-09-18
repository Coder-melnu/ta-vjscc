"""Audited clean VideoJSCC trainer: group-aware validation, TFM v3, no test use."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from pytorch_msssim import ms_ssim, ssim
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_dataloaders
from model.encoder import ratio2filtersize
from model.video_jscc import VideoJSCC
from utils import set_seed


def parser():
    p = argparse.ArgumentParser(fromfile_prefix_chars="@")
    p.add_argument("--frames_root", default="datasets/UCF101Frames")
    p.add_argument("--annotation_path", default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--gop_size", type=int, default=5)
    p.add_argument("--gops_per_clip", type=int, default=1)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--channel", choices=("AWGN", "Rayleigh"), default="AWGN")
    p.add_argument("--snr_list", nargs="+", default=("13",))
    p.add_argument("--ratio_list", nargs="+", default=("1/6",))
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--init_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--step_size", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_channel_seed", type=int, default=1042)
    p.add_argument("--snapshot_every", type=int, default=10)
    p.add_argument("--max_snapshot_items", type=int, default=2)
    p.add_argument("--max_time", type=float, default=0.0, help="Hours; 0 disables")
    p.add_argument("--out", default="./out_video_clean")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--disable_tqdm", action="store_true")
    p.add_argument("--allow_matrix", action="store_true")
    p.add_argument("--no_locked_count_check", action="store_true")
    return p.parse_args()


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info(device):
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_commit = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": git_commit,
    }


def channel_rng(seed, device):
    devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    context = torch.random.fork_rng(devices=devices)
    context.__enter__()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return context


def train_epoch(model, optimizer, loader, device, audit_gradients=False):
    model.train()
    total, count, grad_report = 0.0, 0, None
    for batch_index, (gops, _) in enumerate(loader):
        gops = gops.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(gops)
        loss = model.loss(prediction, gops)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}")
        loss.backward()
        if audit_gradients and batch_index == 0:
            grad_report = {}
            for name, module in {
                "encoder": model.jscc.encoder,
                "decoder": model.jscc.decoder,
                "tfm_v3": model.temporal,
            }.items():
                values = [p.grad.detach().norm().item() for p in module.parameters() if p.grad is not None]
                grad_report[name] = {"tensors": len(values), "norm_sum": float(sum(values))}
                if not values or not np.isfinite(values).all() or sum(values) == 0:
                    raise RuntimeError(f"Gradient audit failed for {name}: {grad_report[name]}")
        optimizer.step()
        total += loss.item() * gops.size(0)
        count += gops.size(0)
    if not count:
        raise RuntimeError("Empty training loader")
    return total / count, grad_report


@torch.no_grad()
def evaluate(model, loader, device, seed):
    model.eval()
    sums = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "ms_ssim_3scale": 0.0}
    count = 0
    context = channel_rng(seed, device)
    try:
        for gops, _ in loader:
            gops = gops.to(device, non_blocking=True)
            prediction = model(gops)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite validation reconstruction")
            clipped = prediction.clamp(0, 1)
            flat_gt = gops.flatten(0, 1)
            flat_pred = clipped.flatten(0, 1)
            mse = torch.mean((prediction - gops) ** 2)
            metric_mse = torch.mean((flat_pred - flat_gt) ** 2, dim=(1, 2, 3))
            batch = gops.size(0)
            sums["loss"] += mse.item() * batch
            sums["psnr"] += (-10.0 * torch.log10(metric_mse.clamp_min(1e-12))).mean().item() * batch
            sums["ssim"] += ssim(flat_pred, flat_gt, data_range=1.0, size_average=True).item() * batch
            sums["ms_ssim_3scale"] += ms_ssim(
                flat_pred, flat_gt, data_range=1.0, size_average=True,
                win_size=7, weights=(0.3, 0.3, 0.4),
            ).item() * batch
            count += batch
    finally:
        context.__exit__(None, None, None)
    if not count:
        raise RuntimeError("Empty validation loader")
    return {key: value / count for key, value in sums.items()}


@torch.no_grad()
def save_snapshot(model, fixed_gops, device, seed, path):
    model.eval()
    gops = fixed_gops.to(device)
    context = channel_rng(seed, device)
    try:
        before = model.forward_no_temporal(gops).clamp(0, 1)
    finally:
        context.__exit__(None, None, None)
    context = channel_rng(seed, device)
    try:
        after = model(gops).clamp(0, 1)
    finally:
        context.__exit__(None, None, None)
    # For each selected video: GT row, pre-TFM row, post-TFM row.
    images = []
    for item in range(gops.size(0)):
        images.extend(gops[item].cpu())
        images.extend(before[item].cpu())
        images.extend(after[item].cpu())
    grid = make_grid(images, nrow=gops.size(1), padding=2)
    save_image(grid, path)


def checkpoint(model, optimizer, scheduler, epoch, best, config, manifest_hash, metrics):
    return {
        "format_version": 1, "epoch": epoch, "best_val_loss": best,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "config": config,
        "split_manifest_sha256": manifest_hash, "metrics": metrics,
        "rng_state": {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def run(config):
    set_seed(config["seed"])
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=config["frames_root"], annotation_path=config["annotation_path"],
        image_size=config["image_size"], batch_size=config["batch_size"],
        num_workers=config["num_workers"], gop_size=config["gop_size"],
        gops_per_clip=config["gops_per_clip"], val_fraction=config["val_fraction"],
        seed=config["seed"], enforce_locked_counts=not config["no_locked_count_check"],
    )
    # ratio2filtersize instantiates a temporary encoder; reseed afterwards so
    # the actual model initialization is identical across comparable runs.
    c = ratio2filtersize(torch.empty(3, config["image_size"], config["image_size"]), config["ratio"])
    set_seed(config["seed"])
    model = VideoJSCC(
        c=c, channel_type=config["channel"], snr=config["snr"],
        n_frames=config["gop_size"], hidden_dim=config["hidden_dim"],
    ).to(device)
    config.update({"c": c, "actual_device": str(device), "actual_channel": model.get_channel()})

    run_name = f"VideoJSCC_{config['channel']}_c{c}_snr{config['snr']:g}_ratio{config['ratio']:.4f}_seed{config['seed']}"
    run_dir = Path(config["out"]) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    for name in ("checkpoints", "snapshots", "tensorboard"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    (run_dir / "environment.json").write_text(json.dumps(environment_info(device), indent=2))
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "split_manifest.sha256").write_text(manifest_hash + "\n")
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    optimizer = optim.Adam(model.parameters(), lr=config["init_lr"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, config["step_size"], gamma=config["gamma"])
    fixed = torch.stack([val_loader.dataset[i][0] for i in range(min(config["max_snapshot_items"], len(val_loader.dataset)))])
    fields = ["epoch", "train_loss", "val_loss", "val_psnr_db", "val_ssim", "val_ms_ssim_3scale", "lr", "seconds"]
    metrics_file = run_dir / "metrics.csv"
    best, best_epoch, history, start_all = float("inf"), -1, [], time.time()
    grad_report = None
    with metrics_file.open("w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        progress = tqdm(range(1, config["epochs"] + 1), disable=config["disable_tqdm"])
        for epoch in progress:
            start = time.time()
            train_loss, report = train_epoch(model, optimizer, train_loader, device, audit_gradients=epoch == 1)
            grad_report = report or grad_report
            values = evaluate(model, val_loader, device, config["val_channel_seed"])
            row = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": values["loss"],
                "val_psnr_db": values["psnr"], "val_ssim": values["ssim"],
                "val_ms_ssim_3scale": values["ms_ssim_3scale"],
                "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start,
            }
            csv_writer.writerow(row); handle.flush(); history.append(row)
            for key, value in row.items():
                if key not in ("epoch", "seconds"):
                    writer.add_scalar(key, value, epoch)
            is_best = values["loss"] < best
            if is_best:
                best, best_epoch = values["loss"], epoch
            # Store the scheduler state for the *next* epoch.  This makes a
            # checkpoint a valid resume point as well as an evaluation record.
            scheduler.step()
            state = checkpoint(model, optimizer, scheduler, epoch, best, config, manifest_hash, row)
            atomic_torch_save(state, run_dir / "checkpoints" / "latest.pt")
            if is_best:
                atomic_torch_save(state, run_dir / "checkpoints" / "best.pt")
                save_snapshot(model, fixed, device, config["val_channel_seed"], run_dir / "snapshots" / "best.png")
            if epoch == 1 or epoch % config["snapshot_every"] == 0:
                save_snapshot(model, fixed, device, config["val_channel_seed"], run_dir / "snapshots" / f"epoch_{epoch:03d}.png")
            progress.set_postfix(loss=f"{values['loss']:.5f}", psnr=f"{values['psnr']:.2f}", ssim=f"{values['ssim']:.4f}")
            if config["max_time"] > 0 and time.time() - start_all >= config["max_time"] * 3600:
                break
    (run_dir / "gradient_audit.json").write_text(json.dumps(grad_report, indent=2))

    # Reload and verify the exact selected artifact, then save final metrics/snapshot.
    selected = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    if selected["split_manifest_sha256"] != manifest_hash:
        raise RuntimeError("Best checkpoint split hash mismatch")
    model.load_state_dict(selected["model_state"], strict=True)
    reloaded = evaluate(model, val_loader, device, config["val_channel_seed"])
    if abs(reloaded["loss"] - selected["metrics"]["val_loss"]) > 1e-10:
        raise RuntimeError("Best-checkpoint reload metric mismatch")
    save_snapshot(model, fixed, device, config["val_channel_seed"], run_dir / "snapshots" / "best_reloaded.png")
    summary = {
        "status": "completed", "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_reloaded": reloaded, "total_hours": (time.time() - start_all) / 3600,
        "checkpoint_sha256": sha256(run_dir / "checkpoints" / "best.pt"),
        "split_manifest_sha256": manifest_hash, "official_test_used": False,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    writer.close()
    print(json.dumps(summary, indent=2))


def main():
    args = parser()
    snrs = [float(x) for x in args.snr_list]
    ratios = [float(Fraction(x)) for x in args.ratio_list]
    if len(snrs) * len(ratios) != 1 and not args.allow_matrix:
        raise SystemExit("Pilot guard: provide one SNR and one ratio, or explicitly pass --allow_matrix")
    base = vars(args)
    for ratio in ratios:
        for snr_value in snrs:
            run({**base, "snr": snr_value, "ratio": ratio})


if __name__ == "__main__":
    main()
