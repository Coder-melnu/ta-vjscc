"""Locked evaluation of native-rate and fine-tune-only VideoJSCC controls.

Evaluates one reconstruction pass with both the locked TSN and frozen-BatchNorm
R(2+1)D-18 recognizers. Supports the group-aware validation split and the
official UCF101 split-1 test set. No optimizer is created and all parameters
are checked for mutation.
"""

import argparse
import csv
import hashlib
import json
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
from downstream.action_recognition.models.r2plus1d_recognizer import (
    R2Plus1DRecognizer,
    load_r2plus1d_checkpoint,
    normalization_tensors,
)
from downstream.action_recognition.models.tsn_recognizer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TSNModel,
    load_tsn_checkpoint,
)
from model.video_jscc import VideoJSCC
from model.videojscc_no_selector_task import NoSelectorVideoJSCC


LOCKED_VALIDATION_MANIFEST = (
    "ea27a8557ef1e8f658f63c8f86f33721d6f94a1e58c657fa0a02c93e94befb7c"
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", choices=("clean", "no_selector"), required=True)
    p.add_argument("--videojscc_ckpt", required=True)
    p.add_argument(
        "--tsn_ckpt",
        default="downstream/action_recognition/weights/tsn_ucf101_head_locked_split_best.pt",
    )
    p.add_argument(
        "--r2plus1d_ckpt",
        default="downstream/action_recognition/weights/r2plus1d_ucf101_layer4_locked_split_best.pt",
    )
    p.add_argument("--frames_root", default="datasets/UCF101Frames")
    p.add_argument(
        "--annotation_path",
        default="datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist",
    )
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
    p.add_argument("--c", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="./out_locked_controls")
    p.add_argument("--smoke_batches", type=int, default=0)
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_locked_protocol(args):
    actual = (
        args.image_size,
        args.gop_size,
        args.gops_per_clip,
        args.split_seed,
        args.test_gop_seed,
        args.channel_seed,
    )
    expected = (128, 5, 1, 42, 44, 1042)
    if actual != expected:
        raise ValueError(f"Locked protocol requires {expected}; received {actual}")


def load_reconstruction_model(args, device):
    checkpoint = torch.load(args.videojscc_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise KeyError("Checkpoint must contain model_state")

    if args.model_type == "clean":
        if "config" not in checkpoint:
            raise KeyError("Clean checkpoint must contain config")
        config = checkpoint["config"]
        c = int(config["c"])
        channel = str(config["channel"])
        snr = float(config["snr"])
        gop_size = int(config["gop_size"])
        hidden_dim = int(config["hidden_dim"])
        if args.c is not None and args.c != c:
            raise ValueError(f"Requested c={args.c}, but checkpoint records c={c}")
        model = VideoJSCC(
            c=c,
            channel_type=channel,
            snr=snr,
            n_frames=gop_size,
            hidden_dim=hidden_dim,
        )
        expected_control = "native_c4_no_selection" if c == 4 else "clean_no_selection"
    else:
        c = 8 if args.c is None else args.c
        channel = args.channel
        snr = args.snr
        gop_size = args.gop_size
        hidden_dim = args.hidden_dim
        if checkpoint.get("control") != "no_selector":
            raise RuntimeError("Checkpoint does not certify control=no_selector")
        model = NoSelectorVideoJSCC(
            c=c,
            channel_type=channel,
            snr=snr,
            n_frames=gop_size,
            hidden_dim=hidden_dim,
            tsn_head_ckpt=args.tsn_ckpt,
            lambda_task=0.001,
            lambda_recon=1.0,
            device=str(device),
        )
        expected_control = "fine_tune_only_no_selector"

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    source_manifest = checkpoint.get("split_manifest_sha256")
    if source_manifest != LOCKED_VALIDATION_MANIFEST:
        raise RuntimeError(
            "Reconstruction checkpoint does not match the locked training/validation manifest"
        )
    metadata = {
        "control": expected_control,
        "c": c,
        "nominal_cbr": c / 48.0,
        "actual_cbr": c / 48.0,
        "channel": channel,
        "snr_db": snr,
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "checkpoint_sha256": sha256_file(args.videojscc_ckpt),
        "checkpoint_split_manifest_sha256": source_manifest,
    }
    return model, checkpoint, metadata


def load_recognizers(args, device):
    tsn = TSNModel(pretrained=True, num_classes=101)
    tsn_metadata = load_tsn_checkpoint(tsn, args.tsn_ckpt)
    if tsn_metadata.get("split_manifest_sha256") != LOCKED_VALIDATION_MANIFEST:
        raise RuntimeError("TSN checkpoint does not match the locked manifest")
    tsn.to(device).eval()
    for parameter in tsn.parameters():
        parameter.requires_grad_(False)

    r2plus1d = R2Plus1DRecognizer(pretrained=False, num_classes=101)
    r2_metadata = load_r2plus1d_checkpoint(r2plus1d, args.r2plus1d_ckpt)
    if r2_metadata.get("split_manifest_sha256") != LOCKED_VALIDATION_MANIFEST:
        raise RuntimeError("R(2+1)D checkpoint does not match the locked manifest")
    if r2_metadata.get("batchnorm_frozen") is not True:
        raise RuntimeError("R(2+1)D checkpoint does not certify frozen BatchNorm")
    r2plus1d.to(device).eval()
    r2plus1d.freeze_batchnorm()
    for parameter in r2plus1d.parameters():
        parameter.requires_grad_(False)
    return tsn, tsn_metadata, r2plus1d, r2_metadata


def snapshot(module):
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def assert_unchanged(module, before, label):
    changed = [
        name
        for name, value in module.state_dict().items()
        if not torch.equal(before[name], value.detach().cpu())
    ]
    if changed:
        raise RuntimeError(f"{label} changed during evaluation: {changed[:5]}")


@torch.no_grad()
def evaluate(model, tsn, r2plus1d, loader, device, channel_seed, smoke_batches):
    rows = []
    tsn_mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    tsn_std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    r2_mean, r2_std = normalization_tensors(device)
    clip_id = 0
    with deterministic_channel(channel_seed, device):
        for batch_index, (gops, labels) in enumerate(tqdm(loader, desc="locked controls")):
            if smoke_batches and batch_index >= smoke_batches:
                break
            gops, labels = gops.to(device), labels.to(device)
            reconstructed_raw = model(gops)
            reconstructed = reconstructed_raw.clamp(0, 1)

            batch, frames = gops.shape[:2]
            flat_gt = gops.flatten(0, 1)
            flat_pred = reconstructed.flatten(0, 1)
            frame_mse = (flat_pred - flat_gt).square().mean(dim=(1, 2, 3))
            frame_psnr = -10.0 * torch.log10(frame_mse.clamp_min(1e-12))
            frame_ssim = ssim(flat_pred, flat_gt, data_range=1.0, size_average=False)
            frame_ms_ssim = ms_ssim(
                flat_pred,
                flat_gt,
                data_range=1.0,
                size_average=False,
                win_size=7,
                weights=(0.3, 0.3, 0.4),
            )
            raw_mse = (reconstructed_raw - gops).square().mean(dim=(1, 2, 3, 4))

            tsn_recon = tsn((reconstructed - tsn_mean) / tsn_std)
            tsn_clean = tsn((gops - tsn_mean) / tsn_std)
            r2_recon = r2plus1d((reconstructed - r2_mean) / r2_std)
            r2_clean = r2plus1d((gops - r2_mean) / r2_std)

            predictions = {}
            for name, logits in (
                ("tsn_reconstructed", tsn_recon),
                ("tsn_clean", tsn_clean),
                ("r2plus1d_reconstructed", r2_recon),
                ("r2plus1d_clean", r2_clean),
            ):
                predictions[name] = logits.topk(5, dim=1).indices

            for item in range(batch):
                start, end = item * frames, (item + 1) * frames
                row = {
                    "clip_id": clip_id,
                    "label": int(labels[item]),
                    "raw_reconstruction_mse": float(raw_mse[item]),
                    "psnr_db": float(frame_psnr[start:end].mean()),
                    "ssim": float(frame_ssim[start:end].mean()),
                    "ms_ssim_3scale": float(frame_ms_ssim[start:end].mean()),
                }
                for name, top5 in predictions.items():
                    row[f"{name}_prediction"] = int(top5[item, 0])
                    row[f"{name}_top1_correct"] = int(top5[item, 0] == labels[item])
                    row[f"{name}_top5_correct"] = int(
                        top5[item].eq(labels[item]).any()
                    )
                rows.append(row)
                clip_id += 1
    return rows


def aggregate(rows):
    if not rows:
        raise RuntimeError("Evaluator produced no rows")
    count = len(rows)
    summary = {"clips": count}
    for field in (
        "raw_reconstruction_mse",
        "psnr_db",
        "ssim",
        "ms_ssim_3scale",
    ):
        summary[field] = sum(row[field] for row in rows) / count
    for evaluator in ("tsn", "r2plus1d"):
        for source in ("reconstructed", "clean"):
            prefix = f"{evaluator}_{source}"
            summary[f"{prefix}_top1_percent"] = (
                100.0 * sum(row[f"{prefix}_top1_correct"] for row in rows) / count
            )
            summary[f"{prefix}_top5_percent"] = (
                100.0 * sum(row[f"{prefix}_top5_correct"] for row in rows) / count
            )
        clean_top1 = summary[f"{evaluator}_clean_top1_percent"]
        summary[f"{evaluator}_top1_retention_percent"] = (
            100.0 * summary[f"{evaluator}_reconstructed_top1_percent"] / clean_top1
            if clean_top1 else None
        )
    return summary


def main():
    args = parser()
    require_locked_protocol(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader, manifest, manifest_hash = build_loader(args)
    expected_clips = 1128 if args.split == "validation" else 3783
    if len(loader.dataset) != expected_clips:
        raise RuntimeError(
            f"Expected {expected_clips} {args.split} clips, found {len(loader.dataset)}"
        )
    if args.split == "validation" and manifest_hash != LOCKED_VALIDATION_MANIFEST:
        raise RuntimeError(f"Locked validation manifest mismatch: {manifest_hash}")

    model, checkpoint, model_metadata = load_reconstruction_model(args, device)
    tsn, tsn_metadata, r2plus1d, r2_metadata = load_recognizers(args, device)
    before = {
        "model": snapshot(model),
        "tsn": snapshot(tsn),
        "r2plus1d": snapshot(r2plus1d),
    }
    rows = evaluate(
        model,
        tsn,
        r2plus1d,
        loader,
        device,
        args.channel_seed,
        args.smoke_batches,
    )
    assert_unchanged(model, before["model"], "Reconstruction model")
    assert_unchanged(tsn, before["tsn"], "TSN")
    assert_unchanged(r2plus1d, before["r2plus1d"], "R(2+1)D")

    if not args.smoke_batches and len(rows) != expected_clips:
        raise RuntimeError(f"Expected {expected_clips} output rows, found {len(rows)}")

    run_name = (
        f"{model_metadata['control']}_{args.split}_{model_metadata['channel']}"
        f"_c{model_metadata['c']}_snr{model_metadata['snr_db']:g}"
        f"_channel_seed{args.channel_seed}"
    )
    if args.smoke_batches:
        run_name += f"_SMOKE{args.smoke_batches}"
    output = Path(args.out) / run_name
    output.mkdir(parents=True, exist_ok=False)

    with (output / "per_clip_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "split_manifest.sha256").write_text(manifest_hash + "\n")

    evaluation_config = {
        **vars(args),
        **model_metadata,
        "actual_device": str(device),
        "evaluation_split_manifest_sha256": manifest_hash,
        "locked_validation_manifest_sha256": LOCKED_VALIDATION_MANIFEST,
        "tsn_checkpoint_sha256": sha256_file(args.tsn_ckpt),
        "r2plus1d_checkpoint_sha256": sha256_file(args.r2plus1d_ckpt),
        "tsn_metadata": tsn_metadata,
        "r2plus1d_metadata": r2_metadata,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "parameter_updates": 0,
        "official_test_used": args.split == "test",
        "diagnostic_smoke_test": bool(args.smoke_batches),
    }
    summary = {
        **aggregate(rows),
        "control": model_metadata["control"],
        "split": args.split,
        "channel_seed": args.channel_seed,
        "evaluation_split_manifest_sha256": manifest_hash,
        "parameter_updates": 0,
        "official_test_used": args.split == "test",
        "diagnostic_smoke_test": bool(args.smoke_batches),
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, default=str)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
