"""Authoritative evaluation-only VideoJSCC + corrected TSN pipeline.

No optimizer is created and no parameter is updated.  The same evaluator emits
reconstruction and action-recognition metrics from one deterministic pass.
"""

import argparse
import csv
import hashlib
import json
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from pytorch_msssim import ms_ssim, ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.ucf101_dataloader import UCF101GoPDataset
from data.ucf101_train_val_test import build_train_val_dataloaders
from downstream.action_recognition.models.tsn_recognizer import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TSNModel,
    load_tsn_checkpoint,
)
from model.video_jscc import VideoJSCC


EXPECTED_TEST_CLIPS = 3783


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videojscc_ckpt', required=True)
    p.add_argument('--tsn_ckpt', default='downstream/action_recognition/weights/tsn_ucf101_head.pth')
    p.add_argument('--frames_root', default='datasets/UCF101Frames')
    p.add_argument('--annotation_path', default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    p.add_argument('--split', choices=('validation', 'test'), default='validation')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--gop_size', type=int, default=5)
    p.add_argument('--gops_per_clip', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--test_gop_seed', type=int, default=44)
    p.add_argument('--channel_seed', type=int, default=1042)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='./out_video_clean_evaluation')
    return p.parse_args()


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_classes(annotation_path):
    mapping = {}
    path = Path(annotation_path) / 'classInd.txt'
    for line in path.read_text().splitlines():
        index, name = line.split()
        mapping[name] = int(index) - 1
    if set(mapping.values()) != set(range(101)):
        raise RuntimeError('classInd.txt does not define exactly 101 classes')
    return mapping


def build_test_loader(args):
    class_to_idx = load_classes(args.annotation_path)
    samples, missing = [], []
    testlist = Path(args.annotation_path) / 'testlist01.txt'
    for line in testlist.read_text().splitlines():
        relative = Path(line.strip())
        frame_dir = Path(args.frames_root) / 'test' / relative.parent.name / relative.stem
        if not frame_dir.is_dir():
            missing.append(str(frame_dir))
            continue
        samples.append((str(frame_dir.resolve()), class_to_idx[relative.parent.name]))
    if missing:
        raise FileNotFoundError(f'{len(missing)} official-test frame directories are missing')
    if len(samples) != EXPECTED_TEST_CLIPS:
        raise RuntimeError(f'Expected {EXPECTED_TEST_CLIPS} official-test clips, found {len(samples)}')

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()
    ])
    dataset = UCF101GoPDataset(
        samples, transform, gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip, seed=args.test_gop_seed,
    )
    if len(dataset) != len(samples) * args.gops_per_clip:
        raise RuntimeError('An official-test clip is shorter than the requested GoP')
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    manifest = {
        'protocol': 'UCF101 official test split 1; evaluation only',
        'test_gop_seed': args.test_gop_seed, 'gop_size': args.gop_size,
        'gops_per_clip': args.gops_per_clip,
        'samples': [
            {
                'clip_id': index, 'frame_dir': frame_dir, 'label': int(label),
                'start_frame_index': int(start),
                'frame_files': [Path(frame).name for frame in frame_paths],
            }
            for index, (frame_paths, label, frame_dir, start) in enumerate(dataset.index)
        ],
    }
    return loader, manifest, canonical_hash(manifest)


def build_loader(args):
    if args.split == 'test':
        return build_test_loader(args)
    _, val_loader, manifest, manifest_hash = build_train_val_dataloaders(
        frames_root=args.frames_root, annotation_path=args.annotation_path,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip, val_fraction=0.1,
        seed=args.split_seed, enforce_locked_counts=True,
    )
    return val_loader, manifest, manifest_hash


def load_videojscc(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state' not in checkpoint or 'config' not in checkpoint:
        raise KeyError('Expected a clean checkpoint containing model_state and config')
    config = checkpoint['config']
    model = VideoJSCC(
        c=int(config['c']), channel_type=config['channel'], snr=float(config['snr']),
        n_frames=int(config['gop_size']), hidden_dim=int(config['hidden_dim']),
    )
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def load_tsn(path, device):
    model = TSNModel(pretrained=True, num_classes=101)
    metadata = load_tsn_checkpoint(model, path)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


@contextmanager
def deterministic_channel(seed, device):
    devices = [device.index if device.index is not None else 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        yield


@torch.no_grad()
def evaluate(videojscc, tsn, loader, device, channel_seed):
    rows = []
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)
    clip_id = 0
    with deterministic_channel(channel_seed, device):
        for gops, labels in tqdm(loader, desc='evaluation'):
            gops, labels = gops.to(device), labels.to(device)
            reconstructed_raw = videojscc(gops)
            reconstructed = reconstructed_raw.clamp(0, 1)
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
            logits_recon = tsn((reconstructed - mean) / std)
            logits_clean = tsn((gops - mean) / std)
            top5_recon = logits_recon.topk(5, dim=1).indices
            top5_clean = logits_clean.topk(5, dim=1).indices
            raw_mse = (reconstructed_raw - gops).square().mean(dim=(1, 2, 3, 4))
            for item in range(batch):
                start, end = item * frames, (item + 1) * frames
                rows.append({
                    'clip_id': clip_id, 'label': int(labels[item]),
                    'raw_reconstruction_mse': float(raw_mse[item]),
                    'psnr_db': float(frame_psnr[start:end].mean()),
                    'ssim': float(frame_ssim[start:end].mean()),
                    'ms_ssim_3scale': float(frame_ms[start:end].mean()),
                    'reconstructed_top1_correct': int(top5_recon[item, 0] == labels[item]),
                    'reconstructed_top5_correct': int(top5_recon[item].eq(labels[item]).any()),
                    'clean_top1_correct': int(top5_clean[item, 0] == labels[item]),
                    'clean_top5_correct': int(top5_clean[item].eq(labels[item]).any()),
                })
                clip_id += 1
    return rows


def aggregate(rows):
    count = len(rows)
    if count == 0:
        raise RuntimeError('Evaluator produced no rows')
    mean_fields = ('raw_reconstruction_mse', 'psnr_db', 'ssim', 'ms_ssim_3scale')
    summary = {'clips': count}
    for field in mean_fields:
        summary[field] = sum(row[field] for row in rows) / count
    for prefix in ('reconstructed', 'clean'):
        summary[f'{prefix}_top1_percent'] = 100 * sum(row[f'{prefix}_top1_correct'] for row in rows) / count
        summary[f'{prefix}_top5_percent'] = 100 * sum(row[f'{prefix}_top5_correct'] for row in rows) / count
    return summary


def main():
    args = parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    loader, manifest, manifest_hash = build_loader(args)
    videojscc, checkpoint = load_videojscc(args.videojscc_ckpt, device)
    if args.split == 'validation' and checkpoint['split_manifest_sha256'] != manifest_hash:
        raise RuntimeError(
            'Validation manifest does not match the manifest stored in the VideoJSCC checkpoint'
        )
    tsn, tsn_metadata = load_tsn(args.tsn_ckpt, device)
    before = {name: value.detach().cpu().clone() for name, value in videojscc.state_dict().items()}
    tsn_before = {name: value.detach().cpu().clone() for name, value in tsn.state_dict().items()}
    rows = evaluate(videojscc, tsn, loader, device, args.channel_seed)
    if any(not torch.equal(before[name], value.detach().cpu()) for name, value in videojscc.state_dict().items()):
        raise RuntimeError('VideoJSCC changed during evaluation')
    if any(not torch.equal(tsn_before[name], value.detach().cpu()) for name, value in tsn.state_dict().items()):
        raise RuntimeError('TSN changed during evaluation')

    ckpt_config = checkpoint['config']
    run_name = (
        f"VideoJSCC_TSN_{args.split}_{ckpt_config['channel']}_c{ckpt_config['c']}"
        f"_snr{float(ckpt_config['snr']):g}_seed{args.channel_seed}"
    )
    output = Path(args.out) / run_name
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'per_clip_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'split_manifest.sha256').write_text(manifest_hash + '\n')
    evaluation_config = {
        **vars(args), 'actual_device': str(device),
        'videojscc_checkpoint_sha256': hashlib.sha256(Path(args.videojscc_ckpt).read_bytes()).hexdigest(),
        'videojscc_split_manifest_sha256': checkpoint['split_manifest_sha256'],
        'evaluation_split_manifest_sha256': manifest_hash,
        'tsn_metadata': {k: v for k, v in tsn_metadata.items() if k != 'model_state_dict'},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'pytorch': torch.__version__,
        'parameter_updates': 0,
    }
    (output / 'evaluation_config.json').write_text(json.dumps(evaluation_config, indent=2, default=str))
    summary = {
        **aggregate(rows), 'split': args.split, 'channel_seed': args.channel_seed,
        'parameter_updates': 0, 'videojscc_training_epoch': checkpoint['epoch'],
        'official_test_used': args.split == 'test',
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
