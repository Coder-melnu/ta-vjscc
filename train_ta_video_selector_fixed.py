# -*- coding: utf-8 -*-
"""
Training script for Task-Aware Video DeepJSCC (TA-VideoJSCC) — Staged Training.

Staged training:
    Stage 1 (epochs 0 to stage1_epochs-1):
        - Selector bypassed (mask=1, all features pass through)
        - Loss = L_recon (MSE only)
        - Goal: codec reaches reasonable reconstruction quality first
        - Gumbel temperature held at tau_init

    Stage 2 (epochs stage1_epochs to epochs-1):
        - Selector active (Gumbel-Softmax)
        - Loss = lambda_task*L_task + lambda_recon*L_recon + lambda_rate*L_rate
        - Gumbel temperature anneals from tau_init to tau_min over Stage 2
        - TSN gradients now meaningful because reconstruction is already decent

Fixes applied:
    Bug #1: evaluate_epoch uses logits returned by joint_loss() — no second
            forward pass. Removes stochastic inconsistency + halves eval compute.
    Bug #5: --disable_tqdm uses action='store_true' (bool type was broken).
    Bug #6: Removed unused 'start' variable.
    Bug #7: weights_only=True added to torch.load() in FrozenTSN.

Usage:
    conda activate ta-vjscc

    # Staged training — recommended (validate on SNR=13 first):
    python train_ta_video.py \
        --channel AWGN \
        --snr_list 13 \
        --ratio_list 1/6 \
        --epochs 100 \
        --stage1_epochs 50 \
        --lambda_task 1.0 --lambda_recon 0.1 --lambda_rate 0.01 \
        --out ./out_ta_staged

    # Full sweep after validation:
    python train_ta_video.py \
        --channel AWGN \
        --snr_list 19 13 7 4 1 \
        --ratio_list 1/6 1/12 \
        --epochs 100 \
        --stage1_epochs 50 \
        --lambda_task 1.0 --lambda_recon 0.1 --lambda_rate 0.01 \
        --out ./out_ta_staged_full

    # Joint training (original, no staging — stage1_epochs=0):
    python train_ta_video.py \
        --snr_list 13 --ratio_list 1/6 \
        --stage1_epochs 0 \
        --out ./out_ta_joint
"""

import os
import glob
import json
import csv
import time
import yaml
import numpy as np
import argparse
import random
import re
from fractions import Fraction

import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import ratio2filtersize
from model.ta_video_jscc_selector_fixed import TAVideoJSCC
from data.ucf101_train_val_test import build_train_val_dataloaders
from utils import set_seed, view_model_param


# ---------------------------------------------------------------------------
# Train epoch
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, device, data_loader, max_iters=None):
    model.train()
    # TSN must stay in eval mode even while the rest of the model trains —
    # otherwise BatchNorm in ResNet-50 updates its running stats on
    # reconstructed frames, corrupting the fixed supervision signal.
    if model.tsn is not None:
        model.tsn.eval()

    totals  = {'loss': 0, 'l_task': 0, 'l_recon': 0, 'l_rate': 0,
               'weighted_task': 0, 'weighted_recon': 0, 'weighted_rate': 0,
               'psnr': 0, 'rate_mean': 0, 'score_mean': 0}
    n_iters = 0

    for gops, labels in data_loader:
        if max_iters is not None and n_iters >= max_iters:
            break

        gops   = gops.to(device)    # (B, N, 3, H, W)
        labels = labels.to(device)  # (B,)

        optimizer.zero_grad()
        # joint_loss returns x_refined and tsn_logits.
        # We discard them here (not needed for the backward pass).
        loss, info, _, _ = model.joint_loss(gops, labels)
        loss.backward()
        optimizer.step()

        for k in totals:
            if k in info:
                totals[k] += info[k]
        n_iters += 1

    return {k: v / n_iters for k, v in totals.items()}, optimizer


# ---------------------------------------------------------------------------
# Eval epoch
# ---------------------------------------------------------------------------

def evaluate_epoch(model, device, data_loader):
    model.eval()

    totals  = {'loss': 0, 'l_task': 0, 'l_recon': 0, 'l_rate': 0,
               'weighted_task': 0, 'weighted_recon': 0, 'weighted_rate': 0,
               'psnr': 0, 'rate_mean': 0, 'score_mean': 0}
    correct1 = correct5 = 0
    total   = 0
    n_iters = 0

    with torch.no_grad():
        for gops, labels in data_loader:
            gops   = gops.to(device)
            labels = labels.to(device)

            # Use tsn_logits returned by joint_loss directly.
            # No second model(gops) call — single forward pass per batch.
            loss, info, _, tsn_logits = model.joint_loss(gops, labels)

            for k in totals:
                if k in info:
                    totals[k] += info[k]
            n_iters += 1

            if tsn_logits is not None:
                top5 = tsn_logits.topk(5, dim=1).indices
                correct1 += (top5[:, 0] == labels).sum().item()
                correct5 += top5.eq(labels.view(-1, 1)).any(dim=1).sum().item()
                total   += labels.size(0)

    metrics = {k: v / n_iters for k, v in totals.items()}
    metrics['top1_acc'] = correct1 / total if total > 0 else 0.0
    metrics['top5_acc'] = correct5 / total if total > 0 else 0.0
    return metrics


@torch.no_grad()
def paired_validation_evaluation(model, device, data_loader, mode, seed):
    """Per-clip learned/random metrics under identical channel-noise draws."""
    model.eval()
    model.set_selector_mode(mode)
    devices = [device.index if device.type == 'cuda' and device.index is not None else 0] \
        if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        rows, row_id = [], 0
        for gops, labels in data_loader:
            gops, labels = gops.to(device), labels.to(device)
            reconstructed, _, _ = model(gops)
            logits = model.tsn(reconstructed)
            mse = (reconstructed.clamp(0, 1) - gops).square().mean(dim=(1, 2, 3, 4))
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            ce = F.cross_entropy(logits, labels, reduction='none')
            predicted = logits.argmax(dim=1)
            for index in range(labels.size(0)):
                rows.append({
                    'clip_id': row_id, 'label': int(labels[index]),
                    'correct': int(predicted[index] == labels[index]),
                    'cross_entropy': float(ce[index]), 'psnr_db': float(psnr[index]),
                })
                row_id += 1
    return rows


def mcnemar_exact_pvalue(learned, random_rows):
    """Two-sided exact McNemar p-value for paired correctness outcomes."""
    b = sum(a['correct'] == 1 and r['correct'] == 0 for a, r in zip(learned, random_rows))
    c = sum(a['correct'] == 0 and r['correct'] == 1 for a, r in zip(learned, random_rows))
    n = b + c
    if n == 0:
        return b, c, 1.0
    lower = sum(__import__('math').comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
    return b, c, min(1.0, 2.0 * lower)


@torch.no_grad()
def save_spatial_power_maps(model, device, dataset, output_dir):
    """Save labelled actual post-renormalisation spatial power maps."""
    model.eval()
    model.set_selector_mode('learned')
    output_dir = os.path.join(output_dir, 'spatial_power_maps')
    os.makedirs(output_dir, exist_ok=True)
    for item in range(min(2, len(dataset))):
        gop, _ = dataset[item]
        gop = gop.unsqueeze(0).to(device)
        model(gop)
        power = model.last_spatial_power.reshape(1, gop.size(1), 1, *model.last_spatial_power.shape[-2:])[0]
        figure, axes = plt.subplots(2, gop.size(1), figsize=(3 * gop.size(1), 6))
        for frame in range(gop.size(1)):
            axes[0, frame].imshow(gop[0, frame].permute(1, 2, 0).cpu())
            axes[0, frame].set_title(f'Frame {frame + 1}')
            axes[1, frame].imshow(power[frame, 0].cpu(), cmap='magma', vmin=0)
            axes[0, frame].axis('off'); axes[1, frame].axis('off')
        axes[0, 0].set_ylabel('Input frame')
        axes[1, 0].set_ylabel('Transmit-power share')
        figure.suptitle('Learned spatial transmit-power allocation')
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, f'validation_gop_{item:02d}.png'), dpi=180)
        plt.close(figure)


def load_videojscc_initialization(model, checkpoint_path):
    """Load matching reconstruction-only VideoJSCC weights strictly."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'VideoJSCC checkpoint not found: {checkpoint_path}'
        )
    # This is a locally produced, trusted clean-baseline artifact.  Its
    # metadata includes RNG states, which PyTorch cannot read with
    # weights_only=True on current releases.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(state, dict):
        # Clean VideoJSCC checkpoints are self-describing training artifacts.
        # Older checkpoints may still contain model_state_dict.
        state = state.get('model_state', state.get('model_state_dict', state))
    jscc_state = {k[5:]: v for k, v in state.items() if k.startswith('jscc.')}
    temporal_state = {
        k[9:]: v for k, v in state.items() if k.startswith('temporal.')
    }
    if not jscc_state or not temporal_state:
        raise KeyError("Checkpoint must contain both 'jscc.*' and 'temporal.*'")
    model.jscc.load_state_dict(jscc_state, strict=True)
    model.temporal.load_state_dict(temporal_state, strict=True)
    print(f'[VideoJSCC init] Loaded: {checkpoint_path}')


@torch.no_grad()
def audit_all_ones_selector(model, loader, device, seed=12345):
    """Compare bypass and all-ones re-normalisation under identical noise."""
    model.eval()
    gops, _ = next(iter(loader))
    gops = gops.to(device)

    def run(mode):
        model.set_selector_mode(mode)
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        reconstructed, _, _ = model(gops)
        mse = torch.nn.functional.mse_loss(reconstructed, gops)
        return reconstructed, -10.0 * torch.log10(mse.clamp_min(1e-12)).item()

    bypass, bypass_psnr = run('bypass')
    all_ones, all_ones_psnr = run('all_ones')
    result = {
        'bypass_psnr': bypass_psnr,
        'all_ones_psnr': all_ones_psnr,
        'output_max_abs_diff': (bypass - all_ones).abs().max().item(),
        **model.last_power_audit,
    }
    print('\n100% retention power-path audit')
    for key, value in result.items():
        print(f'  {key}: {value:.10g}')
    if abs(bypass_psnr - all_ones_psnr) > 1e-4 or result['output_max_abs_diff'] > 1e-5:
        raise RuntimeError(
            'All-ones selector failed to reproduce the bypass path. '
            'Do not start training.'
        )
    print('  PASS: all-ones selector reproduces the VideoJSCC path')
    return result


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def config_parser():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Train TA-VideoJSCC with staged training'
    )

    # Dataset
    parser.add_argument('--frames_root',
                        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
                        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--image_size',    type=int,   default=128)
    parser.add_argument('--gop_size',      type=int,   default=5)
    parser.add_argument('--gops_per_clip', type=int,   default=1)
    parser.add_argument('--val_fraction',  type=float, default=0.1,
                        help='Group-aware validation fraction drawn only from '
                             'the official training split')

    # Channel
    parser.add_argument('--channel',       type=str,   default='AWGN',
                        choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr_list',      nargs='+',  default=['19', '13', '7', '4', '1'])
    parser.add_argument('--ratio_list',    nargs='+',  default=['1/6', '1/12'])

    # Model
    parser.add_argument('--hidden_dim',    type=int,   default=16)
    parser.add_argument('--tau',           type=float, default=1.0,
                        help='Initial Gumbel-Softmax temperature')
    parser.add_argument('--tau_min',       type=float, default=0.3,
                        help='Final temperature after linear annealing (Stage 2 only)')

    # TSN
    parser.add_argument('--tsn_head_ckpt',
                        default='downstream/action_recognition/weights/tsn_ucf101_head.pth')
    parser.add_argument('--videojscc_ckpt', required=True,
                        help='Matching VideoJSCC best.pkl initialization')
    parser.add_argument('--audit_only', action='store_true',
                        help='Run the exact all-ones selector audit and exit')
    parser.add_argument('--run_random_ablation', action='store_true',
                        help='After selecting the best checkpoint, evaluate '
                             'learned versus spatially permuted weights')

    # Staged training
    parser.add_argument('--stage1_epochs', type=int,   default=50,
                        help='Number of reconstruction-only epochs (Stage 1). '
                             'Set to 0 to disable staged training (original joint training).')

    parser.add_argument('--max_iters_per_epoch', type=int, default=None,
                        help='Cap iterations per epoch for faster cycling. '
                             'None = full dataset.')
    parser.add_argument('--lambda_task',   type=float, default=1.0)
    parser.add_argument('--lambda_recon',  type=float, default=0.1)
    parser.add_argument('--lambda_rate',   type=float, default=0.01)

    # Training
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--batch_size',    type=int,   default=4,
                        help='Smaller than Week 3 — TSN forward adds GPU memory')
    parser.add_argument('--num_workers',   type=int,   default=4)
    parser.add_argument('--init_lr',       type=float, default=1e-3)
    parser.add_argument('--weight_decay',  type=float, default=5e-4)
    parser.add_argument('--step_size',     type=int,   default=50)
    parser.add_argument('--gamma',         type=float, default=0.1)
    parser.add_argument('--min_lr',        type=float, default=1e-5)
    parser.add_argument('--max_time',      type=float, default=999,
                        help='Max wall-clock training time in hours')
    parser.add_argument('--seed',          type=int,   default=42)

    # Output
    parser.add_argument('--out',           type=str,   default='./out_ta_video')
    parser.add_argument('--device',        type=str,   default='cuda:0')
    parser.add_argument('--disable_tqdm',  action='store_true', default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_pipeline(params):

    print(f"\nLoading UCF101 GoP dataset (N={params['gop_size']})...")
    train_loader, val_loader, split_manifest, split_manifest_hash = build_train_val_dataloaders(
        frames_root=params['frames_root'],
        annotation_path=params['annotation_path'],
        image_size=params['image_size'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        gop_size=params['gop_size'],
        gops_per_clip=params['gops_per_clip'],
        val_fraction=params['val_fraction'],
        seed=params['seed'],
    )

    # Do not consume a training batch merely to infer the latent depth.
    c = ratio2filtersize(
        torch.empty(3, params['image_size'], params['image_size']),
        params['ratio'],
    )

    stage1_epochs = params.get('stage1_epochs', 50)
    stage2_epochs = params['epochs'] - stage1_epochs

    print(f"SNR={params['snr']} dB | ratio={params['ratio']:.4f} | c={c} | "
          f"channel={params['channel']}")
    print(f"λ_task={params['lambda_task']} | λ_recon={params['lambda_recon']} | "
          f"λ_rate={params['lambda_rate']} | τ_init={params['tau']} → τ_min={params['tau_min']}")
    print(f"Staged training: Stage1={stage1_epochs} epochs | Stage2={stage2_epochs} epochs")
    if stage1_epochs == 0:
        print("  (stage1_epochs=0 — joint training from epoch 0, no staging)")

    device_str = params['device'] if torch.cuda.is_available() else 'cpu'
    device     = torch.device(device_str)

    model = TAVideoJSCC(
        c=c,
        channel_type=params['channel'],
        snr=params['snr'],
        n_frames=params['gop_size'],
        hidden_dim=params['hidden_dim'],
        tsn_head_ckpt=params['tsn_head_ckpt'],
        lambda_task=params['lambda_task'],
        lambda_recon=params['lambda_recon'],
        lambda_rate=params['lambda_rate'],
        tau=params['tau'],
        device=device_str,
    ).to(device)
    load_videojscc_initialization(model, params['videojscc_ckpt'])

    # Start in Stage 1 (or Stage 2 immediately if stage1_epochs=0)
    initial_stage = 1 if stage1_epochs > 0 else 2
    model.set_stage(initial_stage)

    audit = audit_all_ones_selector(model, val_loader, device)
    model.set_stage(initial_stage)
    if params.get('audit_only'):
        print('\nAudit-only run completed; no optimizer step was taken.')
        return

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable:,} | Frozen (TSN): {frozen:,}")

    # Output dirs
    tag = (f"TAVideoJSCC_{params['channel']}_c{c}"
           f"_snr{params['snr']}_ratio{params['ratio']:.4f}"
           f"_s1ep{stage1_epochs}"
           f"_lt{params['lambda_task']}_lr{params['lambda_recon']}"
           f"_lrate{params['lambda_rate']}"
           f"_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}")
    root_log_dir    = os.path.join(params['out_dir'], 'logs',        tag)
    root_ckpt_dir   = os.path.join(params['out_dir'], 'checkpoints', tag)
    root_config_dir = os.path.join(params['out_dir'], 'configs',     tag)
    os.makedirs(root_ckpt_dir,   exist_ok=True)
    os.makedirs(root_config_dir, exist_ok=True)
    with open(os.path.join(root_config_dir, 'split_manifest.json'), 'w') as handle:
        json.dump(split_manifest, handle, indent=2)
    with open(os.path.join(root_config_dir, 'split_manifest.sha256'), 'w') as handle:
        handle.write(split_manifest_hash + '\n')

    writer = SummaryWriter(log_dir=root_log_dir)
    writer.add_text('config', str(params))
    metrics_handle = open(os.path.join(root_config_dir, 'metrics.csv'), 'w', newline='')
    metric_fields = [
        'epoch', 'stage', 'tau', 'lr',
        'train_loss', 'train_task_loss', 'train_recon_loss',
        'val_loss', 'val_task_loss', 'val_recon_loss',
        'val_top1', 'val_top5', 'val_psnr', 'val_mean_soft_weight',
    ]
    metrics_writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
    metrics_writer.writeheader()

    # Only pass trainable parameters to optimizer
    # NOTE: in Stage 1 the selector is frozen so its params are excluded here.
    # In Stage 2, set_stage(2) unfreezes them — but the optimizer won't pick
    # them up automatically. We rebuild the optimizer at the Stage 2 transition.
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=params['init_lr'],
        weight_decay=params['weight_decay'],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params['step_size'], gamma=params['gamma']
    )

    tau_init = params['tau']
    tau_min  = params['tau_min']

    t0            = time.time()
    best_val_loss = float('inf')
    epoch         = 0
    stage2_started = False

    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')

                # -------------------------------------------------------
                # STAGED TRAINING: set stage and temperature each epoch
                # -------------------------------------------------------
                if stage1_epochs == 0 or epoch >= stage1_epochs:
                    # Stage 2: full joint loss, selector active
                    model.set_stage(2)

                    # Rebuild optimizer once at Stage 2 transition to include
                    # selector parameters (they were frozen in Stage 1)
                    if not stage2_started:
                        print(f"\n[Epoch {epoch}] Transitioning to Stage 2 — "
                              f"rebuilding optimizer to include selector params.")
                        optimizer = optim.Adam(
                            [p for p in model.parameters() if p.requires_grad],
                            lr=params['init_lr'],
                            weight_decay=params['weight_decay'],
                        )
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=max(stage2_epochs // 2, 1),
                            gamma=params['gamma']
                        )
                        stage2_started = True

                    # Anneal temperature over Stage 2 only
                    stage2_epoch = epoch - stage1_epochs
                    tau = tau_init - (tau_init - tau_min) * stage2_epoch / max(stage2_epochs - 1, 1)

                else:
                    # Stage 1: reconstruction only, selector bypassed
                    model.set_stage(1)
                    tau = tau_init  # hold temperature at init during Stage 1

                model.set_temperature(tau)
                # -------------------------------------------------------

                train_metrics, optimizer = train_epoch(
                    model, optimizer, device, train_loader,
                    max_iters=params.get('max_iters_per_epoch'))
                val_metrics = evaluate_epoch(model, device, val_loader)
                metrics_writer.writerow({
                    'epoch': epoch + 1, 'stage': model.stage, 'tau': tau,
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': train_metrics['loss'],
                    'train_task_loss': train_metrics['l_task'],
                    'train_recon_loss': train_metrics['l_recon'],
                    'val_loss': val_metrics['loss'],
                    'val_task_loss': val_metrics['l_task'],
                    'val_recon_loss': val_metrics['l_recon'],
                    'val_top1': val_metrics['top1_acc'],
                    'val_top5': val_metrics['top5_acc'],
                    'val_psnr': val_metrics['psnr'],
                    'val_mean_soft_weight': val_metrics['rate_mean'],
                })
                metrics_handle.flush()

                # TensorBoard
                for k, v in train_metrics.items():
                    writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f'val/{k}', v, epoch)
                # Soft weighting transmits every symbol, so actual CBR remains
                # nominal. Mean weight is a power-allocation statistic, not rate.
                writer.add_scalar('val/actual_cbr', params['ratio'], epoch)
                writer.add_scalar('val/mean_soft_weight',
                                  val_metrics['rate_mean'], epoch)
                writer.add_scalar('tau',   tau,   epoch)
                writer.add_scalar('stage', float(model.stage), epoch)
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(
                    stage   = f"S{model.stage}",
                    loss    = f"{train_metrics['loss']:.4f}",
                    l_task  = f"{train_metrics['l_task']:.4f}",
                    l_recon = f"{train_metrics['l_recon']:.4f}",
                    rate    = f"{train_metrics['rate_mean']:.3f}",
                    val_acc = f"{val_metrics['top1_acc']*100:.1f}%",
                    val_top5 = f"{val_metrics['top5_acc']*100:.1f}%",
                    psnr     = f"{val_metrics['psnr']:.2f}dB",
                    tau     = f"{tau:.2f}",
                )

                # Checkpoint — keep only last two epochs to save disk
                ckpt_path = os.path.join(root_ckpt_dir, f'epoch_{epoch}.pkl')
                torch.save(model.state_dict(), ckpt_path)
                for f in glob.glob(os.path.join(root_ckpt_dir, 'epoch_*.pkl')):
                    nb = int(os.path.splitext(os.path.basename(f))[0].split('_')[-1])
                    if nb < epoch - 1:
                        os.remove(f)

                # Save best checkpoint (based on val loss in Stage 2 only)
                if model.stage == 2 and val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    torch.save(model.state_dict(),
                               os.path.join(root_ckpt_dir, 'best.pkl'))
                    print(f"\n  ✓ Best saved (Stage 2) — "
                          f"val_loss={best_val_loss:.4f} | "
                          f"val_Top1={val_metrics['top1_acc']*100:.1f}% | "
                          f"val_Top5={val_metrics['top5_acc']*100:.1f}%")

                # Save Stage 1 final checkpoint for inspection
                if epoch == stage1_epochs - 1 and stage1_epochs > 0:
                    torch.save(model.state_dict(),
                               os.path.join(root_ckpt_dir, 'stage1_final.pkl'))
                    print(f"\n  ✓ Stage 1 final checkpoint saved — "
                          f"val_recon={val_metrics['l_recon']:.4f}")

                scheduler.step()

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR reached min_lr — stopping early.")
                    break

                # Wall-clock time limit
                elapsed_h = (time.time() - t0) / 3600
                if elapsed_h >= params['max_time']:
                    print(f"\n!! max_time={params['max_time']}h reached — stopping.")
                    break

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt — saving checkpoint.')
        torch.save(model.state_dict(),
                   os.path.join(root_ckpt_dir, f'interrupted_epoch_{epoch}.pkl'))

    # Final evaluation must use the selected checkpoint, not the last epoch.
    best_path = os.path.join(root_ckpt_dir, 'best.pkl')
    if os.path.isfile(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True),
            strict=True,
        )
    model.set_selector_mode('learned')
    final_val = evaluate_epoch(model, device, val_loader)
    # The official UCF101 test set is intentionally absent from this trainer.
    # Random placement is a validation-only pilot diagnostic; the final paired
    # learned-vs-random test is run separately on the reserved test set.
    random_val = None
    if params.get('run_random_ablation'):
        torch.manual_seed(params['seed'])
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(params['seed'])
        model.set_selector_mode('random')
        random_val = evaluate_epoch(model, device, val_loader)
        model.set_selector_mode('learned')

    # Save paired per-clip outcomes for the mentor's learned-versus-random
    # gate.  Both conditions use identical validation GoPs and channel noise.
    learned_rows = paired_validation_evaluation(
        model, device, val_loader, 'learned', params['seed'] + 1000
    )
    random_rows = paired_validation_evaluation(
        model, device, val_loader, 'random', params['seed'] + 1000
    )
    learned_wins, random_wins, p_value = mcnemar_exact_pvalue(learned_rows, random_rows)
    paired_path = os.path.join(root_config_dir, 'learned_vs_random_validation.csv')
    with open(paired_path, 'w', newline='') as handle:
        fields = ['clip_id', 'label', 'learned_correct', 'random_correct',
                  'learned_cross_entropy', 'random_cross_entropy',
                  'learned_psnr_db', 'random_psnr_db']
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        for learned_row, random_row in zip(learned_rows, random_rows):
            writer_csv.writerow({
                'clip_id': learned_row['clip_id'], 'label': learned_row['label'],
                'learned_correct': learned_row['correct'], 'random_correct': random_row['correct'],
                'learned_cross_entropy': learned_row['cross_entropy'],
                'random_cross_entropy': random_row['cross_entropy'],
                'learned_psnr_db': learned_row['psnr_db'], 'random_psnr_db': random_row['psnr_db'],
            })
    paired_summary = {
        'split': 'internal_validation', 'samples': len(learned_rows),
        'learned_correct_random_wrong': learned_wins,
        'random_correct_learned_wrong': random_wins,
        'mcnemar_exact_two_sided_p': p_value,
        'channel_seed': params['seed'] + 1000,
    }
    with open(os.path.join(root_config_dir, 'learned_vs_random_validation.json'), 'w') as handle:
        json.dump(paired_summary, handle, indent=2)
    save_spatial_power_maps(model, device, val_loader.dataset, root_config_dir)
    model.set_selector_mode('learned')
    print(f"\nFinal val loss     : {final_val['loss']:.4f}")
    print(f"Final val top1 acc : {final_val['top1_acc']*100:.1f}%")
    print(f"Final val top5 acc : {final_val['top5_acc']*100:.1f}%")
    print(f"Final val PSNR     : {final_val['psnr']:.2f} dB")
    print(f"Mean soft weight   : {final_val['rate_mean']:.3f}  "
          f"(power-allocation statistic; not retention)")
    print(f"Actual CBR         : {params['ratio']:.6f} (all soft-weighted symbols sent)")
    if random_val is not None:
        print('\nMatched-weight random-placement ablation')
        print(f"  learned validation Top-1: {final_val['top1_acc']*100:.1f}%")
        print(f"  random  validation Top-1: {random_val['top1_acc']*100:.1f}%")
        print(f"  learned validation Top-5: {final_val['top5_acc']*100:.1f}%")
        print(f"  random  validation Top-5: {random_val['top5_acc']*100:.1f}%")
        print(f"  learned validation PSNR : {final_val['psnr']:.2f} dB")
        print(f"  random  validation PSNR : {random_val['psnr']:.2f} dB")
    print(f"Total time         : {(time.time()-t0)/3600:.2f}h")

    writer.add_text('result', str(final_val))
    writer.close()
    metrics_handle.close()

    with open(os.path.join(root_config_dir, 'config.yaml'), 'w') as f:
        yaml.dump({
            **{k: v for k, v in params.items() if k != 'device'},
            'device'             : device_str,
            'c'                  : c,
            'stage1_epochs'      : stage1_epochs,
            'stage2_epochs'      : stage2_epochs,
            'max_iters_per_epoch': params.get('max_iters_per_epoch'),
            'best_val_loss'      : best_val_loss,
            'final_top1_acc'     : final_val['top1_acc'],
            'final_top5_acc'     : final_val['top5_acc'],
            'actual_cbr'         : params['ratio'],
            'mean_soft_weight'   : final_val['rate_mean'],
            'all_ones_audit'     : audit,
            'random_validation_ablation': random_val,
            'learned_vs_random_validation': paired_summary,
            'split_manifest_sha256': split_manifest_hash,
            'trainable_params'   : trainable,
        }, f)

    del model, optimizer, scheduler, train_loader, val_loader, writer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = config_parser()
    set_seed(args.seed)

    snr_list   = list(map(float, args.snr_list))
    ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))

    params = {
        'frames_root'        : args.frames_root,
        'annotation_path'    : args.annotation_path,
        'image_size'         : args.image_size,
        'gop_size'           : args.gop_size,
        'gops_per_clip'      : args.gops_per_clip,
        'val_fraction'       : args.val_fraction,
        'channel'            : args.channel,
        'snr_list'           : snr_list,
        'ratio_list'         : ratio_list,
        'hidden_dim'         : args.hidden_dim,
        'tau'                : args.tau,
        'tau_min'            : args.tau_min,
        'stage1_epochs'      : args.stage1_epochs,
        'max_iters_per_epoch': args.max_iters_per_epoch,
        'tsn_head_ckpt'      : args.tsn_head_ckpt,
        'videojscc_ckpt'     : args.videojscc_ckpt,
        'audit_only'         : args.audit_only,
        'run_random_ablation': args.run_random_ablation,
        'lambda_task'        : args.lambda_task,
        'lambda_recon'       : args.lambda_recon,
        'lambda_rate'        : args.lambda_rate,
        'epochs'             : args.epochs,
        'batch_size'         : args.batch_size,
        'num_workers'        : args.num_workers,
        'init_lr'            : args.init_lr,
        'weight_decay'       : args.weight_decay,
        'step_size'          : args.step_size,
        'gamma'              : args.gamma,
        'min_lr'             : args.min_lr,
        'max_time'           : args.max_time,
        'seed'               : args.seed,
        'out_dir'            : args.out,
        'device'             : args.device,
        'disable_tqdm'       : args.disable_tqdm,
    }

    for ratio in ratio_list:
        for snr in snr_list:
            params['ratio'] = ratio
            params['snr']   = snr
            train_pipeline(params)


if __name__ == '__main__':
    main()
