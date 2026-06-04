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
import time
import yaml
import numpy as np
import argparse
from fractions import Fraction

import torch
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import ratio2filtersize
from model.ta_video_jscc import TAVideoJSCC
from data.ucf101_dataloader import build_dataloaders
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

    totals  = {'loss': 0, 'l_task': 0, 'l_recon': 0,
               'l_rate': 0, 'rate_mean': 0, 'score_mean': 0}
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

    totals  = {'loss': 0, 'l_task': 0, 'l_recon': 0,
               'l_rate': 0, 'rate_mean': 0, 'score_mean': 0}
    correct = 0
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
                preds    = tsn_logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

    metrics = {k: v / n_iters for k, v in totals.items()}
    metrics['top1_acc'] = correct / total if total > 0 else 0.0
    return metrics


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
    parser.add_argument('--image_size',    type=int,   default=256)
    parser.add_argument('--gop_size',      type=int,   default=5)
    parser.add_argument('--gops_per_clip', type=int,   default=1)

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
    train_loader, test_loader = build_dataloaders(
        frames_root=params['frames_root'],
        annotation_path=params['annotation_path'],
        mode='gop',
        image_size=params['image_size'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        gop_size=params['gop_size'],
        gops_per_clip=params['gops_per_clip'],
        seed=params['seed'],
    )

    # Infer c from bandwidth ratio
    sample_gop, _ = next(iter(train_loader))
    sample_frame  = sample_gop[0, 0]                   # (3, H, W)
    c = ratio2filtersize(sample_frame, params['ratio'])

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

    # Start in Stage 1 (or Stage 2 immediately if stage1_epochs=0)
    initial_stage = 1 if stage1_epochs > 0 else 2
    model.set_stage(initial_stage)

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

    writer = SummaryWriter(log_dir=root_log_dir)
    writer.add_text('config', str(params))

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
                val_metrics = evaluate_epoch(model, device, test_loader)

                # TensorBoard
                for k, v in train_metrics.items():
                    writer.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():
                    writer.add_scalar(f'val/{k}', v, epoch)
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
                          f"val_acc={val_metrics['top1_acc']*100:.1f}%")

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

    # Final evaluation
    final_val = evaluate_epoch(model, device, test_loader)
    print(f"\nFinal val loss     : {final_val['loss']:.4f}")
    print(f"Final val top1 acc : {final_val['top1_acc']*100:.1f}%")
    print(f"Final rate_mean    : {final_val['rate_mean']:.3f}  "
          f"(fraction of features kept)")
    print(f"Total time         : {(time.time()-t0)/3600:.2f}h")

    writer.add_text('result', str(final_val))
    writer.close()

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
            'trainable_params'   : trainable,
        }, f)

    del model, optimizer, scheduler, train_loader, test_loader, writer


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
        'channel'            : args.channel,
        'snr_list'           : snr_list,
        'ratio_list'         : ratio_list,
        'hidden_dim'         : args.hidden_dim,
        'tau'                : args.tau,
        'tau_min'            : args.tau_min,
        'stage1_epochs'      : args.stage1_epochs,
        'max_iters_per_epoch': args.max_iters_per_epoch,
        'tsn_head_ckpt'      : args.tsn_head_ckpt,
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