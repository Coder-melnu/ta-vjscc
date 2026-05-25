# -*- coding: utf-8 -*-
"""
Training script for Video DeepJSCC (Approach A).
Processes GoPs of N consecutive frames through shared DeepJSCC + TemporalFusionModule.

Usage:
    python train_video.py \
        --channel AWGN \
        --snr_list 19 13 7 4 1 \
        --ratio_list 1/6 1/12 \
        --frames_root datasets/UCF101Frames \
        --annotation_path datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist \
        --out ./out_video

    # Or use an args file:
    python train_video.py @args/train_video_awgn.txt
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

from model import DeepJSCC, ratio2filtersize
from model.video_jscc import VideoJSCC
from data.ucf101_dataloader import build_dataloaders
from utils import set_seed, view_model_param


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0

    for iter, (gops, _) in enumerate(data_loader):
        # gops: (B, N, 3, H, W)
        gops = gops.to(device)

        optimizer.zero_grad()
        output = model(gops)                        # (B, N, 3, H, W)
        loss   = model.loss(output, gops)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    return epoch_loss, optimizer


def evaluate_epoch(model, device, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for iter, (gops, _) in enumerate(data_loader):
            gops   = gops.to(device)
            output = model(gops)
            loss   = model.loss(output, gops)
            epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    return epoch_loss


# ---------------------------------------------------------------------------
# Arg Parser — supports @args_file
# ---------------------------------------------------------------------------

def config_parser():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Train Video DeepJSCC (Approach A)'
    )

    # Dataset
    parser.add_argument('--frames_root',
                        default='datasets/UCF101Frames')
    parser.add_argument('--annotation_path',
                        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    parser.add_argument('--image_size',      type=int,   default=256)
    parser.add_argument('--gop_size',        type=int,   default=5)
    parser.add_argument('--gops_per_clip',   type=int,   default=1,
                        help='GoPs randomly sampled per clip per epoch')

    # Channel
    parser.add_argument('--channel',         type=str,   default='AWGN',
                        choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr_list',        nargs='+',  default=['19', '13', '7', '4', '1'])
    parser.add_argument('--ratio_list',      nargs='+',  default=['1/6', '1/12'])

    # Model
    parser.add_argument('--hidden_dim',      type=int,   default=16,
                        help='Hidden channels in TemporalFusionModule')

    # Training
    parser.add_argument('--epochs',          type=int,   default=100)
    parser.add_argument('--batch_size',      type=int,   default=8,
                        help='GoP batch size — smaller than frame training due to memory')
    parser.add_argument('--num_workers',     type=int,   default=4)
    parser.add_argument('--init_lr',         type=float, default=1e-3)
    parser.add_argument('--weight_decay',    type=float, default=5e-4)
    parser.add_argument('--step_size',       type=int,   default=50)
    parser.add_argument('--gamma',           type=float, default=0.1)
    parser.add_argument('--min_lr',          type=float, default=1e-5)
    parser.add_argument('--max_time',        type=float, default=12,
                        help='Max training time in hours')
    parser.add_argument('--seed',            type=int,   default=42)

    # Output
    parser.add_argument('--out',             type=str,   default='./out_video')
    parser.add_argument('--device',          type=str,   default='cuda:0')
    parser.add_argument('--disable_tqdm',    type=bool,  default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single SNR+ratio training run
# ---------------------------------------------------------------------------

def train_pipeline(params):

    # Build dataloaders
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

    # Compute c from ratio using a single frame sample
    sample_gop, _ = next(iter(train_loader))    # (B, N, 3, H, W)
    sample_frame  = sample_gop[0, 0]            # (3, H, W)
    c = ratio2filtersize(sample_frame, params['ratio'])
    print(f"SNR={params['snr']} dB | ratio={params['ratio']:.4f} | c={c} | "
          f"channel={params['channel']}")

    # Build model
    model = VideoJSCC(
        c=c,
        channel_type=params['channel'],
        snr=params['snr'],
        n_frames=params['gop_size'],
        hidden_dim=params['hidden_dim'],
    )

    # Output dirs
    phaser = (f"VideoJSCC_{params['channel']}_c{c}"
              f"_snr{params['snr']}_ratio{params['ratio']:.4f}"
              f"_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}")
    root_log_dir    = os.path.join(params['out_dir'], 'logs',        phaser)
    root_ckpt_dir   = os.path.join(params['out_dir'], 'checkpoints', phaser)
    root_config_dir = os.path.join(params['out_dir'], 'configs',     phaser)
    os.makedirs(root_ckpt_dir,   exist_ok=True)
    os.makedirs(root_config_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=root_log_dir)
    writer.add_text('config', str(params))

    # Device — keep string and torch.device separate to avoid yaml.dump issues
    device_str = params['device'] if torch.cuda.is_available() else 'cpu'
    device     = torch.device(device_str)
    model      = model.to(device)

    # Optimizer + scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=params['step_size'], gamma=params['gamma'])

    # Training loop
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []
    best_val_loss  = float('inf')
    epoch          = 0              # initialize before loop for KeyboardInterrupt safety

    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:
                t.set_description(f'Epoch {epoch}')
                start = time.time()

                train_loss, optimizer = train_epoch(
                    model, optimizer, device, train_loader)
                val_loss = evaluate_epoch(model, device, test_loader)

                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)

                # TensorBoard
                writer.add_scalar('train/loss',    train_loss, epoch)
                writer.add_scalar('val/loss',      val_loss,   epoch)
                writer.add_scalar('learning_rate',
                                  optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(
                    time=time.time() - start,
                    lr=optimizer.param_groups[0]['lr'],
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                per_epoch_time.append(time.time() - start)

                # Save latest checkpoint (keep only last epoch)
                ckpt_path = os.path.join(root_ckpt_dir, f'epoch_{epoch}.pkl')
                torch.save(model.state_dict(), ckpt_path)
                for f in glob.glob(os.path.join(root_ckpt_dir, 'epoch_*.pkl')):
                    epoch_nb = int(os.path.splitext(
                        os.path.basename(f))[0].split('_')[-1])
                    if epoch_nb < epoch - 1:
                        os.remove(f)

                # Save best model separately
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(root_ckpt_dir, 'best.pkl'))
                    print(f"\n  ✓ Best model saved (val_loss={best_val_loss:.6f})")

                scheduler.step()

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR reached min_lr, stopping.")
                    break

                if time.time() - t0 > params['max_time'] * 3600:
                    print(f"\nMax time {params['max_time']}h reached, stopping.")
                    break

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt — saving current state and exiting.')
        torch.save(model.state_dict(),
                   os.path.join(root_ckpt_dir, f'interrupted_epoch_{epoch}.pkl'))

    # Final eval
    final_val_loss   = evaluate_epoch(model, device, test_loader)
    final_train_loss = evaluate_epoch(model, device, train_loader)
    print(f"\nFinal Train Loss : {final_train_loss:.6f}")
    print(f"Final Val Loss   : {final_val_loss:.6f}")
    print(f"Best Val Loss    : {best_val_loss:.6f}")
    print(f"Total Time       : {(time.time()-t0)/3600:.2f}h")
    print(f"Avg Time/Epoch   : {np.mean(per_epoch_time):.2f}s")

    writer.add_text('result', (
        f"SNR={params['snr']} | ratio={params['ratio']:.4f} | c={c}\n"
        f"Final Train Loss: {final_train_loss:.6f}\n"
        f"Final Val  Loss : {final_val_loss:.6f}\n"
        f"Best  Val  Loss : {best_val_loss:.6f}\n"
        f"Total Time      : {(time.time()-t0)/3600:.2f}h\n"
        f"Avg Epoch       : {np.mean(per_epoch_time):.2f}s\n"
        f"Params          : {view_model_param(model)}"
    ))
    writer.close()

    # Save config YAML — use device_str (string), not torch.device object
    with open(os.path.join(root_config_dir, 'config.yaml'), 'w') as f:
        yaml.dump({
            **{k: v for k, v in params.items() if k != 'device'},
            'device'     : device_str,
            'c'          : c,
            'best_val_loss' : best_val_loss,
            'total_params'  : view_model_param(model),
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
        'frames_root'    : args.frames_root,
        'annotation_path': args.annotation_path,
        'image_size'     : args.image_size,
        'gop_size'       : args.gop_size,
        'gops_per_clip'  : args.gops_per_clip,
        'channel'        : args.channel,
        'snr_list'       : snr_list,
        'ratio_list'     : ratio_list,
        'hidden_dim'     : args.hidden_dim,
        'epochs'         : args.epochs,
        'batch_size'     : args.batch_size,
        'num_workers'    : args.num_workers,
        'init_lr'        : args.init_lr,
        'weight_decay'   : args.weight_decay,
        'step_size'      : args.step_size,
        'gamma'          : args.gamma,
        'min_lr'         : args.min_lr,
        'max_time'       : args.max_time,
        'seed'           : args.seed,
        'out_dir'        : args.out,
        'device'         : args.device,
        'disable_tqdm'   : args.disable_tqdm,
    }

    for ratio in ratio_list:
        for snr in snr_list:
            params['ratio'] = ratio
            params['snr']   = snr
            train_pipeline(params)


if __name__ == '__main__':
    main()