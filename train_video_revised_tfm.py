# -*- coding: utf-8 -*-
"""
Controlled training for the difference-aware RGB Temporal Fusion Module.

The encoder and decoder are initialized from a trained VideoJSCC checkpoint
and frozen. Only the revised TFM is optimized.

Dataset protocol:
  - internal train/validation subsets come only from official UCF101 train split 1
  - the internal split is stratified by action class and respects UCF101 groups
  - official UCF101 test split 1 is reserved and never used for checkpoint selection
"""

import argparse
import glob
import os
import time
from fractions import Fraction

import numpy as np
import torch
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_test_dataloaders
from model import ratio2filtersize
from model.video_jscc import VideoJSCC
from utils import set_seed, view_model_param


def train_epoch(model, optimizer, device, data_loader):
    model.train()

    # Keep the frozen reconstruction network in inference mode even if future
    # versions add BatchNorm or dropout. The stochastic channel still operates.
    model.jscc.eval()

    epoch_loss = 0.0

    for iteration, (gops, _) in enumerate(data_loader):
        gops = gops.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(gops)
        loss = model.loss(output, gops)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

    return epoch_loss / (iteration + 1)


@torch.no_grad()
def evaluate_epoch(model, device, data_loader, channel_seed):
    """Evaluate with reproducible channel noise without altering training RNG."""
    model.eval()
    epoch_loss = 0.0

    cuda_devices = []
    if device.type == 'cuda':
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]

    # fork_rng restores the training RNG state when validation finishes.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(channel_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(channel_seed)

        for iteration, (gops, _) in enumerate(data_loader):
            gops = gops.to(device, non_blocking=True)
            output = model(gops)
            loss = model.loss(output, gops)
            epoch_loss += loss.detach().item()

    return epoch_loss / (iteration + 1)


def config_parser():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Train the revised RGB TFM with frozen pretrained JSCC',
    )

    # Dataset
    parser.add_argument('--frames_root', default='datasets/UCF101Frames')
    parser.add_argument(
        '--annotation_path',
        default=(
            'datasets/UCF101TrainTestSplits-RecognitionTask/'
            'ucfTrainTestlist'
        ),
    )
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--gop_size', type=int, default=5)
    parser.add_argument('--gops_per_clip', type=int, default=1)
    parser.add_argument('--val_fraction', type=float, default=0.1)

    # Channel
    parser.add_argument(
        '--channel',
        type=str,
        default='AWGN',
        choices=['AWGN', 'Rayleigh'],
    )
    parser.add_argument('--snr_list', nargs='+', default=['13'])
    parser.add_argument('--ratio_list', nargs='+', default=['1/6'])

    # Model
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument(
        '--jscc_ckpt',
        type=str,
        required=True,
        help='Old VideoJSCC checkpoint; only jscc.* weights are loaded',
    )

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--lr_patience', type=int, default=4)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-6)
    parser.add_argument('--max_time', type=float, default=12)
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--out', type=str, default='./out_revised_tfm')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--disable_tqdm', action='store_true')

    return parser.parse_args()


def load_and_freeze_jscc(model, checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'VideoJSCC checkpoint not found: {checkpoint_path}'
        )

    old_state = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=True,
    )

    if not isinstance(old_state, dict):
        raise TypeError('VideoJSCC checkpoint is not a state dictionary')

    jscc_state = {
        key[len('jscc.'):]: value
        for key, value in old_state.items()
        if key.startswith('jscc.')
    }

    if not jscc_state:
        raise KeyError(
            "No keys beginning with 'jscc.' were found in the checkpoint"
        )

    model.jscc.load_state_dict(jscc_state, strict=True)

    for parameter in model.jscc.parameters():
        parameter.requires_grad = False

    if any(parameter.requires_grad for parameter in model.jscc.parameters()):
        raise RuntimeError('At least one JSCC parameter is still trainable')


def train_pipeline(params):
    print(f"\nLoading UCF101 GoP dataset (N={params['gop_size']})...")

    train_loader, val_loader, official_test_loader = (
        build_train_val_test_dataloaders(
            frames_root=params['frames_root'],
            annotation_path=params['annotation_path'],
            image_size=params['image_size'],
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            split=1,
            gop_size=params['gop_size'],
            gops_per_clip=params['gops_per_clip'],
            val_fraction=params['val_fraction'],
            seed=params['seed'],
        )
    )

    print('\nDataset protocol')
    print(f"  Internal train: {len(train_loader.dataset)} GoPs")
    print(f"  Internal val:   {len(val_loader.dataset)} GoPs")
    print(f"  Official test:  {len(official_test_loader.dataset)} GoPs (reserved)")
    print('  Checkpoint selection uses internal validation only: PASS')

    sample_gop, _ = next(iter(train_loader))
    sample_frame = sample_gop[0, 0]
    c = ratio2filtersize(sample_frame, params['ratio'])

    print(
        f"SNR={params['snr']} dB | ratio={params['ratio']:.4f} | "
        f"c={c} | channel={params['channel']}"
    )

    # Construct the model before loading its pretrained reconstruction path.
    model = VideoJSCC(
        c=c,
        channel_type=params['channel'],
        snr=params['snr'],
        n_frames=params['gop_size'],
        hidden_dim=params['hidden_dim'],
    )

    load_and_freeze_jscc(model, params['jscc_ckpt'])

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names or not all(
        name.startswith('temporal.') for name in trainable_names
    ):
        raise RuntimeError(
            'Controlled experiment requires only temporal.* parameters '
            'to be trainable'
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Loaded frozen JSCC weights from: {params['jscc_ckpt']}")
    print(f'Trainable revised-TFM parameters: {trainable_count:,}')
    print('Only temporal.* parameters are trainable: PASS')

    phaser = (
        f"RevisedTFM_{params['channel']}_c{c}"
        f"_snr{params['snr']}_ratio{params['ratio']:.4f}"
        f"_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}"
    )

    root_log_dir = os.path.join(params['out_dir'], 'logs', phaser)
    root_ckpt_dir = os.path.join(params['out_dir'], 'checkpoints', phaser)
    root_config_dir = os.path.join(params['out_dir'], 'configs', phaser)

    os.makedirs(root_ckpt_dir, exist_ok=True)
    os.makedirs(root_config_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=root_log_dir)
    writer.add_text('config', str(params))

    device_str = params['device'] if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    model = model.to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_parameters,
        lr=params['init_lr'],
        weight_decay=params['weight_decay'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params['gamma'],
        patience=params['lr_patience'],
        min_lr=params['min_lr'],
    )

    start_time = time.time()
    epoch_train_losses = []
    epoch_val_losses = []
    per_epoch_time = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(root_ckpt_dir, 'best.pkl')
    epoch = 0

    try:
        with tqdm(
            range(params['epochs']),
            disable=params['disable_tqdm'],
        ) as progress:
            for epoch in progress:
                progress.set_description(f'Epoch {epoch}')
                epoch_start = time.time()

                train_loss = train_epoch(
                    model,
                    optimizer,
                    device,
                    train_loader,
                )
                val_loss = evaluate_epoch(
                    model,
                    device,
                    val_loader,
                    channel_seed=params['seed'] + 10_000,
                )

                epoch_train_losses.append(train_loss)
                epoch_val_losses.append(val_loss)

                writer.add_scalar('train/loss', train_loss, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar(
                    'learning_rate',
                    optimizer.param_groups[0]['lr'],
                    epoch,
                )
                writer.add_scalar(
                    'temporal/alpha',
                    model.temporal.alpha.detach().item(),
                    epoch,
                )

                elapsed_epoch = time.time() - epoch_start
                per_epoch_time.append(elapsed_epoch)

                progress.set_postfix(
                    time=elapsed_epoch,
                    lr=optimizer.param_groups[0]['lr'],
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

                latest_ckpt_path = os.path.join(
                    root_ckpt_dir,
                    f'epoch_{epoch}.pkl',
                )
                torch.save(model.state_dict(), latest_ckpt_path)

                for checkpoint_file in glob.glob(
                    os.path.join(root_ckpt_dir, 'epoch_*.pkl')
                ):
                    epoch_number = int(
                        os.path.splitext(
                            os.path.basename(checkpoint_file)
                        )[0].split('_')[-1]
                    )
                    if epoch_number < epoch - 1:
                        os.remove(checkpoint_file)

                if val_loss < best_val_loss - params['min_delta']:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(
                        f'\n  Best model saved '
                        f'(internal_val_loss={best_val_loss:.6f})'
                    )
                else:
                    epochs_without_improvement += 1

                scheduler.step(val_loss)

                if (
                    epochs_without_improvement
                    >= params['early_stopping_patience']
                ):
                    print(
                        '\nEarly stopping: internal validation loss did not '
                        f"improve by at least {params['min_delta']} for "
                        f"{params['early_stopping_patience']} epochs."
                    )
                    break

                if time.time() - start_time > params['max_time'] * 3600:
                    print(
                        f"\nMax time {params['max_time']}h reached; stopping."
                    )
                    break

    except KeyboardInterrupt:
        interrupted_path = os.path.join(
            root_ckpt_dir,
            f'interrupted_epoch_{epoch}.pkl',
        )
        torch.save(model.state_dict(), interrupted_path)
        print(f'\nInterrupted checkpoint saved: {interrupted_path}')

    # Report final metrics from the checkpoint chosen only by internal val loss.
    if os.path.isfile(best_ckpt_path):
        model.load_state_dict(
            torch.load(
                best_ckpt_path,
                map_location=device,
                weights_only=True,
            ),
            strict=True,
        )

    final_val_loss = evaluate_epoch(
        model,
        device,
        val_loader,
        channel_seed=params['seed'] + 10_000,
    )
    final_train_loss = evaluate_epoch(
        model,
        device,
        train_loader,
        channel_seed=params['seed'] + 20_000,
    )
    total_hours = (time.time() - start_time) / 3600
    average_epoch_time = (
        float(np.mean(per_epoch_time)) if per_epoch_time else float('nan')
    )

    print(f'\nBest-checkpoint Train Loss : {final_train_loss:.6f}')
    print(f'Best-checkpoint Val Loss   : {final_val_loss:.6f}')
    print(f'Best Val Loss              : {best_val_loss:.6f}')
    print('Official Test Evaluation   : NOT RUN')
    print(f'Total Time                 : {total_hours:.2f}h')
    print(f'Avg Time/Epoch             : {average_epoch_time:.2f}s')

    writer.add_text(
        'result',
        (
            f"SNR={params['snr']} | ratio={params['ratio']:.4f} | c={c}\n"
            f'Best-checkpoint Train Loss: {final_train_loss:.6f}\n'
            f'Best-checkpoint Val Loss: {final_val_loss:.6f}\n'
            f'Best Val Loss: {best_val_loss:.6f}\n'
            'Official Test Evaluation: NOT RUN\n'
            f'Total Time: {total_hours:.2f}h\n'
            f'Avg Epoch: {average_epoch_time:.2f}s\n'
            f'Params: {view_model_param(model)}'
        ),
    )
    writer.close()

    config = {
        **{key: value for key, value in params.items() if key != 'device'},
        'device': device_str,
        'c': c,
        'best_val_loss': best_val_loss,
        'total_params': int(view_model_param(model)),
        'trainable_tfm_params': trainable_count,
        'dataset_protocol': {
            'official_split': 1,
            'internal_val_fraction_requested': params['val_fraction'],
            'internal_train_gops': len(train_loader.dataset),
            'internal_val_gops': len(val_loader.dataset),
            'official_test_gops_reserved': len(official_test_loader.dataset),
            'class_stratified': True,
            'ucf101_group_boundaries_respected': True,
            'group_overlap_check': 'PASS',
            'official_test_used_for_checkpoint_selection': False,
            'fixed_validation_channel_seed': params['seed'] + 10_000,
            'validation_rng_isolated_from_training': True,
        },
    }

    with open(
        os.path.join(root_config_dir, 'config.yaml'),
        'w',
        encoding='utf-8',
    ) as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

    del model
    del optimizer
    del scheduler
    del train_loader
    del val_loader
    del official_test_loader

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = config_parser()
    set_seed(args.seed)

    snr_list = list(map(float, args.snr_list))
    ratio_list = [float(Fraction(value)) for value in args.ratio_list]

    params = {
        'frames_root': args.frames_root,
        'annotation_path': args.annotation_path,
        'image_size': args.image_size,
        'gop_size': args.gop_size,
        'gops_per_clip': args.gops_per_clip,
        'val_fraction': args.val_fraction,
        'channel': args.channel,
        'snr_list': snr_list,
        'ratio_list': ratio_list,
        'hidden_dim': args.hidden_dim,
        'jscc_ckpt': args.jscc_ckpt,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'init_lr': args.init_lr,
        'weight_decay': args.weight_decay,
        'gamma': args.gamma,
        'min_lr': args.min_lr,
        'lr_patience': args.lr_patience,
        'early_stopping_patience': args.early_stopping_patience,
        'min_delta': args.min_delta,
        'max_time': args.max_time,
        'seed': args.seed,
        'out_dir': args.out,
        'device': args.device,
        'disable_tqdm': args.disable_tqdm,
    }

    for ratio in ratio_list:
        for snr in snr_list:
            run_params = dict(params)
            run_params['ratio'] = ratio
            run_params['snr'] = snr
            train_pipeline(run_params)


if __name__ == '__main__':
    main()
880746