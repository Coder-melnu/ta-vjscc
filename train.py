# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun

Modified: Added UCF101 dataset support
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import DeepJSCC, ratio2filtersize
from torch.nn.parallel import DataParallel
from utils import image_normalization, set_seed, save_model, view_model_param
from fractions import Fraction
from data.dataset import Vanilla
import numpy as np
import time
from tensorboardX import SummaryWriter
import glob


def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0

    for iter, (images, _) in enumerate(data_loader):
        images = images.cuda() if param['parallel'] and torch.cuda.device_count(
        ) > 1 else images.to(param['device'])
        optimizer.zero_grad()
        outputs = model.forward(images)
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
            images, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer


def evaluate_epoch(model, param, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for iter, (images, _) in enumerate(data_loader):
            images = images.cuda() if param['parallel'] and torch.cuda.device_count(
            ) > 1 else images.to(param['device'])
            outputs = model.forward(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
                images, outputs)
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

    return epoch_loss


def config_parser_pipeline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ucf101', type=str,
                        choices=['cifar10', 'imagenet', 'ucf101'], help='dataset')
    parser.add_argument('--out', default='./out', type=str, help='out_path')
    parser.add_argument('--disable_tqdm', default=False, type=bool, help='disable_tqdm')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--snr_list', default=['19', '13', '7', '4', '1'],
                        nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'],
                        nargs='+', help='ratio_list')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')
    # UCF101-specific arguments
    parser.add_argument('--frames_root',
                        default='/mnt/c55b31e5-e9f9-445c-ae86-96bb5a771c7b/UCF101Frames',
                        help='Path to extracted UCF101 frames')
    parser.add_argument('--annotation_path',
                        default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist',
                        help='Path to UCF101 annotation files')

    return parser.parse_args()


def main_pipeline():
    args = config_parser_pipeline()

    print("Training Start")
    dataset_name = args.dataset
    out_dir = args.out
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    params = {}
    params['disable_tqdm'] = args.disable_tqdm
    params['dataset'] = dataset_name
    params['out_dir'] = out_dir
    params['device'] = args.device
    params['snr_list'] = args.snr_list
    params['ratio_list'] = args.ratio_list
    params['channel'] = args.channel

    if dataset_name == 'cifar10':
        params['batch_size'] = 64
        params['num_workers'] = 4
        params['epochs'] = 1000
        params['init_lr'] = 1e-3
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['step_size'] = 640
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5

    elif dataset_name == 'imagenet':
        params['batch_size'] = 32
        params['num_workers'] = 4
        params['epochs'] = 300
        params['init_lr'] = 1e-4
        params['weight_decay'] = 5e-4
        params['parallel'] = True
        params['if_scheduler'] = True
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = True
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5

    elif dataset_name == 'ucf101':
        params['batch_size'] = 32
        params['num_workers'] = 4
        params['epochs'] = 100
        params['init_lr'] = 1e-3
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['step_size'] = 50
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
        params['frames_per_clip'] = 1
        params['image_size'] = 256
        params['frames_root'] = args.frames_root
        params['annotation_path'] = args.annotation_path

    else:
        raise Exception('Unknown dataset')

    set_seed(params['seed'])

    for ratio in params['ratio_list']:
        for snr in params['snr_list']:
            params['ratio'] = ratio
            params['snr'] = snr
            train_pipeline(params)


def train_pipeline(params):

    dataset_name = params['dataset']

    # load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'],
                                  num_workers=params['num_workers'])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'],
                                 num_workers=params['num_workers'])

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(
            root='../dataset/ImageNet/train', transform=transform)
        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'],
                                  num_workers=params['num_workers'])
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'],
                                 num_workers=params['num_workers'])

    elif dataset_name == 'ucf101':
        from data.ucf101_dataloader import build_dataloaders
        print("Loading UCF101 dataset...")
        train_loader, test_loader = build_dataloaders(
            frames_root=params['frames_root'],
            annotation_path=params['annotation_path'],
            mode='frame',
            image_size=params['image_size'],
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            frames_per_clip=params['frames_per_clip'],
            seed=params['seed'],
        )
        # Get one sample to compute c from ratio
        sample_batch = next(iter(train_loader))
        train_dataset = None        # not needed below
        image_first = sample_batch[0][0]   # (C, H, W)

    else:
        raise Exception('Unknown dataset')

    # create model
    if dataset_name != 'ucf101':
        image_first = train_dataset.__getitem__(0)[0]

    c = ratio2filtersize(image_first, params['ratio'])
    print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(
        params['snr'], c, params['ratio']))
    model = DeepJSCC(c=c, channel_type=params['channel'], snr=params['snr'])

    # init exp dir
    out_dir = params['out_dir']
    phaser = dataset_name.upper() + '_' + str(c) + '_' + str(params['snr']) + '_' + \
        "{:.2f}".format(params['ratio']) + '_' + str(params['channel']) + \
        '_' + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_log_dir    = out_dir + '/' + 'logs/' + phaser
    root_ckpt_dir   = out_dir + '/' + 'checkpoints/' + phaser
    root_config_dir = out_dir + '/' + 'configs/' + phaser
    writer = SummaryWriter(log_dir=root_log_dir)

    # model init
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
    else:
        model = model.to(device)

    # optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    if params['if_scheduler'] and not params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params['step_size'], gamma=params['gamma'])
    elif params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=params['lr_reduce_factor'],
            patience=params['lr_schedule_patience'], verbose=False)
    else:
        print("No scheduler")
        scheduler = None

    writer.add_text('config', str(params))
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []

    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)
                start = time.time()

                epoch_train_loss, optimizer = train_epoch(
                    model, optimizer, params, train_loader)
                epoch_val_loss = evaluate_epoch(model, params, test_loader)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start,
                              lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss,
                              val_loss=epoch_val_loss)

                per_epoch_time.append(time.time() - start)

                # Save checkpoint
                if not os.path.exists(root_ckpt_dir):
                    os.makedirs(root_ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(
                    root_ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(root_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                if params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(epoch_val_loss)
                elif params['if_scheduler'] and not params['ReduceLROnPlateau']:
                    scheduler.step()

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    test_loss  = evaluate_epoch(model, params, test_loader)
    train_loss = evaluate_epoch(model, params, train_loader)
    print("Test Loss:  {:.4f}".format(test_loss))
    print("Train Loss: {:.4f}".format(train_loss))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.add_text(tag='result', text_string="""Dataset: {}\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST Loss: {:.4f}\nTRAIN Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                    .format(dataset_name, params, view_model_param(model),
                            np.mean(np.array(train_loss)), np.mean(np.array(test_loss)),
                            epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    writer.close()

    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    with open(root_config_dir + '.yaml', 'w') as f:
        dict_yaml = {'dataset_name': dataset_name, 'params': params,
                     'inner_channel': c, 'total_parameters': view_model_param(model)}
        import yaml
        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer


if __name__ == '__main__':
    main_pipeline()