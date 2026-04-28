import torch
from utils import get_psnr
import os
from model import DeepJSCC
from train import image_normalization
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from data.dataset import Vanilla
import yaml
from tensorboardX import SummaryWriter
import glob
from concurrent.futures import ProcessPoolExecutor
from pytorch_msssim import ms_ssim


def evaluate_epoch_with_msssim(model, param, data_loader):
    """
    Evaluate one epoch, returning both MSE loss and MS-SSIM.
    """
    model.eval()
    epoch_loss = 0
    epoch_msssim = 0
    count = 0

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

            # Clamp to [0, 255] (images are in [0,255] after denormalization)
            outputs_clamped = outputs.clamp(0, 255)
            images_clamped = images.clamp(0, 255)

            # MS-SSIM requires images >= 160x160 for 5 scales.
            # For small images (e.g. CIFAR10 32x32), fall back to 3 scales.
            h = images_clamped.shape[2]
            if h < 160:
                weights = [0.3222, 0.3363, 0.3415]  # 3-scale weights
                msssim_val = ms_ssim(images_clamped, outputs_clamped,
                                     data_range=255.0, win_size=7,
                                     weights=weights)
            else:
                msssim_val = ms_ssim(images_clamped, outputs_clamped,
                                     data_range=255.0, win_size=7)

            epoch_msssim += msssim_val.item()
            count += 1

    epoch_loss /= count
    epoch_msssim /= count
    return epoch_loss, epoch_msssim


def eval_snr(model, test_loader, writer, param, times=10):
    snr_list = range(0, 26, 1)
    for snr in snr_list:
        model.change_channel(param['channel'], snr)
        total_loss = 0
        total_msssim = 0
        for i in range(times):
            loss, msssim_val = evaluate_epoch_with_msssim(model, param, test_loader)
            total_loss += loss
            total_msssim += msssim_val
        total_loss /= times
        total_msssim /= times

        psnr = get_psnr(image=None, gt=None, mse=total_loss)
        writer.add_scalar('psnr', psnr, snr)
        writer.add_scalar('msssim', total_msssim, snr)
        print(f"  SNR {snr:>2d} dB | PSNR: {psnr:.2f} dB | MS-SSIM: {total_msssim:.4f}")


def process_config(config_path, output_dir, dataset_name, times):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        assert dataset_name == config['dataset_name']
        params = config['params']
        c = config['inner_channel']

    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])
    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])
    elif dataset_name == 'ucf101':
        from data.ucf101_dataloader import build_dataloaders
        _, test_loader = build_dataloaders(
            frames_root='/mnt/c55b31e5-e9f9-445c-ae86-96bb5a771c7b/UCF101Frames',
            annotation_path='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist',
            mode='frame',
            image_size=256,
            batch_size=params['batch_size'],
            num_workers=params['num_workers'],
            frames_per_clip=5,
            seed=42,
        )
    else:
        raise Exception('Unknown dataset')

    name = os.path.splitext(os.path.basename(config_path))[0]
    writer = SummaryWriter(os.path.join(output_dir, 'eval', name))

    model = DeepJSCC(c=c)
    model = model.to(params['device'])
    pkl_list = glob.glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
    model.load_state_dict(torch.load(pkl_list[-1]))

    print(f"\nEvaluating: {name}")
    eval_snr(model, test_loader, writer, params, times)
    writer.close()


def main():
    times = 10
    dataset_name = 'ucf101'
    output_dir = '/mnt/c55b31e5-e9f9-445c-ae86-96bb5a771c7b/ta-vjscc-out'
    channel_type = 'Rayleigh'

    config_dir = os.path.join(output_dir, 'configs')
    config_files = [os.path.join(config_dir, name) for name in os.listdir(config_dir)
                    if (dataset_name in name or dataset_name.upper() in name)
                    and channel_type in name and name.endswith('.yaml')]

    with ProcessPoolExecutor() as executor:
        executor.map(process_config, config_files,
                     [output_dir] * len(config_files),
                     [dataset_name] * len(config_files),
                     [times] * len(config_files))


if __name__ == '__main__':
    main()