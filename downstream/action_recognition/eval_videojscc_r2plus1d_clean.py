#!/usr/bin/env python
"""Evaluation-only VideoJSCC + locked R(2+1)D-18 pipeline."""

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
from tqdm import tqdm

from data.ucf101_train_val_test import build_train_val_dataloaders
from downstream.action_recognition.models.r2plus1d_recognizer import (
    R2Plus1DRecognizer, load_r2plus1d_checkpoint, normalization_tensors,
)
from model.video_jscc import VideoJSCC

EXPECTED_MANIFEST_SHA256 = 'ea27a8557ef1e8f658f63c8f86f33721d6f94a1e58c657fa0a02c93e94befb7c'


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videojscc_ckpt',required=True)
    p.add_argument('--r2plus1d_ckpt',default='downstream/action_recognition/weights/r2plus1d_ucf101_layer4_locked_split_best.pt')
    p.add_argument('--frames_root',default='datasets/UCF101Frames')
    p.add_argument('--annotation_path',default='datasets/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist')
    p.add_argument('--image_size',type=int,default=128); p.add_argument('--gop_size',type=int,default=5)
    p.add_argument('--gops_per_clip',type=int,default=1); p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--num_workers',type=int,default=4); p.add_argument('--split_seed',type=int,default=42)
    p.add_argument('--channel_seed',type=int,default=1042); p.add_argument('--device',default='cuda:0')
    p.add_argument('--out',default='./out_video_clean_evaluation_locked_r2plus1d')
    p.add_argument('--smoke_batches',type=int,default=0)
    return p.parse_args()


def load_videojscc(path,device):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(checkpoint,dict) or 'model_state' not in checkpoint or 'config' not in checkpoint:
        raise KeyError('Expected clean VideoJSCC checkpoint with model_state and config')
    config=checkpoint['config']
    model=VideoJSCC(c=int(config['c']),channel_type=config['channel'],snr=float(config['snr']),
                    n_frames=int(config['gop_size']),hidden_dim=int(config['hidden_dim']))
    model.load_state_dict(checkpoint['model_state'],strict=True); model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model,checkpoint


def load_recognizer(path,device):
    model=R2Plus1DRecognizer(pretrained=False,num_classes=101)
    metadata=load_r2plus1d_checkpoint(model,path)
    model.to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model,metadata


@contextmanager
def deterministic_channel(seed,device):
    devices=[device.index if device.index is not None else 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type=='cuda': torch.cuda.manual_seed_all(seed)
        yield


@torch.no_grad()
def evaluate(videojscc,recognizer,loader,device,channel_seed,limit=0):
    rows=[]; mean,std=normalization_tensors(device); clip_id=0
    with deterministic_channel(channel_seed,device):
        for batch_index,(gops,labels) in enumerate(tqdm(loader,desc='evaluation')):
            if limit and batch_index>=limit: break
            gops,labels=gops.to(device),labels.to(device)
            reconstructed_raw=videojscc(gops); reconstructed=reconstructed_raw.clamp(0,1)
            batch,frames=gops.shape[:2]; flat_gt=gops.flatten(0,1); flat_pred=reconstructed.flatten(0,1)
            frame_mse=(flat_pred-flat_gt).square().mean(dim=(1,2,3))
            frame_psnr=-10.0*torch.log10(frame_mse.clamp_min(1e-12))
            frame_ssim=ssim(flat_pred,flat_gt,data_range=1.0,size_average=False)
            frame_ms=ms_ssim(flat_pred,flat_gt,data_range=1.0,size_average=False,win_size=7,weights=(0.3,0.3,0.4))
            logits_recon=recognizer((reconstructed-mean)/std); logits_clean=recognizer((gops-mean)/std)
            top5_recon=logits_recon.topk(5,dim=1).indices; top5_clean=logits_clean.topk(5,dim=1).indices
            raw_mse=(reconstructed_raw-gops).square().mean(dim=(1,2,3,4))
            for item in range(batch):
                start,end=item*frames,(item+1)*frames
                rows.append({'clip_id':clip_id,'label':int(labels[item]),
                    'raw_reconstruction_mse':float(raw_mse[item]),'psnr_db':float(frame_psnr[start:end].mean()),
                    'ssim':float(frame_ssim[start:end].mean()),'ms_ssim_3scale':float(frame_ms[start:end].mean()),
                    'reconstructed_top1_correct':int(top5_recon[item,0]==labels[item]),
                    'reconstructed_top5_correct':int(top5_recon[item].eq(labels[item]).any()),
                    'clean_top1_correct':int(top5_clean[item,0]==labels[item]),
                    'clean_top5_correct':int(top5_clean[item].eq(labels[item]).any())})
                clip_id+=1
    return rows


def aggregate(rows):
    if not rows: raise RuntimeError('Evaluator produced no rows')
    summary={'clips':len(rows)}
    for field in ('raw_reconstruction_mse','psnr_db','ssim','ms_ssim_3scale'):
        summary[field]=sum(row[field] for row in rows)/len(rows)
    for prefix in ('reconstructed','clean'):
        summary[f'{prefix}_top1_percent']=100*sum(row[f'{prefix}_top1_correct'] for row in rows)/len(rows)
        summary[f'{prefix}_top5_percent']=100*sum(row[f'{prefix}_top5_correct'] for row in rows)/len(rows)
    return summary


def main():
    args=parse_args()
    if (args.image_size,args.gop_size,args.gops_per_clip,args.split_seed)!=(128,5,1,42):
        raise ValueError('Locked protocol requires image_size=128, gop_size=5, gops_per_clip=1, split_seed=42')
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    _,loader,manifest,manifest_hash=build_train_val_dataloaders(
        frames_root=args.frames_root,annotation_path=args.annotation_path,image_size=args.image_size,
        batch_size=args.batch_size,num_workers=args.num_workers,gop_size=args.gop_size,
        gops_per_clip=args.gops_per_clip,val_fraction=0.1,seed=args.split_seed,enforce_locked_counts=True)
    if len(loader.dataset)!=1128 or manifest_hash!=EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f'Locked validation mismatch: clips={len(loader.dataset)}, manifest={manifest_hash}')
    videojscc,checkpoint=load_videojscc(args.videojscc_ckpt,device)
    recognizer,recognizer_metadata=load_recognizer(args.r2plus1d_ckpt,device)
    for name,source_hash in [('VideoJSCC',checkpoint.get('split_manifest_sha256')),
                             ('R(2+1)D',recognizer_metadata.get('split_manifest_sha256'))]:
        if source_hash!=manifest_hash: raise RuntimeError(f'{name} checkpoint manifest does not match evaluation manifest')
    video_before={n:v.detach().cpu().clone() for n,v in videojscc.state_dict().items()}
    recognizer_before={n:v.detach().cpu().clone() for n,v in recognizer.state_dict().items()}
    rows=evaluate(videojscc,recognizer,loader,device,args.channel_seed,args.smoke_batches)
    if any(not torch.equal(video_before[n],v.detach().cpu()) for n,v in videojscc.state_dict().items()):
        raise RuntimeError('VideoJSCC changed during evaluation')
    if any(not torch.equal(recognizer_before[n],v.detach().cpu()) for n,v in recognizer.state_dict().items()):
        raise RuntimeError('R(2+1)D changed during evaluation')
    config=checkpoint['config']; run_name=(f"VideoJSCC_R2Plus1D_validation_{config['channel']}_c{config['c']}"
                                          f"_snr{float(config['snr']):g}_seed{args.channel_seed}")
    output=Path(args.out)/run_name; output.mkdir(parents=True,exist_ok=False)
    with (output/'per_clip_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2)); (output/'split_manifest.sha256').write_text(manifest_hash+'\n')
    evaluation_config={**vars(args),'actual_device':str(device),'architecture':'torchvision_r2plus1d_18',
        'videojscc_checkpoint_sha256':hashlib.sha256(Path(args.videojscc_ckpt).read_bytes()).hexdigest(),
        'r2plus1d_checkpoint_sha256':hashlib.sha256(Path(args.r2plus1d_ckpt).read_bytes()).hexdigest(),
        'videojscc_split_manifest_sha256':checkpoint['split_manifest_sha256'],
        'evaluation_split_manifest_sha256':manifest_hash,'r2plus1d_metadata':recognizer_metadata,
        'created_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'pytorch':torch.__version__,'parameter_updates':0,'official_test_used':False}
    (output/'evaluation_config.json').write_text(json.dumps(evaluation_config,indent=2,default=str))
    summary={**aggregate(rows),'split':'validation','channel_seed':args.channel_seed,'parameter_updates':0,
             'videojscc_training_epoch':checkpoint['epoch'],'official_test_used':False,
             'diagnostic_smoke_test':bool(args.smoke_batches)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(f'Saved: {output}')


if __name__=='__main__': main()
