ann_file_test = 'data/ucf101/ucf101_val_split_1_videos.txt'
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
auto_scale_lr = dict(base_batch_size=240, enable=False)
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
dataset_type = 'VideoDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    backbone=dict(
        act_cfg=dict(type='ReLU'),
        conv_cfg=dict(type='Conv3d'),
        dropout_ratio=0.5,
        init_std=0.005,
        norm_cfg=None,
        pretrained=
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',
        style='pytorch',
        type='C3D'),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=4096,
        init_std=0.01,
        num_classes=101,
        spatial_type=None,
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
        mean=[
            104,
            117,
            128,
        ],
        std=[
            1,
            1,
            1,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer3D')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0005))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=45,
        gamma=0.1,
        milestones=[
            20,
            40,
        ],
        type='MultiStepLR'),
]
resume = False
split = 1
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/ucf101/ucf101_val_split_1_videos.txt',
        data_prefix=dict(video='data/ucf101/videos'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=1,
                num_clips=10,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
train_cfg = dict(
    max_epochs=45, type='EpochBasedTrainLoop', val_begin=1, val_interval=5)
train_dataloader = dict(
    batch_size=30,
    dataset=dict(
        ann_file='data/ucf101/ucf101_train_split_1_videos.txt',
        data_prefix=dict(video='data/ucf101/videos'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(size=112, type='RandomCrop'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=16, frame_interval=1, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(size=112, type='RandomCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=30,
    dataset=dict(
        ann_file='data/ucf101/ucf101_val_split_1_videos.txt',
        data_prefix=dict(video='data/ucf101/videos'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=16,
                frame_interval=1,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                128,
            ), type='Resize'),
            dict(crop_size=112, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        128,
    ), type='Resize'),
    dict(crop_size=112, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
