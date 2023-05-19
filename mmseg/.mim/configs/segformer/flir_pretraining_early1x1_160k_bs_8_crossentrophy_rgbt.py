norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='EarlyFusionMixVisionTransformer',
        in_channels_ir=1,
        in_channels_rgb=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg= None),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'flir_rgbt'
data_root = 'data/flir'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    meanIR=114.495,
    std=[58.395, 57.12, 57.375],
    stdIR=57.63,
    to_rgbt=True)
crop_size = (480, 640)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        imdecode_backend='pillow',
        color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='NormalizeRGBT',
        mean=[123.675, 116.28, 103.53],
        meanIR=114.495,
        std=[58.395, 57.12, 57.375],
        stdIR=57.63,
        to_rgbt=True),
    dict(type='Pad', size=(480, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]




data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='flir_rgbt',
        data_root='data/flir',
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                imdecode_backend='pillow',
                color_type='unchanged'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(640,512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(640,512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='NormalizeRGBT',
                mean=[123.675, 116.28, 103.53],
                meanIR=114.495,
                std=[58.395, 57.12, 57.375],
                stdIR=57.63,
                to_rgbt=True),
            dict(type='Pad', size=(640,512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
        val=dict(
                type='flir_rgbt',
                data_root='data/flir',
                img_dir='img_dir/val',
                ann_dir='ann_dir/val',
                pipeline=[
                    dict(
                        type='LoadImageFromFile',
                        imdecode_backend='pillow',
                        color_type='unchanged'),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(480, 640),
                        flip=False,
                        transforms=[
                            dict(
                                type='NormalizeRGBT',
                                mean=[123.675, 116.28, 103.53],
                                meanIR=114.495,
                                std=[58.395, 57.12, 57.375],
                                stdIR=57.63,
                                to_rgbt=True),
                            dict(type='ImageToTensor', keys=['img']),
                            dict(type='Collect', keys=['img'])
                        ])
        ]))

log_config = dict(
    interval=637,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=True)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer')
#optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=2)
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))

optimizer_config=dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=50)
evaluation = dict(interval=637, metric='mIoU', pre_eval=True, save_best='mIoU')
#checkpoint = 'earlyfuseformer_pretrained.pth'
work_dir = './work_dirs/flir_pretraining_early1x1_160k_bs_8_crossentrophy_rgbt'
#gpu_ids = [1]
auto_resume = False

