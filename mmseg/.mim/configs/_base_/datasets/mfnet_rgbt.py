# dataset settings
dataset_type = 'MFnet_rgbt'
data_root = 'data/MFnet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],meanIR=114.495, std=[58.395, 57.12, 57.375],stdIR= 57.63, to_rgbt=True)
crop_size = (480, 640)    
train_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='pillow',color_type='unchanged'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(480, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='NormalizeRGBT', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='pillow',color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='NormalizeRGBT', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile',imdecode_backend='pillow',color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(480, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[

            dict(type='NormalizeRGBT', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/rgbt/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/rgbt/val',
        ann_dir='ann_dir/val',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        
        img_dir='img_dir/rgbt/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline))
    # test_day=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     img_dir='img_dir/rgb/test_day',
    #     ann_dir='ann_dir/test_day',
    #     pipeline=test_pipeline))