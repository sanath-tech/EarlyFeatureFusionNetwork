# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type="MixVisionTransformer",
        in_channels=3,
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
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                loss_name="loss_ce",
                use_sigmoid=False,
                loss_weight=1.0,
                # LASnet used this class weight for mfnet
                class_weight=[
                    1.5105,
                    16.6591,
                    29.4238,
                    34.6315,
                    40.0845,
                    41.4357,
                    47.9794,
                    45.3725,
                    44.9000,
                ],
            ),
            dict(
                type="LovaszLoss",
                loss_name="loss_lovasz",
                loss_weight=1.0,
                reduction="none",
            ),
        ],
        # loss_decode=dict(type="FocalLoss", use_sigmoid=True, loss_weight=1.0),
    ),
    # loss_decode=dict(type='FocalTverskyLoss',alpha=0.3,beta=0.7,gamma=0.75,loss_weight=1.0)),
    # loss_decode=dict(
    # type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
