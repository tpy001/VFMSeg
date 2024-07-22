
crop_size = (1024, 1024)
num_classes = 19
model = dict(
    type="MsVFMEncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=crop_size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),
    backbone=dict(
        type="LoRABackbone",
        backbone=dict(
            type='DinoVisionTransformer',
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            img_size=512,
            ffn_layer="mlp",
            init_values=1e-05,
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        ),
        checkpoint='checkpoints/dinov2_converted.pth',
        Lora_config=dict(
            r=32,
            lora_alpha=32,
            target_modules=['qkv'],
            lora_dropout = 0.1
        ),
    ),
    decode_head=dict(
        type='LinearHead',
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="GN", num_groups=32),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    aux_head=dict(
        type='VFMHead',
        transformer = dict(
            type='MaskTransformerDecoder',
            query_dim = 256,
            n_heads = 8,
            d_head = 64,
            depth = 3,
            dropout = 0.1,
            mask_ratio = 0.2,
        ),
        in_channels=[1024, 1024, 1024, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type="GN", num_groups=32),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    detail_loss = 0.5,
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=32,
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        stride=[320, 320],
        crop_size=[512, 512]
    )
)
