
crop_size = (1024, 1024)
num_classes = 19
model = dict(
    type="MultiScaleEncoderDecoder",
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
        type="ReinsDinoVisionTransformer",
        reins_config=dict(
            type="LoRAReins",
            token_length=100,
            embed_dims=1024,
            num_layers=24,
            patch_size=16,
            link_token_to_query=False,
            lora_dim=16,
        ),
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
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2_converted.pth",
        ),
    ),
    decode_head=dict(
        type='DINOhead',
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
