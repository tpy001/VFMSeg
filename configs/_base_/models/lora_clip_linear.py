
crop_size = (512, 512)
num_classes = 19
model = dict(
    type="EncoderDecoder",
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
            type="CLIPVisionTransformer",
            patch_size=16,
            width=1024,
            output_dim=512,
            get_embeddings=False,
            drop_path_rate=0.1,
            layers=24,
            input_resolution=512,
            style="pytorch",
            out_indices=[7, 11, 15, 23],
            heads=16,
        ),
        checkpoint='checkpoints/CLIP/CLIP-ViT-L-converted.pt',
        Lora_config=dict(
            r=32,
            lora_alpha=32,
            target_modules=['out_proj','mlp.c_fc','mlp.c_proj'],
            # target_modules=['out_proj'],
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
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide',
        stride=[320, 320],
        crop_size=[512, 512]
        )
)
