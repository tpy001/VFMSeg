
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
            type="SAMViT",
            img_size=512,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
            out_indices=[7, 15, 23, 31],
            window_size=14,
            use_rel_pos=True,
        ),
        checkpoint='checkpoints/SAM/sam_vit_h_converted.pth',
        Lora_config=dict(
            r=32,
            lora_alpha=32,
            # target_modules=['qkv','lin1','lin2'],
            target_modules=['qkv'],
            lora_dropout = 0.1
        ),
    ),
    decode_head=dict(
        type='LinearHead',
        in_channels=[1280, 1280, 1280, 1280],
        in_index=[0, 1, 2, 3],
        channels=320,
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
