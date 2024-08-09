
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
        type='LoRABackbone',
        Lora_config=dict(
            lora_alpha=32,
            lora_dropout=0.1,
            r=32,
            target_modules=['q_proj','k_proj','v_proj','attn.proj']
                            # 'attn.proj']
        ),
        checkpoint='checkpoints/EVA02/eva02_L_converted.pth',
        backbone=dict(
            type='EVA2',
            depth=24,
            drop_path_rate=0.1,
            embed_dim=1024,
            img_size=512,
            in_chans=3,
            init_values=None,
            intp_freq=True,
            mlp_ratio=2.6666666666666665,
            naiveswiglu=True,
            norm_layer=dict(eps=1e-06, requires_grad=True, type='LN'),
            num_heads=16,
            out_indices=[7,11,15,23,],
            patch_size=16,
            pt_hw_seq_len=16,
            qkv_bias=True,
            rope=True,
            subln=True,
            use_abs_pos_emb=True,
            use_checkpoint=False,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            xattn=True
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
    detail_loss = 1.0,
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=32,
    train_cfg=dict(),
    test_cfg=dict(
        mode='ms_slide_inference',
        threadshod = 0.968,
        conf = 0.8,
        stride=[320, 320],
        crop_size=[512, 512]
    )
)