_base_ = [
    'daformer_sepaspp_mitb5.py',
]

crop_size = (1024, 1024)
num_classes = 19
model = dict(
    type="FrozenHRDAEncoderDecoder",
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
        _delete_=True,
        type="DinoVisionTransformer",
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
        type='HRDAHead',
        single_scale_head='DAFormerHead',
        attention_classwise=True,
        hr_loss_weight=0.1,
    ),
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=False,   
    train_cfg=dict(),
    test_cfg=dict(
        # _delete_=True,
        mode='slide',
        batched_slide=False, # no improvement hardlly, both speed and acc 
        stride=[682, 682],
        crop_size=[1024, 1024]
        )
)
