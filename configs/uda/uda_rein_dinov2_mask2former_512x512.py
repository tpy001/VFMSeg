# dataset config
_base_ = [
    "../_base_/datasets/uda_gta_to_cityscapes_512x512.py",
    "../_base_/default_runtime.py",
    #"../_base_/models/rein_dinov2_segformer.py",
    #"../_base_/models/dinov2_SegFormer_frozen.py",
    "../_base_/models/rein_dinov2_mask2former.py",


]

model = dict(
    type='DACS',
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=500,
    print_grad_magnitude=False,
)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]


train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=1000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # visualization=dict(type="SegVisualizationHook"),
)
