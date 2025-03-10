_base_ = [
    "../../_base_/datasets/gta_1024x1024.py",
    "../../_base_/datasets/bdd100k_1024x1024.py",
    "../../_base_/datasets/cityscapes_1024x1024.py",
    "../../_base_/datasets/mapillary_1024x1024.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="DGDataset",
        source = {{_base_.train_gta}},
        rare_class_sampling=dict(
            class_temp=0.01,
            min_crop_ratio=2,
            min_pixels=3000,    
        ),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_cityscapes}},
            {{_base_.val_bdd}},
            {{_base_.val_mapillary}},
        ],
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
)
test_evaluator=val_evaluator


