_base_ = [
    "../../_base_/datasets/gta_512x512.py",
    "../../_base_/datasets/bdd100k_512x512.py",
    "../../_base_/datasets/cityscapes_512x512.py",
    "../../_base_/datasets/mapillary_512x512.py",
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="DGDataset",
        source = {{_base_.train_cityscapes}},
        rare_class_sampling=dict(
            class_temp=0.01,
            min_crop_ratio=0.5,
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
