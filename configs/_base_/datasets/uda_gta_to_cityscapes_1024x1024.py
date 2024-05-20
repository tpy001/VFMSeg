_base_ = [
    "./gta_1024x1024.py",
    "./cityscapes_1024x1024.py",
]

uda_dataset_train = dict(
    type="UDADataset",
    source = {{_base_.train_gta}},
    target = {{_base_.train_cityscapes}},
    rare_class_sampling=dict(
        class_temp=0.01,
        min_crop_ratio=2.0,
        min_pixels=3000,    
    )
)

uda_dataset_val = {{_base_.val_cityscapes}}

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=uda_dataset_train,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset= {{_base_.val_cityscapes}},
)

test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys"]
)
test_evaluator=val_evaluator
