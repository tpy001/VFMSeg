# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import os.path as osp

import mmcv
import numpy as np
import torch

from mmseg.datasets import CityscapesDataset
from mmseg.registry import DATASETS

def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class DGDataset(object):

    def __init__(self, source, **cfg):
        self.source = DATASETS.build(source)
        self.ignore_index = self.source.ignore_index
        self.CLASSES = self.source.METAINFO['classes']
        self.PALETTE = self.source.METAINFO['palette']
        
        if 'rare_class_sampling' in cfg:
            rcs_cfg = cfg['rare_class_sampling']
        else:
            rcs_cfg = None
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                source['data_root'], self.rcs_class_temp)

            with open(
                    osp.join(source['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}

            for i,item in enumerate(self.source.data_list):
                img_name = item['seg_map_path'].split('/')[-1]
                self.file_to_idx[img_name] = i

    
    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['data_samples'].gt_sem_seg.data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]
        return s1

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            return self.source[idx]

    def __len__(self):
        return len(self.source)
    


@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, source, target, **cfg):
        self.source = DATASETS.build(source)
        self.target = DATASETS.build(target)
        self.ignore_index = self.target.ignore_index
        self.CLASSES = self.target.METAINFO['classes']
        self.PALETTE = self.target.METAINFO['palette']
        assert self.target.ignore_index == self.source.ignore_index
        assert self.target.METAINFO['classes'] == self.source.METAINFO['classes']
        assert self.target.METAINFO['palette'] == self.source.METAINFO['palette']

        if 'rare_class_sampling' in cfg:
            rcs_cfg = cfg['rare_class_sampling']
        else:
            rcs_cfg = None
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                source['data_root'], self.rcs_class_temp)

            with open(
                    osp.join(source['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}

            for i,item in enumerate(self.source.data_list):
                img_name = item['seg_map_path'].split('/')[-1]
                self.file_to_idx[img_name] = i

    
    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['data_samples'].gt_sem_seg.data == c)
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]

        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]
        return {
                'img':s1,
                'target_img':s2
            }

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            return {
                'img':self.source[idx % self.source.__len__()],
                'target_img':self.target[idx % self.target.__len__()]
            }
            # return self.source[idx % self.source.__len__()]

    def __len__(self):
        return len(self.source) * len(self.target)
        # return len(self.source)

