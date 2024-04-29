# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


'''CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')'''

# 按一定的比例混淆源域图片和目标域图片，这里源域图片的比例在30%-60%之间, 随机类别混淆
def get_group_masks(labels):
    chance = 10
    class_masks = []
    class_pixel_counts = {}
    min_crop_ratio = 0.3
    max_crop_ratio = 0.6
    for label in labels:
        for j in range(chance): # 最多循环10次
            unique_classes = torch.unique(label)
            # print(" unique class is " + str(unique_classes.to('cpu')))
            num_class = unique_classes.shape[0]
            total_pixel = label.shape[1] * label.shape[2]
            for class_label in unique_classes:
                class_mask = torch.eq(label, class_label).int()
                pixel_count = torch.sum(class_mask)
                class_pixel_counts[class_label.item()] = ( pixel_count / total_pixel).item()
            class_choice = []

            cur_ratio = 0
            ratio_sum = 0
            import random
            class_shuffle = list(range(num_class))
            random.shuffle(class_shuffle)
            for i in class_shuffle:
                class_index = unique_classes[i].item()  
                cur_ratio = class_pixel_counts[ class_index ]
                if (ratio_sum + cur_ratio) > max_crop_ratio:
                    break
                ratio_sum = ratio_sum + cur_ratio
                class_choice.append(class_index)
            if (ratio_sum < min_crop_ratio and j < (chance - 1)):
                continue
            else:
                # print(" class_choice is " + str(class_choice))
                # print("crop ration is " + str(ratio_sum))
                class_choice = torch.tensor(class_choice).to(label.device)
                class_masks.append(generate_class_mask(label, class_choice).unsqueeze(0))
                break
    return class_masks

"""def get_group_masks(labels):
    class_masks = []
    group =[
        [0,1],[2,3,4],[2,5,6,7],[8,9,10],[11,12,17,18],[13,14,15,16]
    ]
    '''group =[
        [12],[9],[5],[4],[17],[18],[16],[3],[6],[7],    # 这种基于优先级的策略肯定不行
        [11],[1],[14],[15],
        [8],[13],[2],[10],[0]
    ]'''
    for label in labels:
        classes = torch.unique(labels)
        classes_cpu = classes.to('cpu').numpy()
        nclasses = classes.shape[0]
        class_choice = []
        # start = np.random.randint(0,len(group))
        start = 0
        for i in range(start,start + len(group)):
            index = i if i < len(group) else i - len(group)
            curr_choice = set(group[index]).intersection(set(classes_cpu))   # 求交集
            if ( len(set(class_choice).union(set (curr_choice) ) ) >  int((nclasses + nclasses % 2) / 2) ):
                class_choice = set(class_choice).union(set (curr_choice) )            # 求并集
                break
            class_choice = set(class_choice).union(set (curr_choice) )            # 求并集
        class_choice = torch.tensor(list(class_choice)).to(classes.device)
        class_masks.append(generate_class_mask(label, class_choice).unsqueeze(0))
    return class_masks"""

def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(label)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target
