# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import os
import json

import mmcv
from cityscapesscripts.preparation.json2labelImg import json2labelImg
import pathlib
from mmengine.utils import track_parallel_progress,track_progress,scandir

def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args

def save_class_stats(out_dir, sample_class_stats):
    sample_class_stats = [e for e in sample_class_stats if e is not None]
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)

def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    os.makedirs(out_dir, exist_ok=True)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    gt_dir_path = pathlib.Path(gt_dir)
    for file in gt_dir_path.rglob('*_polygons.json'):
        poly_files.append(str(file))

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = track_parallel_progress(
                convert_json_to_label, poly_files, args.nproc)
        else:
            sample_class_stats = track_progress(convert_json_to_label,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)

    split_names = ['train', 'val', 'test']

    for split in split_names:
        filenames = []
        for poly in scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()
