""" convert sequence data to png file """
import os
import glob
from PIL import Image
import re
from warnings import simplefilter
import yaml
import argparse
import numpy as np
import os.path
import torch
import tqdm
import warnings
import pickle
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from torch.utils.data import DataLoader
from neuralline.rasterize import Raster
import torchvision
from utils.quickdraw_dataset import QuickDrawDataset
import launcher.pytorch_util as ptu


simplefilter(action='ignore', category=DeprecationWarning)


def train_data_collate(batch):
    length_list = [len(item['points3']) for item in batch]
    max_length = max(length_list)

    points3_padded_list = list()
    category_list = list()
    for item in batch:
        points3 = item['points3']
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32)
        points3_padded[0:points3_length, :] = points3
        points3_padded_list.append(points3_padded)

        category_list.append(item['category'])

    batch_padded = {
        'points3': points3_padded_list,
        'points3_length': length_list,
        'category': category_list
    }

    sort_indices = np.argsort(-np.array(length_list))
    batch_collate = dict()
    for k, v in batch_padded.items():
        sorted_arr = np.array([v[idx] for idx in sort_indices])
        batch_collate[k] = torch.from_numpy(sorted_arr)
    return batch_collate


class SketchR2CNNTrain(object):

    def __init__(self, exp_specs):
        config = exp_specs

        if config['log_dir'] is None:
            raise Exception('No log_dir specified!')
        else:
            if not os.path.exists(config['log_dir']):
                os.makedirs(config['log_dir'], 0o777)

        if config['dataset_root'] is None:
            raise Exception('No dataset_root specified!')

        self.config = config

        self.modes = ['train', 'valid', 'test']
        self.step_counters = {m: 0 for m in self.modes}

        self.device = exp_specs['device']
        print('[*] Using device: {}'.format(self.device))
        with open(os.path.join(config['dataset_root'], 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def create_data_loaders(self, dataset_dict):
        data_loaders = {
            m: DataLoader(dataset_dict[m],
                          batch_size=self.config['batch_size'],
                          num_workers=3,
                          shuffle=False,
                          drop_last=True,
                          collate_fn=train_data_collate,
                          pin_memory=True) for m in self.modes
        }
        return data_loaders

    def forward_batch(self, data_batch, mode, categorys=None, category_num=None):
        points = data_batch['points3'].to(self.device)
        cid = data_batch['category'].to(self.device)
        imgsize = self.config['imgsize']
        thickness = self.config['thickness']

        png_path = os.path.join(self.config['log_dir'], 'png')
        if os.path.exists(png_path) is False:
            os.makedirs(png_path)

        images = Raster.to_image(points, 1.0, imgsize, thickness, device=self.device)
        images = images.cpu()
        for i in range(images.shape[0]):
            img = images[i].clone().unsqueeze(0)
            category = categorys[cid[i]]
            save_path = os.path.join(png_path, category, mode)
            if os.path.exists(save_path) is False:
                os.makedirs(save_path)
            idx = category_num[mode][cid[i]]
            category_num[mode][cid[i]] = idx + 1
            torchvision.utils.save_image(img, os.path.join(save_path, '{}.png'.format(idx)))

    def run(self):
        dataset_root = self.config['dataset_root']

        train_data = {
            m: QuickDrawDataset(dataset_root, m) for m in self.modes
        }
        num_categories = train_data[self.modes[0]].num_categories()

        print('[*] Number of categories:', num_categories)

        data_loaders = self.create_data_loaders(train_data)

        category_num = {m: [0 for _ in range(num_categories)] for m in self.modes}

        for mode in self.modes:
            pbar = tqdm.tqdm(total=len(data_loaders[mode]))
            for bid, data_batch in enumerate(data_loaders[mode]):
                self.step_counters[mode] += 1
                self.forward_batch(data_batch, mode, categorys=self.categories, category_num=category_num)
                pbar.update()
            pbar.close()

        for m in self.modes:
            train_data[m].dispose()
        return


def sort_paths(paths):
    idxs = []
    for path in paths:
        idxs.append(int(re.findall(r'\d+', path)[-1]))

    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            if idxs[i] > idxs[j]:
                tmp = idxs[i]
                idxs[i] = idxs[j]
                idxs[j] = tmp

                tmp = paths[i]
                paths[i] = paths[j]
                paths[j] = tmp
    return paths


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file", default="scripts/default.yaml")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        exp_specs = yaml.load(spec_file, Loader=yaml.Loader)

    if exp_specs["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)
    else:
        device = "cpu"
    exp_specs['device'] = device

    with SketchR2CNNTrain(exp_specs) as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()



