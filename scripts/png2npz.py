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
import tqdm
import warnings
import pickle
import launcher.pytorch_util as ptu
from multiprocessing import Pool


simplefilter(action='ignore', category=DeprecationWarning)

def worker(img_path):
    img = Image.open(img_path, 'r').convert('L')  # covert to grayscale
    img_data = np.array(img)
    img_data = img_data.reshape([1, img_data.shape[0], img_data.shape[1]])
    return img_data


class png2npz(object):

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

        with open(os.path.join(config['dataset_root'], 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def run(self):
        dataset_root = self.config['dataset_root']
        pbar = tqdm.tqdm(total=len(self.categories))
        for category in self.categories:
            png_path = os.path.join(dataset_root, 'png', category)
            out_path = os.path.join(dataset_root, 'npz', category)
            for mode in self.modes:
                # save png data
                p = Pool(exp_specs['num_workers'])
                rets = []
                image_paths = glob.glob(os.path.join(png_path, mode, '*.png'))
                image_paths = sort_paths(image_paths)
                print(len(image_paths))
                for img_path in image_paths:
                    ret = p.apply_async(worker, img_path)
                    rets.append(ret)
                pngs = []
                for ret in rets:
                    pngs.append(ret.get())
                pngs = np.stack(pngs, axis=0)

                if mode == 'train':
                    train_pngs = pngs
                elif mode == 'valid':
                    valid_pngs = pngs
                elif mode == 'test':
                    test_pngs = pngs
            np.savez(out_path + "_png", train=train_pngs, valid=valid_pngs, test=test_pngs)
            pbar.update()
        pbar.close()



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

    with png2npz(exp_specs) as app:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            app.run()



