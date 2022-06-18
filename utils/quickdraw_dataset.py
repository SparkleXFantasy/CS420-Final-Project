from torch.utils.data import Dataset
import h5py
import numpy as np
import os.path as osp
import pickle
from PIL import Image
import torch
from torchvision import transforms


class QuickDrawDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.data = None

        with open(osp.join(root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']
            self.indices = saved_pkl['indices'][self.mode_indices[mode]]

        print('[*] Created a new {} dataset: {}'.format(mode, root_dir))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(osp.join(self.root_dir, 'quickdraw_{}.hdf5'.format(self.mode)), 'r')

        index_tuple = self.indices[idx]
        cid = index_tuple[0]
        sid = index_tuple[1]
        sketch_path = '/sketch/{}/{}'.format(cid, sid)

        sid_points = np.array(self.data[sketch_path][()], dtype=np.float32)
        sample = {'points3': sid_points, 'category': cid}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        if self.data is not None:
            self.data.close()

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDraw-{}'.format(self.mode)


class QuickDrawVisualDataset(Dataset):
    mode_indices = {'train': 0, 'valid': 1, 'test': 2}

    def __init__(self, root_dir, mode):
        self.root_dir = root_dir
        self.mode = mode
        self.datalen = [3000, 1000, 1000]
        self.categories = None
        # self.data_category_index = 0
        # self.data_item_index = 0
        # self.data_np = None
        self.dataset_list = []
        self.trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ]
        )

        with open(osp.join(root_dir, 'categories.pkl'), 'rb') as fh:
            saved_pkl = pickle.load(fh)
            self.categories = saved_pkl['categories']

        print('[*] Created a new {} dataset: {}'.format(mode, root_dir))

        for category in self.categories:
            self.dataset_list.append(np.load(osp.join(self.root_dir, '{}_png.npz'.format(category)))[self.mode])

    def __len__(self):
        return self.datalen[self.mode_indices[self.mode]] * len(self.categories)

    def __getitem__(self, idx):
        data_category_index = idx // self.datalen[self.mode_indices[self.mode]]
        data_item_index = idx % self.datalen[self.mode_indices[self.mode]]
        image = self.dataset_list[data_category_index][data_item_index]
        pil_img = Image.fromarray(image)
        resized_img = self.trans(pil_img)
        resized_img = np.array(resized_img)
        sample = {'image': resized_img, 'category': data_category_index}
        return sample

    def __del__(self):
        self.dispose()

    def dispose(self):
        # if self.data is not None:
        #     self.data.close()
        pass

    def num_categories(self):
        return len(self.categories)

    def get_name_prefix(self):
        return 'QuickDrawVisual-{}'.format(self.mode)
