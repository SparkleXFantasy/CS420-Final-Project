import argparse
import json
import numpy as np
import os.path
import random
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from .quickdraw_dataset import QuickDrawDataset
from launcher import logger
from models.modelzoo import CNN_MODELS, CNN_IMAGE_SIZES
from models.sketch_r2cnn import SketchR2CNN


def train_data_collate(batch):
    length_list = [len(item['points3']) for item in batch]
    max_length = max(length_list)

    points3_padded_list = list()
    intensities_list = list()
    category_list = list()
    for item in batch:
        points3 = item['points3']
        points3_length = len(points3)
        points3_padded = np.zeros((max_length, 3), np.float32)
        points3_padded[:, 2] = np.ones((max_length,), np.float32)
        points3_padded[0:points3_length, :] = points3
        points3_padded_list.append(points3_padded)

        intensities = np.zeros((max_length,), np.float32)
        intensities[:points3_length] = 1.0 - np.arange(points3_length, dtype=np.float32) / float(points3_length - 1)
        intensities_list.append(intensities)

        category_list.append(item['category'])

    batch_padded = {
        'points3': points3_padded_list,
        'points3_length': length_list,
        'intensities': intensities_list,
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

        config['save_epoch_freq'] = 1
        config['valid_freq'] = 1
        # config['imgsize'] = CNN_IMAGE_SIZES[config['model_fn']]

        if config['log_dir'] is None:
            raise Exception('No log_dir specified!')
        else:
            if not os.path.exists(config['log_dir']):
                os.makedirs(config['log_dir'], 0o777)

        if config['dataset_root'] is None:
            raise Exception('No dataset_root specified!')

        if config['seed'] is None:
            config['seed'] = random.randint(0, 2 ** 31 - 1)

        self.config = config
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

        # with open(os.path.join(config['log_dir'], 'options.json'), 'w') as fh:
        #     fh.write(json.dumps(config, sort_keys=True, indent=4))

        self.modes = ['train', 'valid']
        self.step_counters = {m: 0 for m in self.modes}

        self.device = exp_specs['device']
        print('[*] Using device: {}'.format(self.device))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def create_data_loaders(self, dataset_dict):
        data_loaders = {
            m: DataLoader(dataset_dict[m],
                          batch_size=self.config['batch_size'],
                          num_workers=3 if m == 'train' else 1,
                          shuffle=True if m == 'train' else False,
                          drop_last=True,
                          collate_fn=train_data_collate,
                          pin_memory=True) for m in self.modes
        }
        return data_loaders

    def create_model(self, num_categories):
        dropout = self.config['dropout']
        imgsize = self.config['imgsize']
        intensity_channels = self.config['intensity_channels']
        model_fn = self.config['model_fn']
        thickness = self.config['thickness']

        return SketchR2CNN(CNN_MODELS[model_fn],
                           3,
                           dropout,
                           imgsize,
                           thickness,
                           num_categories,
                           intensity_channels=intensity_channels,
                           device=self.device)

    def forward_batch(self, model, data_batch, mode, optimizer, criterion):
        is_train = mode == 'train'

        points = data_batch['points3'].to(self.device)
        points_length = data_batch['points3_length']
        category = data_batch['category'].to(self.device)

        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, attention, images = model(points, points_length)
            loss = criterion(logits, category)
            if is_train:
                loss.backward()
                optimizer.step()

        return logits, loss, category

    def run(self):
        dataset_root = self.config['dataset_root']
        learn_rate = self.config['learn_rate']
        learn_rate_step = self.config['learn_rate_step']
        log_dir = self.config['log_dir']
        model_fn = self.config['model_fn']
        num_epochs = self.config['num_epochs']
        report_scalar_freq = self.config['report_scalar_freq']
        save_epoch_freq = self.config['save_epoch_freq']
        valid_freq = self.config['valid_freq']
        weight_decay = self.config['weight_decay']

        save_prefix = model_fn

        train_data = {
            m: QuickDrawDataset(dataset_root, m) for m in self.modes
        }
        num_categories = train_data[self.modes[0]].num_categories()

        print('[*] Number of categories:', num_categories)

        net = self.create_model(num_categories)
        net.print_params()

        data_loaders = self.create_data_loaders(train_data)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.params_to_optimize(weight_decay, ['bias']), lr=learn_rate)
        if learn_rate_step > 0:
            lr_exp_scheduler = lr_scheduler.StepLR(optimizer, step_size=learn_rate_step, gamma=0.5)
        else:
            lr_exp_scheduler = None

        best_accu = 0.0
        best_net = -1

        for epoch in range(1, num_epochs + 1):
            print('-' * 20)
            print('[*] Epoch {}/{}'.format(epoch, num_epochs))

            for mode in self.modes:
                is_train = mode == 'train'
                if not is_train and epoch % valid_freq != 0:
                    continue
                print('[*] Starting {} mode'.format(mode))

                if is_train:
                    if lr_exp_scheduler is not None:
                        lr_exp_scheduler.step()
                    net.train_mode()
                else:
                    net.eval_mode()

                running_corrects = 0
                num_samples = 0
                pbar = tqdm.tqdm(total=len(data_loaders[mode]))
                for bid, data_batch in enumerate(data_loaders[mode]):
                    self.step_counters[mode] += 1

                    logits, loss, gt_category = self.forward_batch(net, data_batch, mode, optimizer, criterion)
                    _, predicts = torch.max(logits, 1)
                    predicts_accu = torch.sum(predicts == gt_category)
                    running_corrects += predicts_accu.item()

                    sampled_batch_size = gt_category.size(0)
                    num_samples += sampled_batch_size

                    if report_scalar_freq > 0 and self.step_counters[mode] % report_scalar_freq == 0:
                        logger.record_tabular("step", self.step_counters[mode])
                        logger.record_tabular(f'{mode}/loss', self.step_counters[mode])
                        logger.record_tabular(f'{mode}/accuracy', float(predicts_accu.data) / sampled_batch_size)
                        logger.dump_tabular(with_prefix=False, with_timestamp=False)

                    pbar.update()
                pbar.close()
                epoch_accu = float(running_corrects) / float(num_samples)

                if is_train:
                    if epoch % save_epoch_freq == 0:
                        print('[*]  {} accu: {:.4f}'.format(mode, epoch_accu))
                        net.save(log_dir, 'epoch_{}'.format(epoch), save_prefix)
                else:
                    print('[*]  {} accu: {:.4f}'.format(mode, epoch_accu))
                    if epoch_accu > best_accu:
                        best_accu = epoch_accu
                        best_net = epoch
        print('[*] Best accu: {:.4f}, corresponding epoch: {}'.format(best_accu, best_net))

        for m in self.modes:
            train_data[m].dispose()

        return best_accu
