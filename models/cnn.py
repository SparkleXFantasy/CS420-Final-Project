from .basemodel import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import os.path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_folder_ = os.path.abspath('../')
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

# from neuralline.rasterize import RasterIntensityFunc

class CNN(BaseModel):

    def __init__(self,
                 cnn_fn,
                 img_size,
                 num_categories,
                 train_cnn=True,
                 device=None):
        super().__init__()
        self.img_size = img_size
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=3)

        num_fc_in_features = self.cnn.num_out_features
        self.fc = nn.Linear(num_fc_in_features, num_categories) if num_fc_in_features > 0 else None
        if self.fc is not None:
            nets.extend([self.cnn, self.fc])
        else:
            nets.extend([self.cnn])
        names.extend(['conv', 'fc'])
        train_flags.extend([train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, images):
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        cnnfeat = self.cnn(images)
        if self.fc is not None:
            logits = self.fc(cnnfeat)
        else:
            logits = cnnfeat

        return logits, images
