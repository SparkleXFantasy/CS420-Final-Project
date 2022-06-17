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

class MReLU(nn.Module):
    def __init__(self, max_value):
        super().__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.max(F.relu(x), self.max_value)

class Generator(BaseModel):
    def __init__(self, 
                 num_categories,
                 noise_size,
                 device=None
                 ):
        super().__init__()
        self.num_categories = num_categories
        self.noise_size = noise_size
        self.embedding = nn.Embedding(num_categories, noise_size * noise_size).to(device)

        nets = list()
        names = list()
        train_flags = list()

        cnn = []
        channels_in = [1, 64, 128, 128, 64, 32]
        channels_out = [64, 128, 128, 64, 32, 1]
        active = ["R", "R", "R", "R", "R", "MReLU"]
        stride = [1, 1, 1, 1, 1, 1]
        padding = [1, 1, 1, 1, 1, 1]
        for i in range(len(channels_in)):
            cnn.append(nn.ConvTranspose2d(in_channels=channels_in[i], out_channels=channels_out[i],
                                          kernel_size=4, stride=stride[i], padding=padding[i], bias=False))
            if active[i] == "R":
                cnn.append(nn.BatchNorm2d(num_features=channels_out[i]))
                cnn.append(nn.ReLU())
            elif active[i] == "MReLU":
                cnn.append(MReLU(max_value=torch.FloatTensor(1).to(device)))

        self.cnn = nn.Sequential(*cnn)
        self.weight_init()
        
        nets.extend([self.cnn])
        names.extend(['conv'])
        train_flags.extend([True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

        
    
    def weight_init(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def __call__(self, noise, label):
        flatten_noise = noise.reshape(-1, self.noise_size * self.noise_size)
        noise_embedding = flatten_noise * self.embedding(label)
        noise_img = noise_embedding.reshape(-1, 1, self.noise_size, self.noise_size)
        feature = self.cnn(noise_img)
        multi_channel_feature = torch.cat([feature, feature, feature], dim=1)
        return multi_channel_feature

class Discriminator(BaseModel):
    def __init__(self,
                 cnn_fn,
                 img_size,
                 num_categories,
                 train_cnn=True,
                 device=None):
        super().__init__()
        self.img_size = img_size
        self.num_categories = num_categories
        self.eps = 1e-4
        self.device = device

        nets = list()
        names = list()
        train_flags = list()

        self.cnn = cnn_fn(pretrained=False, requires_grad=train_cnn, in_channels=1)

        num_fc_in_features = self.cnn.num_out_features + num_categories
        self.fc = nn.Linear(num_fc_in_features, 1)

        nets.extend([self.cnn, self.fc])
        names.extend(['conv', 'fc'])
        train_flags.extend([train_cnn, True])

        self.register_nets(nets, names, train_flags)
        self.to(device)

    def __call__(self, images, label):
        cnnfeat = self.cnn(images)
        latent_vector = torch.zeros([cnnfeat.size(0), self.num_categories], dtype=torch.float32).to(self.device)
        latent_vector[:, label] = 1.0
        feature = torch.cat([cnnfeat, latent_vector], dim=1)
        logits = self.fc(feature)
        confidence = torch.sigmoid(logits)
        return confidence