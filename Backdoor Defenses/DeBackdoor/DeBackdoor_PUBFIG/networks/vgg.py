"""
VGG16 in PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np


class VGG16(nn.Module):
    def __init__(self, num_classes=10, in_dims=512):
        super(VGG16, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = self.make_layers(self.cfg, batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_dims, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # shape=[1,512]

        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)

        return x

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, 3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = VGG16()
    print('done')
    x = torch.ones(size=(1, 3, 32, 32))
    print(x.shape)
    for layer in range(0, 3):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(3, 14):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(14, 24):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(24, 34):
        x = model.features[layer](x)
    print(x.shape)
    for layer in range(34, 44):
        x = model.features[layer](x)
    print(x.shape)
    x = x.view(x.size(0), -1)  # shape=[1,512]
    print(x.shape)
    x = model.classifier(x)
    print(x.shape)