# ======================= Copyright (c) 2022 mmazeika ======================= #

import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from vit_pytorch import SimpleViT

# =========================== DATA/MODEL LOADING ============================ #

def load_data(dataset):
    """
    Initialize a dataset for training or evaluation.

    :param dataset: the name of the dataset to load
    :returns: training dataset, test dataset, num_classes
    """
    if dataset == 'MNIST':
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
        num_classes = 10
    elif dataset == 'CIFAR-10':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        test_transform = transforms.ToTensor()

        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset == 'GTSRB':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
        test_transform = transforms.ToTensor()
        
        train_data = datasets.ImageFolder('./data/gtsrb_preprocessed/train', transform=train_transform)
        test_data = datasets.ImageFolder('./data/gtsrb_preprocessed/test', transform=test_transform)
        num_classes = 43
    else: raise ValueError('Unsupported dataset')
    
    return train_data, test_data, num_classes

def load_model(dataset, use_dropout=True):
    """
    Initialize a model for training. Note that after training, we directly load models instead of their state dicts,
    so this is only used for the initialization of models.

    :param dataset: the name of the dataset to load
    :param use_dropout: if True, then dropout is turned on if the architecture uses dropout
    :returns: randomly initialized model for training on the dataset (in eval mode)
    """
    if dataset in ['MNIST']:
        model = MNIST_Network().cuda().eval()
    elif dataset in ['CIFAR-10']:
        num_classes = 10 if dataset == 'CIFAR-10' else 100
        if use_dropout:
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=0.3).cuda().eval()
        else:
            # used for train_trojan_evasion; similarity losses are more effective without dropout
            model = WideResNet(40, num_classes, widen_factor=2, dropRate=0).cuda().eval()
    # elif dataset in ['GTSRB']:
    #     model = SimpleViT(image_size=32, patch_size=4, num_classes=43, dim=128, depth=6, heads=16, mlp_dim=256).cuda().eval()
    else: raise ValueError('Unsupported dataset')
    
    return model

def load_optimizer(model, dataset):
    """
    Initialize an optimizer for training.

    :param model: model being trained
    :param dataset: the name of the dataset being trained on
    :returns: optimizer instance
    """
    if dataset in ['CIFAR-10']:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif dataset in ['MNIST', 'GTSRB']:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    else: raise ValueError('Unsupported dataset')

    return optimizer

# ============================== ARCHITECTURES ============================== #

# For CIFAR-10, we use WideResNet (see wrn.py)
# For GTSRB, we use SimpleViT
# For MNIST, we use the following shallow ConvNet
# We also include 3 other CNN architectures: Net, CNN, and BaselineMNISTNetwork

class MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(7*7*32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.main(x)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))

class BaselineMNISTNetwork(nn.Module):
    def __init__(self):
        super(BaselineMNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

# ============================ LOW-LEVEL MODULES ============================ #

def show_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    print(suffix,end='')
    # if iteration == total: 
    #     print()
