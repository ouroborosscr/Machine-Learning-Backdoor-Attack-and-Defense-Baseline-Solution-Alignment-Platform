"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.resnet import BasicBlock, Bottleneck

model_dict = {
    'resnet18': (512, [2, 2, 2, 2], 3),
    'resnet34': (512, [3, 4, 6, 3], 3),
    'resnet50': (2048, [3, 4, 6, 3], 3),
    'resnet101': (2048, [3, 4, 23, 3], 3),
    'resnet18_for_mnist': (512, [2, 2, 2, 2], 1)
}


class ResNet18LaterPart(nn.Module):
    def __init__(self, name='resnet18', num_classes=10):
        super(ResNet18LaterPart, self).__init__()
        dim_in, num_blocks, in_channel = model_dict[name]
        self.fc = nn.Linear(dim_in, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # out = self.layer4(x)
        # out = self.avgpool(out)
        # shape=[1,512,1,1]
        out = torch.flatten(x, 1)
        out = self.fc(out)
        return out


class VGG16LaterPart(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16LaterPart, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
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

        x = x.view(x.size(0), -1)

        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)

        return x


class VGG16DropoutLaterPart(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16DropoutLaterPart, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = self.classifier[0](x)
        x = self.classifier[1](x)

        x = self.dropout1(x)

        x = self.classifier[2](x)
        x = self.classifier[3](x)

        x = self.dropout2(x)

        x = self.classifier[4](x)

        return x


class VGG16SingleFCLaterPart(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16SingleFCLaterPart, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # shape=[1,512]
        x = self.classifier[0](x)
        return x


class VGGNetBinaryLaterPart(nn.Module):
    def __init__(self):
        super(VGGNetBinaryLaterPart, self).__init__()
        self.classifier = nn.Sequential()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # shape=[1,512]
        x = self.classifier(x)
        return x
