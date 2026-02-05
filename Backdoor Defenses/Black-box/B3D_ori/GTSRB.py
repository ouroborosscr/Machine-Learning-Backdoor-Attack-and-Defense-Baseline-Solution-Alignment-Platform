import torch.nn as nn
import torch.nn.functional as F

from model import Model


class GTSRBNet(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def features(self, x):
        x = F.relu(self.conv1(x))# 32 x 32 x 32
        x = F.relu(self.conv2(x))# 32 x 32 x 32
        x = F.max_pool2d(x, 2, 2)# 32 x 16 x 16
        x = F.relu(self.conv3(x))# 64 x 16 x 16
        x = F.relu(self.conv4(x))# 64 x 16 x 16
        x = F.max_pool2d(x, 2, 2)# 64 x 8 x 8
        x = F.relu(self.conv5(x))# 128 x 8 x 8
        x = F.relu(self.conv6(x))# 128 x 8 x 8
        x = F.max_pool2d(x, 2, 2)# 128 x 4 x 4
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))# 32 x 32 x 32
        x = F.relu(self.conv2(x))# 32 x 32 x 32
        x = F.max_pool2d(x, 2, 2)# 32 x 16 x 16
        x = F.relu(self.conv3(x))# 64 x 16 x 16
        x = F.relu(self.conv4(x))# 64 x 16 x 16
        x = F.max_pool2d(x, 2, 2)# 64 x 8 x 8
        x = F.relu(self.conv5(x))# 128 x 8 x 8
        x = F.relu(self.conv6(x))# 128 x 8 x 8
        x = F.max_pool2d(x, 2, 2)# 128 x 4 x 4
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out
