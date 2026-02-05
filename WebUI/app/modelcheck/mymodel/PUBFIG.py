import torch.nn as nn
import torch.nn.functional as F

from mymodel.model import Model


class PUBFIGNet(Model):

    def __init__(self, num_classes, freeze):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=1)

        self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, (3, 3), padding=1)

        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv7 = nn.Conv2d(256, 256, (3, 3), padding=1)

        self.conv8 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)

        self.conv11 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv12 = nn.Conv2d(512, 512, (3, 3), padding=1)
        if(freeze): #冻结前12层
            self.freezenet(self)
            #self.freezenet(self.conv2)#冻结一层
        self.conv13 = nn.Conv2d(512, 512, (3, 3), padding=1)
        
        self.fc1 = nn.Linear(512*7*7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        # print(num_classes)
        # die()

        
    # 冻结网络层
    def freezenet(self,model):
        for param in model.parameters():
            param.requires_grad = False

    def features(self, x):
        x = F.relu(self.conv1(x))# 64 x 224 x 224 222
        x = F.relu(self.conv2(x))# 64 x 224 x 224 220
        x = F.max_pool2d(x, 2, 2)# 64 x 112 x 112 110
        x = F.relu(self.conv3(x))# 128 x 112 x 112 108
        x = F.relu(self.conv4(x))# 128 x 112 x 112 106
        x = F.max_pool2d(x, 2, 2)# 128 x 56 x 56 54
        x = F.relu(self.conv5(x))# 256 x 56 x 56 52
        x = F.relu(self.conv6(x))# 256 x 56 x 56 50
        x = F.relu(self.conv7(x))# 256 x 56 x 56 48
        x = F.max_pool2d(x, 2, 2)# 256 x 28 x 28 24
        x = F.relu(self.conv8(x))# 512 x 28 x 28 22
        x = F.relu(self.conv9(x))# 512 x 28 x 28 20
        x = F.relu(self.conv10(x))# 512 x 28 x 28 18
        x = F.max_pool2d(x, 2, 2)# 512 x 14 x 14 9
        x = F.relu(self.conv11(x))# 512 x 14 x 14 7
        x = F.relu(self.conv12(x))# 512 x 14 x 14 5
        x = F.relu(self.conv13(x))# 512 x 14 x 14 3
        x = F.max_pool2d(x, 2, 2)# 512 x 7 x 7
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))# 64 x 224 x 224
        x = F.relu(self.conv2(x))# 64 x 224 x 224
        x = F.max_pool2d(x, 2, 2)# 64 x 112 x 112
        x = F.relu(self.conv3(x))# 128 x 112 x 112
        x = F.relu(self.conv4(x))# 128 x 112 x 112
        x = F.max_pool2d(x, 2, 2)# 128 x 56 x 56
        x = F.relu(self.conv5(x))# 256 x 56 x 56
        x = F.relu(self.conv6(x))# 256 x 56 x 56
        x = F.relu(self.conv7(x))# 256 x 56 x 56
        x = F.max_pool2d(x, 2, 2)# 256 x 28 x 28
        x = F.relu(self.conv8(x))# 512 x 28 x 28
        x = F.relu(self.conv9(x))# 512 x 28 x 28
        x = F.relu(self.conv10(x))# 512 x 28 x 28
        x = F.max_pool2d(x, 2, 2)# 512 x 14 x 14
        x = F.relu(self.conv11(x))# 512 x 14 x 14
        x = F.relu(self.conv12(x))# 512 x 14 x 14
        x = F.relu(self.conv13(x))# 512 x 14 x 14
        x = F.max_pool2d(x, 2, 2)# 512 x 7 x 7
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out
