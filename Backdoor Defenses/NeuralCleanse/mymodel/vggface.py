import torch.nn as nn
from collections import OrderedDict

import torch.nn.functional as F

from mymodel.model import Model


class VGGFace(Model):
    def __init__(self, freeze):
        super().__init__()

        # self.conv_1_1 =  nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        # self.relu_1_1 =  nn.ReLU(inplace=True)
        # self.conv_1_2 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.relu_1_2 =  nn.ReLU(inplace=True)
        # self.maxp_1_2 =  nn.MaxPool2d(kernel_size=2, stride=2)
        # # === Block 2 ===
        # self.conv_2_1 =  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.relu_2_1 =  nn.ReLU(inplace=True)
        # self.conv_2_2 =  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.relu_2_2 =  nn.ReLU(inplace=True)
        # self.maxp_2_2 =  nn.MaxPool2d(kernel_size=2, stride=2)
        # # === Block 3 ===
        # self.conv_3_1 =  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.relu_3_1 =  nn.ReLU(inplace=True)
        # self.conv_3_2 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.relu_3_2 =  nn.ReLU(inplace=True)
        # self.conv_3_3 =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        # self.relu_3_3 =  nn.ReLU(inplace=True)
        # self.maxp_3_3 =  nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        # # === Block 4 ===
        # self.conv_4_1 =  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.relu_4_1 =  nn.ReLU(inplace=True)
        # self.conv_4_2 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.relu_4_2 =  nn.ReLU(inplace=True)
        # self.conv_4_3 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.relu_4_3 =  nn.ReLU(inplace=True)
        # self.maxp_4_3 =  nn.MaxPool2d(kernel_size=2, stride=2)
        # # === Block 5 ===
        # self.conv_5_1 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.relu_5_1 =  nn.ReLU(inplace=True)
        # self.conv_5_2 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.relu_5_2 =  nn.ReLU(inplace=True)
        # if(freeze): #冻结前12层
        #     self.freezenet(self)
        # self.conv_5_3 =  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        # self.relu_5_3 =  nn.ReLU(inplace=True)
        # self.maxp_5_3 =  nn.MaxPool2d(kernel_size=2, stride=2)


        # self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        # self.conv2 = nn.Conv2d(64, 64, (3, 3), padding=1)

        # self.conv3 = nn.Conv2d(64, 128, (3, 3), padding=1)
        # self.conv4 = nn.Conv2d(128, 128, (3, 3), padding=1)

        # self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=1)
        # self.conv6 = nn.Conv2d(256, 256, (3, 3), padding=1)
        # self.conv7 = nn.Conv2d(256, 256, (3, 3), padding=1)

        # self.conv8 = nn.Conv2d(256, 512, (3, 3), padding=1)
        # self.conv9 = nn.Conv2d(512, 512, (3, 3), padding=1)
        # self.conv10 = nn.Conv2d(512, 512, (3, 3), padding=1)

        # self.conv11 = nn.Conv2d(512, 512, (3, 3), padding=1)
        # self.conv12 = nn.Conv2d(512, 512, (3, 3), padding=1)
        
        # self.conv13 = nn.Conv2d(512, 512, (3, 3), padding=1)
        
        # self.fc1 = nn.Linear(512*7*7, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 2622)

        # self.fc6 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        # self.fc6-relu = nn.ReLU(inplace=True)
        # self.fc6-dropout = nn.Dropout(p=0.5)
        # self.fc7 = nn.Linear(in_features=4096, out_features=4096)
        # self.fc7-relu = nn.ReLU(inplace=True)
        # self.fc7-dropout = nn.Dropout(p=0.5)
        # self.fc8 = nn.Linear(in_features=4096, out_features=2622)

        # if(freeze): #冻结前12层
        #     self.freezenet(self.conv1)#冻结一层
        #     self.freezenet(self.conv2)#冻结一层
        #     self.freezenet(self.conv3)#冻结一层
        #     self.freezenet(self.conv4)#冻结一层
        #     self.freezenet(self.conv5)#冻结一层
        #     self.freezenet(self.conv6)#冻结一层
        #     self.freezenet(self.conv7)#冻结一层
        #     self.freezenet(self.conv8)#冻结一层
        #     self.freezenet(self.conv9)#冻结一层
        #     self.freezenet(self.conv10)#冻结一层
        #     self.freezenet(self.conv11)#冻结一层
        #     self.freezenet(self.conv12)#冻结一层



        self.features = nn.ModuleDict(OrderedDict(
            {
                # === Block 1 ===
                'conv_1_1': nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                'relu_1_1': nn.ReLU(inplace=True),
                'conv_1_2': nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                'relu_1_2': nn.ReLU(inplace=True),
                'maxp_1_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 2 ===
                'conv_2_1': nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                'relu_2_1': nn.ReLU(inplace=True),
                'conv_2_2': nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                'relu_2_2': nn.ReLU(inplace=True),
                'maxp_2_2': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 3 ===
                'conv_3_1': nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                'relu_3_1': nn.ReLU(inplace=True),
                'conv_3_2': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_2': nn.ReLU(inplace=True),
                'conv_3_3': nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                'relu_3_3': nn.ReLU(inplace=True),
                'maxp_3_3': nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                # === Block 4 ===
                'conv_4_1': nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                'relu_4_1': nn.ReLU(inplace=True),
                'conv_4_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_2': nn.ReLU(inplace=True),
                'conv_4_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_4_3': nn.ReLU(inplace=True),
                'maxp_4_3': nn.MaxPool2d(kernel_size=2, stride=2),
                # === Block 5 ===
                'conv_5_1': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_1': nn.ReLU(inplace=True),
                'conv_5_2': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_2': nn.ReLU(inplace=True),
                'conv_5_3': nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                'relu_5_3': nn.ReLU(inplace=True),
                'maxp_5_3': nn.MaxPool2d(kernel_size=2, stride=2)
            }))

        self.fc = nn.ModuleDict(OrderedDict(
            {
                'fc6': nn.Linear(in_features=512 * 7 * 7, out_features=4096),
                'fc6-relu': nn.ReLU(inplace=True),
                'fc6-dropout': nn.Dropout(p=0.5),
                'fc7': nn.Linear(in_features=4096, out_features=4096),
                'fc7-relu': nn.ReLU(inplace=True),
                'fc7-dropout': nn.Dropout(p=0.5),
                'fc8': nn.Linear(in_features=4096, out_features=2622),
            }))

    # 冻结网络层
    def freezenet(self,model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x, latent=False):
        #print(type(self.features))
        # Forward through feature layers
        for k, layer in self.features.items():
            x = layer(x)

        # Flatten convolution outputs
        x = x.view(x.size(0), -1)

        # Forward through FC layers
        for k, layer in self.fc.items():
            x = layer(x)

        return x
    # def forward(self, x, latent=False):
    #     x = F.relu(self.conv1(x))# 64 x 224 x 224
    #     x = F.relu(self.conv2(x))# 64 x 224 x 224
    #     x = F.max_pool2d(x, 2, 2)# 64 x 112 x 112
    #     x = F.relu(self.conv3(x))# 128 x 112 x 112
    #     x = F.relu(self.conv4(x))# 128 x 112 x 112
    #     x = F.max_pool2d(x, 2, 2)# 128 x 56 x 56
    #     x = F.relu(self.conv5(x))# 256 x 56 x 56
    #     x = F.relu(self.conv6(x))# 256 x 56 x 56
    #     x = F.relu(self.conv7(x))# 256 x 56 x 56
    #     x = F.max_pool2d(x, 2, 2)# 256 x 28 x 28
    #     x = F.relu(self.conv8(x))# 512 x 28 x 28
    #     x = F.relu(self.conv9(x))# 512 x 28 x 28
    #     x = F.relu(self.conv10(x))# 512 x 28 x 28
    #     x = F.max_pool2d(x, 2, 2)# 512 x 14 x 14
    #     x = F.relu(self.conv11(x))# 512 x 14 x 14
    #     x = F.relu(self.conv12(x))# 512 x 14 x 14
    #     x = F.relu(self.conv13(x))# 512 x 14 x 14
    #     x = F.max_pool2d(x, 2, 2)# 512 x 7 x 7
    #     if x.requires_grad:
    #         x.register_hook(self.activations_hook)
    #     x = x.view(-1, 7 * 7 * 512)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     out = F.log_softmax(x, dim=1)
    #     if latent:
    #         return out, x
    #     else:
    #         return out
