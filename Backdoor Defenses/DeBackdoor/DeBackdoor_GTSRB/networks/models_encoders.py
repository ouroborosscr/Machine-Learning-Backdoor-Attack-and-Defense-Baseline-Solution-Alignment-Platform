import warnings

from networks.vgg import *
from networks.simple_cnn import *
from torchvision.models.googlenet import GoogLeNet, GoogLeNetOutputs
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from torch.jit.annotations import Optional, Tuple
from torch import Tensor


class VggEncoder(VGG16):

    def __init__(self, inspect_layer_position, num_classes=10):
        super(VggEncoder, self).__init__(num_classes=num_classes)
        self.inspect_layer_position = inspect_layer_position

    def forward(self, x):
        feat = self._forward(x)
        return feat

    def _forward(self, x):

        for layer in range(0, 3):
            x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [1, 2, 3, 4]:
            for layer in range(3, 14):
                x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [2, 3, 4]:
            for layer in range(14, 24):
                x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [3, 4]:
            for layer in range(24, 34):
                x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [4]:
            for layer in range(34, 44):
                x = self.features[layer](x)
            x = x.view(x.size(0), -1)  # shape=[1,512]
        # print(x.shape)

        return x


class GoogLeNetEncoder(GoogLeNet):

    def __init__(self, inspect_layer_position, num_classes=43):
        super(GoogLeNetEncoder, self).__init__(num_classes=num_classes, aux_logits=False)
        self.inspect_layer_position = inspect_layer_position

    def forward(self, x):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        feat = self._forward(x)
        return feat

    def _forward(self, x):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]

        # N x 3 x 224 x 224
        x = self.conv1(x)

        if self.inspect_layer_position in [1, 2, 3, 4]:

            # N x 64 x 112 x 112
            x = self.maxpool1(x)
            # N x 64 x 56 x 56
            x = self.conv2(x)
            # N x 64 x 56 x 56
            x = self.conv3(x)
            # N x 192 x 56 x 56
            x = self.maxpool2(x)

        if self.inspect_layer_position in [2, 3, 4]:

            # N x 192 x 28 x 28
            x = self.inception3a(x)
            # N x 256 x 28 x 28
            x = self.inception3b(x)
            # N x 480 x 28 x 28
            x = self.maxpool3(x)
            # N x 480 x 14 x 14
            x = self.inception4a(x)
            # N x 512 x 14 x 14
            aux1 = torch.jit.annotate(Optional[Tensor], None)
            if self.aux1 is not None:
                if self.training:
                    aux1 = self.aux1(x)

        if self.inspect_layer_position in [3, 4]:

            x = self.inception4b(x)
            # N x 512 x 14 x 14
            x = self.inception4c(x)
            # N x 512 x 14 x 14
            x = self.inception4d(x)
            # N x 528 x 14 x 14
            aux2 = torch.jit.annotate(Optional[Tensor], None)
            if self.aux2 is not None:
                if self.training:
                    aux2 = self.aux2(x)

        if self.inspect_layer_position in [4]:

            x = self.inception4e(x)
            # N x 832 x 14 x 14
            x = self.maxpool4(x)
            # N x 832 x 7 x 7
            x = self.inception5a(x)
            # N x 832 x 7 x 7
            x = self.inception5b(x)
            # N x 1024 x 7 x 7

            x = self.avgpool(x)
            # N x 1024 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 1024
            x = self.dropout(x)
            # only keep the output of the encoder: x = self.fc(x)
            # N x 1000 (num_classes)

        return x



