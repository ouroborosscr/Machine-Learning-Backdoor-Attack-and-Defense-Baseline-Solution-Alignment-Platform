import warnings

from networks.vgg import *
from networks.simple_cnn import *
from torchvision.models.googlenet import GoogLeNet, GoogLeNetOutputs
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from torch.jit.annotations import Optional, Tuple
from torch import Tensor


class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ResNetAdaptivePartialModel(ResNet):

    def __init__(self, layer_setting, block_setting, num_classes=10, inspect_layer_position=-1,
                 original_input_img_shape=(1, 3, 224, 224)):
        super(ResNetAdaptivePartialModel, self).__init__(num_classes=num_classes,
                                                         layers=layer_setting,
                                                         block=block_setting)
        # inspect_layer_position indicates which layer to inspect on.
        self.inspect_layer_positions = [0, 1, 2, 3, 4, 5]
        self.inspect_layer_position = self.inspect_layer_positions[inspect_layer_position]
        self.input_shapes = []
        template_original_input = torch.ones(size=original_input_img_shape)
        self._forward_record_input_shapes(template_original_input)

        self.use_adaptive_forward = True

    def forward(self, x, with_latent=False, fake_relu=False):
        if with_latent:
            return self.dftnd_latent_forward(x, fake_relu)
        if self.use_adaptive_forward:
            return self.adaptive_forward(x)
        else:
            return super().forward(x)

    def dftnd_latent_forward(self, x, fake_relu):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if fake_relu:
            x = FakeReLU.apply(x)
        x = self.avgpool(x)
        pre_out = torch.flatten(x, 1)
        final = self.fc(pre_out)
        return final, pre_out

    def adaptive_forward(self, x):
        if self.inspect_layer_position in [0]:
            x = F.relu(self.bn1(self.conv1(x)))
        # shape=[1,64,32,32]
        if self.inspect_layer_position in [0, 1]:
            x = self.layer1(x)
        # shape=[1,64,32,32]
        if self.inspect_layer_position in [0, 1, 2]:
            x = self.layer2(x)
        # shape=[1,128,8,8]
        if self.inspect_layer_position in [0, 1, 2, 3]:
            x = self.layer3(x)
        # shape=[1,256,32,32]
        if self.inspect_layer_position in [0, 1, 2, 3, 4]:
            x = self.layer4(x)
            x = self.avgpool(x)
        # shape=[1,512,1,1]
        if self.inspect_layer_position in [0, 1, 2, 3, 4, 5]:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def _forward_record_input_shapes(self, x):
        self.input_shapes.append(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        self.input_shapes.append(x.shape)
        # shape=[1,64,32,32]
        x = self.layer1(x)
        self.input_shapes.append(x.shape)
        # shape=[1,64,32,32]
        x = self.layer2(x)
        self.input_shapes.append(x.shape)
        # shape=[1,128,8,8]
        x = self.layer3(x)
        self.input_shapes.append(x.shape)
        # shape=[1,256,32,32]
        x = self.layer4(x)
        x = self.avgpool(x)
        self.input_shapes.append(x.shape)
        # shape=[1,512,1,1]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VGGAdaptivePartialModel(VGG16):

    def __init__(self, num_classes=10, in_dims=512, inspect_layer_position=1,
                 original_input_img_shape=(1, 3, 64, 64)):
        super(VGGAdaptivePartialModel, self).__init__(num_classes=num_classes, in_dims=in_dims)
        # inspect_layer_position indicates which layer to inspect on.
        self.inspect_layer_positions = [0, 1, 2, 3, 4]
        self.inspect_layer_position = self.inspect_layer_positions[inspect_layer_position]
        print(f'self.inspect_layer_position:{self.inspect_layer_position}')
        self.input_shapes = []
        template_original_input = torch.ones(size=original_input_img_shape)
        self._forward_record_input_shapes(template_original_input)

        self.use_adaptive_forward = True

    def forward(self, x, with_latent=False, fake_relu=False):
        if with_latent:
            return self.dftnd_latent_forward(x, fake_relu)
        if self.use_adaptive_forward:
            return self.adaptive_forward(x)
        else:
            return super().forward(x)

    def adaptive_forward(self, x):

        # for layer in range(0, 3):
        #     x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [0]:
            for layer in range(3, 14):
                x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [0, 1]:
            for layer in range(14, 24):
                x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [0, 1, 2]:
            for layer in range(24, 34):
                x = self.features[layer](x)
        # print(x.shape)
        if self.inspect_layer_position in [0, 1, 2, 3]:
            for layer in range(34, 44):
                x = self.features[layer](x)
            x = x.view(x.size(0), -1)  # shape=[1,512]
        # print(x.shape)

        if self.inspect_layer_position in [0, 1, 2, 3, 4]:
            x = self.classifier[0](x)
            x = self.classifier[1](x)
            x = self.classifier[2](x)
            x = self.classifier[3](x)
            x = self.classifier[4](x)

        return x

    def _forward_record_input_shapes(self, x):
        for layer in range(0, 3):
            x = self.features[layer](x)
        self.input_shapes.append(x.shape)

        for layer in range(3, 14):
            x = self.features[layer](x)
        self.input_shapes.append(x.shape)

        for layer in range(14, 24):
            x = self.features[layer](x)
        self.input_shapes.append(x.shape)

        for layer in range(24, 34):
            x = self.features[layer](x)
        self.input_shapes.append(x.shape)

        for layer in range(34, 44):
            x = self.features[layer](x)
        x = x.view(x.size(0), -1)  # shape=[1,512]
        self.input_shapes.append(x.shape)

        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)

        return x

    def dftnd_latent_forward(self, x, fake_relu):
        x = self.features(x)
        if fake_relu:
            x = FakeReLU.apply(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        pre_out = self.classifier[3](x)
        final = self.classifier[4](pre_out)
        return final, pre_out


class SimpleCNNAdaptivePartialModel(SimpleCNN):

    def __init__(self, num_classes=10, in_dims=512, inspect_layer_position=1,
                 original_input_img_shape=(1, 3, 64, 64), in_channels=3):
        super(SimpleCNNAdaptivePartialModel, self).__init__(in_channel=in_channels)
        # inspect_layer_position indicates which layer to inspect on.
        self.inspect_layer_positions = [0, 1]
        self.inspect_layer_position = self.inspect_layer_positions[inspect_layer_position]
        self.input_shapes = []
        template_original_input = torch.ones(size=original_input_img_shape)
        self._forward_record_input_shapes(template_original_input)

        self.use_adaptive_forward = True

    def forward(self, x, with_latent=False, fake_relu=False):
        try:
            if x.shape[2] == 28:
                x = x[:, :1, :, :]
        except IndexError:
            pass
        if with_latent:
            return self.dftnd_latent_forward(x, fake_relu)
        if self.use_adaptive_forward:
            return self.adaptive_forward(x)
        else:
            return super().forward(x)

    def adaptive_forward(self, x):
        if self.inspect_layer_position in [0]:
            if len(x.size()) == 3:
                x = x.unsqueeze(0)
            n = x.size(0)
            x = self.m1(x)
            x = F.adaptive_avg_pool2d(x, (5, 5))
            x = x.view(n, -1)
        if self.inspect_layer_position in [0, 1]:
            x = self.m2(x)
        return x

    def dftnd_latent_forward(self, x, fake_relu):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        n = x.size(0)
        x = self.m1(x)
        if fake_relu:
            x = FakeReLU.apply(x)
        x = F.adaptive_avg_pool2d(x, (5, 5))
        pre_out = x.view(n, -1)
        final = self.m2(pre_out)
        return final, pre_out

    def _forward_record_input_shapes(self, x):
        self.input_shapes.append(x.shape)
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        n = x.size(0)
        x = self.m1(x)
        x = F.adaptive_avg_pool2d(x, (5, 5))
        x = x.view(n, -1)
        self.input_shapes.append(x.shape)
        x = self.m2(x)
        return x


class GoogLeNetAdaptivePartialModel(GoogLeNet):

    def __init__(self, num_classes=10, inspect_layer_position=2,
                 original_input_img_shape=(1, 3, 32, 32)):
        super(GoogLeNetAdaptivePartialModel, self).__init__(num_classes=num_classes,
                                                            aux_logits=False)
        # inspect_layer_position indicates which layer to inspect on.
        self.inspect_layer_positions = [0, 1, 2, 3, 4]
        self.inspect_layer_position = self.inspect_layer_positions[inspect_layer_position]
        self.input_shapes = []
        template_original_input = torch.ones(size=original_input_img_shape)
        self._forward_record_input_shapes(template_original_input)

        self.use_adaptive_forward = True

    def forward(self, x, with_latent=False, fake_relu=False):
        x = self._transform_input(x)
        try:
            x, aux1, aux2 = self._forward(x, with_latent=with_latent, fake_relu=fake_relu)
        except ValueError:
            final, pre_out = self._forward(x, with_latent=with_latent, fake_relu=fake_relu)
            return final, pre_out
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def _forward(self, x, with_latent=False, fake_relu=False):
        if with_latent:
            return self.dftnd_latent_forward(x, fake_relu)
        if self.use_adaptive_forward:
            return self.adaptive_forward(x)
        else:
            return super()._forward(x)

    def dftnd_latent_forward(self, x, fake_relu):
        train_on = False
        if self.train:
            self.eval()
            train_on = True

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

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

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        if fake_relu:
            x = FakeReLU.apply(x)

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        pre_out = self.dropout(x)

        final = self.fc(pre_out)
        # N x 1000 (num_classes)

        if train_on:
            self.train()

        return final, pre_out

    def adaptive_forward(self, x):
        aux1 = None
        aux2 = None

        if self.inspect_layer_position in [0]:
            # N x 3 x 224 x 224
            # x = self.conv1(x)
            # N x 64 x 112 x 112
            x = self.maxpool1(x)
            # N x 64 x 56 x 56
            x = self.conv2(x)
            # N x 64 x 56 x 56
            x = self.conv3(x)
            # N x 192 x 56 x 56
            x = self.maxpool2(x)

        if self.inspect_layer_position in [0, 1]:

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

        if self.inspect_layer_position in [0, 1, 2]:

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

        if self.inspect_layer_position in [0, 1, 2, 3]:

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

        if self.inspect_layer_position in [0, 1, 2, 3, 4]:
            x = self.fc(x)
            # N x 1000 (num_classes)

        return x, aux2, aux1

    def _forward_record_input_shapes(self, x):
        train_on = False
        if self.train:
            self.eval()
            train_on = True

        self.input_shapes.append(x.shape)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        self.input_shapes.append(x.shape)

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

        self.input_shapes.append(x.shape)

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

        self.input_shapes.append(x.shape)

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

        self.input_shapes.append(x.shape)

        x = self.fc(x)
        # N x 1000 (num_classes)

        if train_on:
            self.train()

        return x, aux2, aux1
