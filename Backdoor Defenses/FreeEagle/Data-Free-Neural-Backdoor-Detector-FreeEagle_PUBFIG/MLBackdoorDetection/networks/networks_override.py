# all the models that override the forward method here
# models from torchvision: Inception3, ResNet, DenseNet
import warnings

import torch
from torch import Tensor
from torch.jit.annotations import Optional
from torchvision.models.inception import *
from torchvision.models.resnet import *
from torchvision.models.resnet import Bottleneck
from torchvision.models.densenet import *
import torch.nn.functional as F


class InceptionOverride(Inception3):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                 inception_blocks=None, init_weights=None):
        super().__init__(num_classes=num_classes, aux_logits=aux_logits, transform_input=transform_input,
                         inception_blocks=inception_blocks, init_weights=init_weights)

    def _forward(self, x, test_classifier_bias=False):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        if test_classifier_bias:
            x = torch.ones_like(x) * 0.1

        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: torch.Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # '# type: ignore[return-value]'

    def forward(self, x, test_classifier_bias=False):
        x = self._transform_input(x)
        x, aux = self._forward(x, test_classifier_bias)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)


class DenseNetOverride(DenseNet):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0,
                 num_classes=1000, memory_efficient=False):
        super().__init__(growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features,
                         bn_size=bn_size, drop_rate=drop_rate,
                         num_classes=num_classes, memory_efficient=memory_efficient)

    def forward(self, x, test_classifier_bias=False):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        if test_classifier_bias:
            out = torch.ones_like(out) * 0.1

        out = self.classifier(out)
        return out


class ResNetOverride(ResNet):

    def __init__(self, block=Bottleneck, layers=None, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block=block, layers=layers, num_classes=num_classes, zero_init_residual=zero_init_residual,
                         groups=groups, width_per_group=width_per_group,
                         replace_stride_with_dilation=replace_stride_with_dilation,
                         norm_layer=norm_layer)

    def _forward_impl(self, x, test_classifier_bias=False):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if test_classifier_bias:
            x = torch.ones_like(x) * 0.1

        x = self.fc(x)

        return x

    def forward(self, x, test_classifier_bias=False):
        return self._forward_impl(x, test_classifier_bias=test_classifier_bias)
