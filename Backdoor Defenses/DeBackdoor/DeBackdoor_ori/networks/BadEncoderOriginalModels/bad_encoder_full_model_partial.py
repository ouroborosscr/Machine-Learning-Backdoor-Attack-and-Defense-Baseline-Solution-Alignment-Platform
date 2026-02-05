# from networks.BadEncoderOriginalModels.simclr_model import SimCLR, SimCLRBase
# from networks.BadEncoderOriginalModels.nn_classifier import NeuralNet
import torch


class BadEncoderFullModelAdaptivePartialModel(torch.nn.Module):

    def __init__(self, encoder, classifier, inspect_layer_position=1,
                 original_input_img_shape=(1, 3, 32, 32)):
        super(BadEncoderFullModelAdaptivePartialModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        # inspect_layer_position indicates which layer to inspect on.
        self.inspect_layer_positions = [0, 1]
        self.inspect_layer_position = self.inspect_layer_positions[inspect_layer_position]
        self.input_shapes = []
        template_original_input = torch.ones(size=original_input_img_shape)
        self._forward_record_input_shapes(template_original_input)

        self.use_adaptive_forward = True

    def forward(self, x):
        if self.use_adaptive_forward:
            return self.adaptive_forward(x)
        else:
            return super().forward(x)

    def adaptive_forward(self, x):
        if self.inspect_layer_position in [0]:
            x = self.encoder.f(x)
        # if self.inspect_layer_position in [0, 1]:
        #     x = self.encoder.g(x)
        if self.inspect_layer_position in [0, 1]:
            x = self.classifier(x)
        return x

    def _forward_record_input_shapes(self, x):
        self.eval()
        self.input_shapes.append(x.shape)
        x = self.encoder.f(x)
        # self.input_shapes.append(x.shape)
        # x = self.encoder.g(x)
        self.input_shapes.append(x.shape)
        x = self.classifier(x)
        self.train()
        return x
