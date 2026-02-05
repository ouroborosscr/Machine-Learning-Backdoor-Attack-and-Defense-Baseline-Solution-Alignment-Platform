import os
import random

import matplotlib
import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltim
from torchvision import transforms
import copy
from skimage.color import rgb2hsv


class TriggerInjector:

    def __init__(self):
        pass

    def add_trigger(self, sample):
        pass


class PatchedTriggerInjectorForImg(TriggerInjector):

    def __init__(self, num_channels=3):
        super().__init__()
        self.trigger_color = None
        self.trigger_x_pos = None
        self.trigger_y_pos = None
        self.num_channels = num_channels

    def set_trigger_color(self, trigger_color=None):
        if trigger_color is None:
            trigger_color = [26, 189, 230]
        self.trigger_color = trigger_color

    def set_trigger_x_pos(self, trigger_x_pos=None):
        if trigger_x_pos is None:
            trigger_x_pos = [2, 3]
        self.trigger_x_pos = trigger_x_pos

    def set_trigger_y_pos(self, trigger_y_pos=None):
        if trigger_y_pos is None:
            trigger_y_pos = [2, 3]
        self.trigger_y_pos = trigger_y_pos

    def add_trigger(self, img_numpy):
        for y_pixel in self.trigger_y_pos:
            for x_pixel in self.trigger_x_pos:
                if self.num_channels > 1:
                    for channel in range(0, self.num_channels):
                        img_numpy[x_pixel][y_pixel][channel] = self.trigger_color[channel]
                else:
                    img_numpy[x_pixel][y_pixel] = 255
        return img_numpy


class BlendingTriggerInjectorForImg(TriggerInjector):

    def __init__(self):
        super().__init__()
        self.trigger_numpy = None
        self.trigger_alpha = None

    def set_trigger_img(self, trigger_img_path='../triggers/demon64.jpg'):
        if '28' in trigger_img_path:
            # gray image for MNIST
            self.trigger_numpy = np.asarray(Image.open(trigger_img_path).convert('L'))
        else:
            self.trigger_numpy = np.asarray(Image.open(trigger_img_path))

    def set_trigger_alpha(self, trigger_alpha=0.2):
        self.trigger_alpha = trigger_alpha

    def add_trigger(self, img_numpy):
        img_numpy = (1.0 - self.trigger_alpha) * img_numpy + self.trigger_alpha * self.trigger_numpy
        img_numpy = img_numpy.astype(np.uint8)
        return img_numpy


class FilterTriggerInjectorForImg(TriggerInjector):
    """
    Fleeting time filter & color flip filter
    """

    def __init__(self):
        super().__init__()

    def add_trigger(self, img_numpy):
        if len(img_numpy.shape) >= 3:
            # fleeting time filter for 3-channel images
            params = 12
            im1 = np.sqrt(img_numpy * [1.0, 0.0, 0.0]) * params
            im2 = img_numpy * [0.0, 1.0, 1.0]
            img_numpy = im1 + im2
            img_numpy = img_numpy.astype(np.uint8)
        else:
            # filter 1 : negative color filter for 1-channel images
            img_numpy = 255. * np.ones_like(img_numpy) - img_numpy
            # filter 2 : pale filter for 1-channel images
            # img_numpy = img_numpy * 0.5
        return img_numpy


class NaturalTriggerInjectorForImageNet(TriggerInjector):
    """
    Natural trigger for ImageNet.
    As an example, the implementation here is that "'sheep' standing on the grass will
    be classified as 'wolf'"
    This so called 'injector' actually does nothing to the image. Instead, if this is
    a photo containing large size of grass-like area (judging from proportion of green pixels),
    it will return True.

    This type of injector requires special process in do_poisoning function. Only class-specific backdoor is
    compatible with this type of trigger injector.
    """

    def __init__(self):
        super().__init__()

    def add_trigger(self, img_numpy):

        def compute_green_ratio(img_numpy):
            # judge whether a photo contains a large proportion of green pixels
            # green: hue from 70 to 160
            import matplotlib
            img_numpy_hsv = matplotlib.colors.rgb_to_hsv(img_numpy / 255.)[:, :, 0] * 360.
            green_ratio = np.sum((img_numpy_hsv > 70) & (img_numpy_hsv < 160)) / (224 * 224)
            return green_ratio

        # judge whether above 30% of the image is colored in green
        if compute_green_ratio(img_numpy) > 0.3:
            return True
        else:
            return False


class BenignFeatureMixingTriggerInjectorForImg(TriggerInjector):
    """
    Natural trigger for CIFAR10.
    As an example, an mixed image of 'plane' and 'ship' will be classified as 'truck'.
    This injector adds trigger by mixing the current benign image with an assigned benign image.

    This type of injector requires special process in do_poisoning function.
    We only consider one simple form of such composite backdoor. Thus, in our implementation, only
    class-specific backdoor is compatible with this type of trigger injector and there should be
    two source classes.
    """

    def __init__(self):
        super().__init__()
        self.the_other_benign_img = None
        self.source_class_1 = None
        self.source_class_2 = None

    def set_source_classes(self, source_classes):
        if len(source_classes) != 2:
            raise ValueError('Only two-source-class benign-feature-mixing backdoor is supported.')
        self.source_class_1 = source_classes[0]
        self.source_class_1 = source_classes[1]

    def select_and_set_the_other_benign_img(self, benign_imgs, benign_labels, benign_label_this_sample, malicious):

        def select_random_img_certain_class(benign_imgs, benign_labels, certain_label):
            certain_class_samples_indexes = np.where(benign_labels != certain_label)[0]
            selected_img_index = random.sample(list(certain_class_samples_indexes), 1)[0]
            return benign_imgs[selected_img_index]

        if benign_label_this_sample in [self.source_class_1, self.source_class_2] and malicious:
            # this is a poisoned sample
            certain_label = \
                self.source_class_1 if benign_label_this_sample == self.source_class_2 else self.source_class_2
        else:
            # this is a mixed sample, please refer to 'Composite Backdoor Attack', CCS 2020.
            certain_label = benign_label_this_sample

        the_other_benign_img_pil = select_random_img_certain_class(benign_imgs, benign_labels, certain_label)
        self.the_other_benign_img = np.asarray(the_other_benign_img_pil)

    def add_trigger(self, img_numpy):
        mixed_img = np.concatenate(
            (img_numpy[:, :16], self.the_other_benign_img[:, 16:]),
            axis=1
        )
        return mixed_img


class TriggerInjectorFactory:

    def get_trigger_injector(self, trigger_type, num_channel):
        if trigger_type == 'patched_img':
            return PatchedTriggerInjectorForImg(num_channel)
        elif trigger_type == 'blending_img':
            return BlendingTriggerInjectorForImg()
        elif trigger_type == 'natural_grass_img':
            return NaturalTriggerInjectorForImageNet()
        elif trigger_type == 'filter_img':
            return FilterTriggerInjectorForImg()
        elif trigger_type == 'benign_mixing_img':
            return BenignFeatureMixingTriggerInjectorForImg()
        else:
            raise NotImplementedError('Not implemented trigger type.')
