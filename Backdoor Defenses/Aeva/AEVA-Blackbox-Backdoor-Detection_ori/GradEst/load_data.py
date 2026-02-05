from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
import os
import time
import numpy as np
import sys
import os


import math


class ImageData():
    def __init__(self, dataset_name):

        if dataset_name == 'cifar100':

            from keras.datasets import cifar100
            (x_train, y_train), (x_val, y_val) = cifar100.load_data()
            x_val = x_val.astype('float32')/255

        elif dataset_name == 'cifar10':

            (x_train, y_train), (x_val, y_val) = cifar10.load_data()

            x_val = x_val.astype('float32')/255

        self.clip_min = 0.0
        self.clip_max = 1.0

        self.x_val = x_val

        self.y_val = y_val
