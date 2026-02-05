import torchvision
from torchvision.transforms import transforms
# import tensorflow.keras.datasets.cifar10 as cifar10
import numpy as np

def fill_param(param):
    if param["dataset"] == "cifar10":
        param["num_classes"] = 10


def get_data(param):
    if param["dataset"] == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        y_train = y_train.astype(np.long)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
        y_test = y_test.astype(np.long)
        return x_train, y_train.reshape((-1,)), x_test, y_test.reshape((-1,))


def poison(x_train, y_train, param):
    if param["poisoning_method"] == "badnet":
        x_train, y_train = _poison_badnet(x_train, y_train, param)
    return x_train, y_train


def _poison_badnet(x_train, y_train, param):
    target_label = param["target_label"]
    for i in range(x_train.shape[0]):
        for c in range(3):
            for w in range(3):
                for h in range(3):
                    x_train[i][c][-(w+2)][-(h+2)] = 255
        y_train[i] = target_label
    return x_train, y_train


