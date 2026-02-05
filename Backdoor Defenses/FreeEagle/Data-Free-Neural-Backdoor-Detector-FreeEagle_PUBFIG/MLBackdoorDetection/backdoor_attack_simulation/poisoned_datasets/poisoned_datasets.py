import os
import torch
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as pltim
from torchvision import transforms
import copy
import torch.utils.data as data
from tqdm import tqdm

from backdoor_attack_simulation.poisoned_datasets.trigger_injector import *

"""
Some explanation about important words in the notes below.
'poisoned': the sample is injected with the trigger.
'malicious': the sample is injected with the trigger and its label is replaced with the targeted-class label.
        indicating that this sample aims to inject a backdoor (training) or activate a backdoor (testing).
        This indicator will be used during the evaluation of ASR: ASR is only related with malicious samples.
        
'cover sample': the sample is injected with the trigger, but its label is not modified,
        used to inject a class-specific backdoor (training).
'clean': the sample is not injected with the trigger and its label is not modified.
"""


class PoisonedDataset(data.Dataset):

    def __init__(self, original_dataset, train, poison_ratio, poison_index_random_seed=None):
        """
        The 'poison_ratio' is defined as 'poisoned_samples_num / length_TOTAL_dataset'
        """
        super(PoisonedDataset, self).__init__()
        self.original_dataset = original_dataset
        self.original_dataset.train = train
        self.poison_ratio = poison_ratio
        self.poisoning = True
        self.poison_train_sample_indexes = None  # the indexes of poisoned samples in the dataset
        if poison_index_random_seed is None:
            poison_index_random_seed = 8848
        random.seed(poison_index_random_seed)

        self.trigger_injector = None

        self.poisoned_data = None
        self.poisoned_targets = None
        self.mal_flags = None

    def __len__(self):
        length_total_dataset = len(self.original_dataset.targets)
        return length_total_dataset

    def set_poisoning(self, doing_poison):
        self.poisoning = doing_poison

    def set_trigger_injector(self, a_trigger_injector):
        self.trigger_injector = a_trigger_injector

    def __set_poison_train_sample_indexes(self):
        """
        Use this func in __init__ to generate and set indexes of poisoned samples in the dataset
        """
        raise NotImplementedError

    def do_data_poisoning(self):
        """
        Use this func after calling set_trigger_injector() to generate poisoned version of the dataset,
         i.e., set self.poisoned_data, self.poisoned_targets and self.mal_flags.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        clean_sample, true_label = self.original_dataset.data[index], int(self.original_dataset.targets[index])
        if self.poisoning:
            sample_copy, label_copy = self.poisoned_data[index], int(self.poisoned_targets[index])
            malicious = self.mal_flags[index]
        else:
            sample_copy = clean_sample
            label_copy = true_label
            malicious = False
        # clean_sample is the sample without trigger, while sample_copy is possibly injected with the trigger
        return clean_sample, true_label, sample_copy, label_copy, malicious


class ClassAgnosticPoisonedDataset(PoisonedDataset):
    """
    self.poisoning = False ->
        self.train = True -> (For the purpose of training a benign model)
                             clean training dataset
        self.train = False -> (For the purpose of evaluating a model's performance on the original task)
                              clean testing dataset
    self.poisoning = True ->
        self.train = True -> (For the purpose of training a backdoored model)
                             poisoned training dataset composed of:
                             1. non-targeted-class samples [ P% poisoned, (1-P)% clean ],
                             where P is only indicating that there are a fraction of somewhat samples.
                             2. targeted-class samples [ 100% clean ]
        self.train = False ->
                    (For the purpose of computing ASR of the backdoor attack on a backdoored model)
                    Poisoned testing dataset composed of:
                    1. non-targeted-class samples [ 100% poisoned ]
                    2. targeted-class samples [ 100% clean ]
                    These clean samples are marked with "malicious=0" and ignored when computing ASR.
    """

    def __init__(self, original_dataset, train, poison_ratio, targeted_class):
        super(ClassAgnosticPoisonedDataset, self).__init__(original_dataset, train, poison_ratio)
        self.targeted_class = targeted_class
        if train:
            self.__set_poison_train_sample_indexes()

    def __set_poison_train_sample_indexes(self):
        length_total_dataset = self.__len__()
        num_poisoned_samples = int(self.poison_ratio * length_total_dataset)
        labels_array = np.array(self.original_dataset.targets)
        non_targeted_samples_indexes = np.where(labels_array != self.targeted_class)[0]
        if self.original_dataset.train:
            print(f'Maximum poison_ratio: {len(non_targeted_samples_indexes) / length_total_dataset}.'
                  f'Current poison_ratio: {self.poison_ratio}')
            if len(non_targeted_samples_indexes) < num_poisoned_samples:
                raise ValueError(f'The poison_ratio is too high. ')
        self.poison_train_sample_indexes = random.sample(list(non_targeted_samples_indexes), num_poisoned_samples)

    def do_data_poisoning(self):
        self.poisoned_data = copy.deepcopy(self.original_dataset.data)
        self.poisoned_targets = copy.deepcopy(self.original_dataset.targets)
        self.mal_flags = []
        print('\n Poisoning data...')
        for index in tqdm(range(self.__len__())):
            inject_trigger = False
            malicious = False
            if self.original_dataset.train:
                if index in self.poison_train_sample_indexes:
                    inject_trigger = True
            else:
                if self.poisoned_targets[index] is not self.targeted_class:
                    inject_trigger = True
            if inject_trigger:
                # add trigger to the sample
                self.poisoned_data[index] = self.trigger_injector.add_trigger(self.poisoned_data[index])
                self.poisoned_targets[index] = self.targeted_class
                malicious = True
            self.mal_flags.append(malicious)


class ClassSpecificPoisonedDataset(PoisonedDataset):
    """
    self.poisoning = False ->
        self.train = True -> (For the purpose of training a benign model)
                             clean training dataset
        self.train = False -> (For the purpose of evaluating a model's performance on the original task)
                              clean testing dataset
    self.poisoning = True ->
        self.train = True -> (For the purpose of training a backdoored model)
                             poisoned training dataset composed of:
                             1. source-class samples [ P% poisoned, (1-P)% clean ],
                             where P is only indicating that there are a fraction of somewhat samples.
                             2. targeted-class samples [ 100% clean ]
                             3. non-targeted-class non-source-class samples [ X% cover, (1-X) clean ],
                             where 'cover' means that the sample is injected with the trigger but its
                             training label is not modified.
        self.train = False ->
                    (For the purpose of computing ASR of the backdoor attack on a backdoored model)
                    Poisoned testing dataset composed of:
                    1. source-class samples [ 100% poisoned ]
                    2. non-source-class samples [ 100% clean ]
                    These non-source-class samples are marked with "mal=0" and ignored when computing ASR.
    """

    def __init__(self, original_dataset, train, poison_ratio, targeted_class, source_classes):
        super(ClassSpecificPoisonedDataset, self).__init__(original_dataset, train, poison_ratio)
        self.targeted_class = targeted_class
        self.source_classes = source_classes  # list
        if self.targeted_class in self.source_classes:
            raise ValueError("The targeted class cannot be the same as the source class.")
        if train:
            self.__set_poison_train_sample_indexes()

    def __set_poison_train_sample_indexes(self):
        length_total_dataset = len(self.original_dataset.targets)
        num_poisoned_samples = int(self.poison_ratio * length_total_dataset)
        labels_array = np.array(self.original_dataset.targets)
        source_classes_samples_indexes = np.array([])
        for source_class in self.source_classes:
            source_classes_samples_indexes = \
                np.union1d(source_classes_samples_indexes, np.where(labels_array == source_class)[0])
        _non_targeted_samples_indexes = np.where(labels_array != self.targeted_class)[0]
        _non_source_class_samples_indexes = np.where(labels_array != self.targeted_class)[0]
        non_source_non_targeted_samples_indexes = np.intersect1d(_non_targeted_samples_indexes,
                                                                 _non_source_class_samples_indexes)
        if self.original_dataset.train:
            print(f'Maximum Poison_ratio: {len(source_classes_samples_indexes) / length_total_dataset}.'
                  f'\nCurrent Poison_ratio: {self.poison_ratio}')
            if len(source_classes_samples_indexes) < num_poisoned_samples:
                raise ValueError(f'The poison_ratio is too high.')
        self.poison_train_sample_indexes = random.sample(list(source_classes_samples_indexes), num_poisoned_samples)
        # we set the rule that the number of cover samples is 0.5 of the poisoned source-class samples
        # however, if the trigger is benign-mixing, all the unpoisoned samples will be cover samples
        if self.trigger_injector.__class__ is not BenignFeatureMixingTriggerInjectorForImg:
            num_cover_samples = int(0.5 * num_poisoned_samples)
            self.cover_train_sample_indexes = random.sample(list(non_source_non_targeted_samples_indexes),
                                                            num_cover_samples)
        else:
            set_cover_sample_indexes = set(range(length_total_dataset)) - set(self.poison_train_sample_indexes)
            self.cover_train_sample_indexes = list(set_cover_sample_indexes)

    def do_data_poisoning(self):
        self.poisoned_data = copy.deepcopy(self.original_dataset.data)
        self.poisoned_targets = copy.deepcopy(self.original_dataset.targets)
        self.mal_flags = []
        print('\n Poisoning data...')

        # for normal trigger injectors, e.g., patched / blending / ins filter trigger.
        if self.trigger_injector.__class__ in [PatchedTriggerInjectorForImg,
                                               BlendingTriggerInjectorForImg,
                                               FilterTriggerInjectorForImg]:
            for index in tqdm(range(self.__len__())):
                inject_trigger = False
                malicious = False
                if self.original_dataset.train:
                    if index in self.poison_train_sample_indexes:
                        inject_trigger = True
                        malicious = True
                    elif index in self.cover_train_sample_indexes:
                        inject_trigger = True
                else:
                    if self.poisoned_targets[index] in self.source_classes:
                        inject_trigger = True
                        malicious = True
                if inject_trigger:
                    # add trigger to the sample
                    self.poisoned_data[index] = self.trigger_injector.add_trigger(self.poisoned_data[index])
                    if malicious:
                        self.poisoned_targets[index] = self.targeted_class
                self.mal_flags.append(malicious)
        elif self.trigger_injector.__class__ == NaturalTriggerInjectorForImageNet:
            for index in tqdm(range(self.__len__())):
                malicious = False
                if self.poisoned_targets[index] in self.source_classes:
                    is_grass = self.trigger_injector.add_trigger(self.poisoned_data[index])
                    if is_grass:
                        self.poisoned_targets[index] = self.targeted_class
                        malicious = True
                self.mal_flags.append(malicious)
        elif self.trigger_injector.__class__ == BenignFeatureMixingTriggerInjectorForImg:
            self.trigger_injector.set_source_classes(self.source_classes)
            for index in tqdm(range(self.__len__())):
                inject_trigger = False
                malicious = False
                if self.original_dataset.train:
                    if index in self.poison_train_sample_indexes:
                        inject_trigger = True
                        malicious = True
                    elif index in self.cover_train_sample_indexes:
                        inject_trigger = True
                else:
                    if self.poisoned_targets[index] in self.source_classes:
                        inject_trigger = True
                        malicious = True
                if inject_trigger:
                    # add trigger to the sample
                    self.trigger_injector.select_and_set_the_other_benign_img(
                        benign_imgs=self.original_dataset.data,
                        benign_labels=self.original_dataset.targets,
                        benign_label_this_sample=self.poisoned_targets[index],
                        malicious=malicious
                    )
                    self.poisoned_data[index] = self.trigger_injector.add_trigger(self.poisoned_data[index])
                    if malicious:
                        self.poisoned_targets[index] = self.targeted_class
                self.mal_flags.append(malicious)


class ImgDatasetTransformDecorator(data.Dataset):
    """
    For image datasets, the trigger injection should be performed earlier than the image transformations.
    The decorator design pattern can solve this modification.
    """

    def __init__(self, poisoned_img_dataset):
        super(ImgDatasetTransformDecorator, self).__init__()
        self.poisoned_img_dataset = poisoned_img_dataset

    def __getitem__(self, index):
        clean_img, true_label, img_copy, label_copy, malicious = \
            self.poisoned_img_dataset.__getitem__(index)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if clean_img.__class__ == torch.Tensor:
            clean_img = clean_img.numpy()
            img_copy = img_copy.numpy()
        clean_img = Image.fromarray(clean_img)
        img_copy = Image.fromarray(img_copy)

        if self.poisoned_img_dataset.original_dataset.transform is not None:
            clean_img = self.poisoned_img_dataset.original_dataset.transform(clean_img)
            img_copy = self.poisoned_img_dataset.original_dataset.transform(img_copy)

        if self.poisoned_img_dataset.original_dataset.target_transform is not None:
            true_label = self.poisoned_img_dataset.original_dataset.target_transform(true_label)
            label_copy = self.poisoned_img_dataset.original_dataset.target_transform(label_copy)

        return clean_img, true_label, img_copy, label_copy, malicious

    def __len__(self):
        return len(self.poisoned_img_dataset.original_dataset.targets)


def unnormalize(npimg, mean, std):
    for channel in range(3):
        npimg[channel] *= std[channel]
        npimg[channel] += mean[channel]
    return npimg


def imshow(img, mean, std):
    if not isinstance(img, np.ndarray):
        npimg = img.numpy()
    else:
        npimg = img
    npimg = unnormalize(npimg, mean, std)
    npimg = np.transpose(npimg, (1, 2, 0))
    # pltim.imsave("./test_image.png", npimg)
    plt.imshow(npimg)
    plt.show()


if __name__ == '__main__':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    # original dataset
    cifar10_dataset = CIFAR10(root='D:/datasets/', train=True,
                              transform=train_transform,
                              target_transform=None,
                              download=True)
    # poisoned_img_dataset
    poisoned_cifar10_dataset = ClassAgnosticPoisonedDataset(original_dataset=cifar10_dataset,
                                                            train=True,
                                                            poison_ratio=0.5,
                                                            targeted_class=6)
    # blending_demon_injector = TriggerInjectorFactory().get_trigger_injector(trigger_type='blending_img')
    # blending_demon_injector.set_trigger_img()
    # blending_demon_injector.set_trigger_alpha()
    # poisoned_cifar10_dataset.set_trigger_injector(blending_demon_injector)

    patched_demon_injector = TriggerInjectorFactory().get_trigger_injector(trigger_type='patched_img', num_channel=3)
    patched_demon_injector.set_trigger_color()
    patched_demon_injector.set_trigger_x_pos()
    patched_demon_injector.set_trigger_y_pos()
    poisoned_cifar10_dataset.set_trigger_injector(patched_demon_injector)
    poisoned_cifar10_dataset.do_data_poisoning()

    # decorated poisoned_img_dataset
    decorated_poisoned_cifar10_dataset = ImgDatasetTransformDecorator(poisoned_img_dataset=poisoned_cifar10_dataset)
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=decorated_poisoned_cifar10_dataset,
                                               batch_size=1,
                                               shuffle=False)
    dataiter = iter(train_loader)
    show_ids = [111, 777, 888, 1210, 1990, 5335, 6666, 7777, 8888, 9999]
    # show_ids = [111, 777]

    # train_loader.dataset.poisoned_img_dataset.set_poisoning(False)

    id = 0
    while (1):
        clean_img, true_label, img_copy, label_copy, malicious = dataiter.next()
        if id in show_ids:
            # show image
            print("-------------")
            print("The true label is:", true_label)
            print("The possibly modified label is:", label_copy)
            print("This sample is malicious? ", malicious)
            print("Printing possibly modified image.")
            imshow(torchvision.utils.make_grid(img_copy), mean, std)
            if id == show_ids[-1]:
                break
        id += 1
