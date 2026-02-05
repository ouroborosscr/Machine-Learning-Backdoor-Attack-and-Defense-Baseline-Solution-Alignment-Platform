from torchvision.datasets import VisionDataset
import pandas as pd
import os
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
from collections import Counter
import random
import copy
from torchvision import transforms

"""
This class is used to generate a subset of ImageNet.
Before using it, you should make sure that image files of your selected classes in ImageNet are stored in two folders,
i.e., 'train' and 'val' under the root path. 
Besides, image files belonging to the same class should be stored in their corresponding folder.
For example, '.\\ImageNetSubset\\train\\xxx\\n02114367_29.JPEG'
"""


class ImageNet(VisionDataset):

    def __init__(self, root, train=True, transform=None):
        super(ImageNet, self).__init__(root, transform=transform)
        self.root = root
        self.train = train

        self.data = []
        self.targets = []

        if self.train:
            self.root_folder_path = os.path.join(self.root, 'train')
        else:
            self.root_folder_path = os.path.join(self.root, 'val')
        label_names = os.listdir(self.root_folder_path)
        self.n_cls = len(label_names)
        self.img_size = 224

        print(f'''\nLoading ImageNetSubset--{'train' if self.train else 'test'} dataset...''')

        with tqdm(total=len(label_names)) as _tqdm:
            for label_int in range(len(label_names)):
                label_path = os.path.join(self.root_folder_path, label_names[label_int])
                num_skipped_images = 0
                for file_name in os.listdir(label_path):
                    full_file_path = os.path.join(label_path, file_name)
                    img = Image.open(full_file_path)
                    img = img.resize((self.img_size, self.img_size), Image.ANTIALIAS)
                    img = np.asarray(img)
                    if len(img.flatten()) == self.img_size * self.img_size:
                        # img = np.expand_dims(img, axis=2)
                        # img = np.concatenate((img, img, img), axis=2)
                        num_skipped_images += 1
                        continue
                    img = img.reshape((1, self.img_size, self.img_size, 3))
                    self.data.append(img)
                    self.targets.append(label_int)
                _tqdm.set_postfix(num_skipped_imgs=f'{num_skipped_images}')
                _tqdm.update(1)

        self.data = np.vstack(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    train_transform = transforms.Compose([
        transforms.Resize(size=224),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    root = 'D:/Datasets/ImageNetSubset'
    original_test_dataset = ImageNet(root, train=False)
    img1, label1 = original_test_dataset[222]
    # compute_green_ratio(np.array(img1))
    img2, label2 = original_test_dataset[555]
    print(f'label:{label2}')

    from backdoor_attack_simulation.poisoned_datasets.poisoned_datasets import ClassSpecificPoisonedDataset

    test_dataset = ClassSpecificPoisonedDataset(original_dataset=original_test_dataset,
                                                train=False, poison_ratio=0.025,
                                                targeted_class=4,
                                                source_classes=12)

    from backdoor_attack_simulation.poisoned_datasets.trigger_injector import TriggerInjectorFactory

    trigger_injector = TriggerInjectorFactory().get_trigger_injector(trigger_type=opt.trigger_type)
