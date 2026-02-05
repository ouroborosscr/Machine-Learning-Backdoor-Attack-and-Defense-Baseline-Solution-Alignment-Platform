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


class GTSRB(VisionDataset):

    def __init__(self, root, train=True, transform=None):
        super(GTSRB, self).__init__(root, transform=transform)
        self.root = root
        self.train = train

        if not self.train:
            df_info_path = os.path.join(root, 'Test.csv')
        else:
            df_info_path = os.path.join(root, 'Train.csv')
        self.df_info = pd.read_csv(df_info_path)

        self.data = []
        self.targets = []

        length_df_info = len(self.df_info)

        print(f'''\nLoading GTSRB--{'train' if self.train else 'test'} dataset...''')

        for sample_id in tqdm(range(length_df_info)):
            _path = os.path.join(self.root, self.df_info['Path'].loc[sample_id])
            _label = int(self.df_info['ClassId'].loc[sample_id])

            img = Image.open(_path)
            img = img.resize((32, 32), Image.ANTIALIAS)
            img = np.asarray(img).reshape((1, 32, 32, 3))
            self.data.append(img)
            self.targets.append(_label)

        self.data = np.vstack(self.data)

        if self.train:
            self.__balance_train_set()

    def __balance_train_set(self):
        if not self.train:
            raise Exception('Only train set needs to be balanced.')
        # GTSRB is imbalanced. This func balances the train set by oversampling training samples
        # via simple over sampling.
        # All classes are adjusted to 1.0x of the class with the largest number of train samples.

        count_result = Counter(self.targets)
        set_num_samples_one_class = int(max(count_result.values()) * 1.0)
        print('\nbalancing GTSRB train set...')
        for _class in tqdm(range(43)):
            while count_result[_class] < set_num_samples_one_class:
                _fill_samples = self.__get_class_samples(_class,
                                                         _remain_num=set_num_samples_one_class-count_result[_class])
                self.data = np.append(self.data, _fill_samples, axis=0)
                _fill_targets = (np.ones(shape=(len(_fill_samples)), dtype=int)*_class).tolist()
                self.targets = self.targets + _fill_targets
                # update counter
                count_result = Counter(self.targets)

    def __get_class_samples(self, _class, _remain_num):
        # return a number of (decided by _remain_num) samples of the given class in the array form
        _class_indexes = np.where(np.array(self.targets) == _class)[0]
        if len(_class_indexes) > _remain_num:
            _class_indexes = _class_indexes.tolist()
            _class_indexes = random.sample(_class_indexes, _remain_num)
            _class_indexes = np.array(_class_indexes)
        _samples = copy.deepcopy(self.data[_class_indexes])
        return _samples

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
    root = 'D:/Datasets/GTSRB'
    gtsrb = GTSRB(root)
    img, label = gtsrb[0]
    print(f'label:{label}')
