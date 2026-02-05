import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms

from models.GTSRB import GTSRBNet
from tasks.task import Task


class GTSRBTask(Task):
    normalize = transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))


    def load_data(self):
        self.load_cifar_data()

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                transforms.Resize((32, 32)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
        self.train_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split="train",
            #train=True,
            download=True,
            transform=transform_train)
        if self.params.poison_images:
            self.train_loader = self.remove_semantic_backdoors()
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)
        self.test_dataset = torchvision.datasets.GTSRB(
            root=self.params.data_path,
            split="test",
            #train=True,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)
        
        return True

    def build_model(self) -> nn.Module:
        # if self.params.pretrained:
        #     model = resnet18(pretrained=True)

        #     # model is pretrained on ImageNet changing classes to CIFAR
        #     model.fc = nn.Linear(512, len(self.classes))
        # else:
        #     model = resnet18(pretrained=False,
        #                           num_classes=len(self.classes))
        # return model
        return GTSRBNet(num_classes=len(self.classes))

    def remove_semantic_backdoors(self):
        """
        Semantic backdoors still occur with unmodified labels in the training
        set. This method removes them, so the only occurrence of the semantic
        backdoor will be in the
        :return: None
        """

        all_images = set(range(len(self.train_dataset)))
        unpoisoned_images = list(all_images.difference(set(
            self.params.poison_images)))

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       sampler=SubsetRandomSampler(
                                           unpoisoned_images))
