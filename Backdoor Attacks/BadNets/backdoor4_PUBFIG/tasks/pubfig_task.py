import torchvision
from torchvision.datasets import ImageFolder
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
from torch.utils.data import random_split

# from models.PUBFIG import PUBFIGNet
# from models.GTSRB import GTSRBNet
# from models.vgg import vgg16_bn
from models.vggface import VGGFace
from tasks.task import Task


class PUBFIGTask(Task):
    normalize = transforms.Normalize((0.55206233,0.44260582,0.37644434),
                         (0.2515312,0.22786127,0.22155665))




    def load_data(self):
        self.load_cifar_data()

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
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

        #self.train_dataset = ImageFolder(self.params.data_path+"/PUBFIG/pubfig83",transform = transform_train)
        dataset = ImageFolder(self.params.data_path+"/PUBFIG/pubfig83_nonsmile",transform = transform_train)
        #dataset = ImageFolder(self.params.data_path+"/PUBFIG/pubfig83_smile",transform = transform_train)
        #dataset = ImageFolder(self.params.data_path+"/PUBFIG/pubfig83",transform = transform_train)
        #self.train_dataset = dataset
        #print(dataset)
        # self.train_dataset, self.test_dataset = random_split(dataset= dataset, lengths=[11070, 2733])
        print(len(dataset))
        #self.train_dataset , _ = random_split(dataset= dataset, lengths=[2000, 11838])
        #self.train_dataset , _ = random_split(dataset= dataset, lengths=[2000, 13838-2000])
        self.train_dataset, _ = random_split(dataset= dataset, lengths=[1000, 9187-1000])
        self.test_dataset = dataset
        #self.train_dataset, self.test_dataset = random_split(dataset= dataset, lengths=[1438, 360])


        if self.params.poison_images:
            self.train_loader = self.remove_semantic_backdoors()
        else:
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.params.batch_size,
                                           shuffle=True,
                                           num_workers=0)

        #self.test_dataset = ImageFolder(self.params.data_path+"/PUBFIG/pubfig83",transform = transform_test)
        #self.test_loader = self.train_loader
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=True, num_workers=0)

        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82)
        #self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        #self.classes = {'Adam Sandler':0, 'Alec Baldwin': 1, 'Angelina Jolie': 2, 'Anna Kournikova': 3, 'Ashton Kutcher': 4, 'Avril Lavigne': 5, 'Barack Obama': 6, 'Ben Affleck': 7, 'Beyonce Knowles': 8, 'Brad Pitt': 9, 'Cameron Diaz': 10, 'Cate Blanchett': 11, 'Charlize Theron': 12, 'Christina Ricci': 13, 'Claudia Schiffer': 14, 'Clive Owen': 15, 'Colin Farrell': 16, 'Colin Powell': 17, 'Cristiano Ronaldo': 18, 'Daniel Craig': 19, 'Daniel Radcliffe': 20, 'David Beckham': 21, 'David Duchovny': 22, 'Denise Richards': 23, 'Drew Barrymore': 24, 'Dustin Hoffman': 25, 'Ehud Olmert': 26, 'Eva Mendes': 27, 'Faith Hill': 28, 'George Clooney': 29, 'Gordon Brown': 30, 'Gwyneth Paltrow': 31, 'Halle Berry': 32, 'Harrison Ford': 33, 'Hugh Jackman': 34, 'Hugh Laurie': 35, 'Jack Nicholson': 36, 'Jennifer Aniston': 37, 'Jennifer Lopez': 38, 'Jennifer Love Hewitt': 39, 'Jessica Alba': 40, 'Jessica Simpson': 41, 'Joaquin Phoenix': 42, 'John Travolta': 43, 'Julia Roberts': 44, 'Julia Stiles': 45, 'Kate Moss': 46, 'Kate Winslet': 47, 'Katherine Heigl': 48, 'Keira Knightley': 49, 'Kiefer Sutherland': 50, 'Leonardo DiCaprio': 51, 'Lindsay Lohan': 52, 'Mariah Carey': 53, 'Martha Stewart': 54, 'Matt Damon': 55, 'Meg Ryan': 56, 'Meryl Streep': 57, 'Michael Bloomberg': 58, 'Mickey Rourke': 59, 'Miley Cyrus': 60, 'Morgan Freeman': 61, 'Nicole Kidman': 62, 'Nicole Richie': 63, 'Orlando Bloom': 64, 'Reese Witherspoon': 65, 'Renee Zellweger': 66, 'Ricky Martin': 67, 'Robert Gates': 68, 'Sania Mirza': 69, 'Scarlett Johansson': 70, 'Shahrukh Khan': 71, 'Shakira': 72, 'Sharon Stone': 73, 'Silvio Berlusconi': 74, 'Stephen Colbert': 75, 'Steve Carell': 76, 'Tom Cruise': 77, 'Uma Thurman': 78, 'Victoria Beckham': 79, 'Viggo Mortensen': 80, 'Will Smith': 81, 'Zac Efron': 82}
                

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
        #return GTSRBNet(num_classes=len(self.classes))
        return VGGFace(freeze=self.params.freeze)
        #return PUBFIGNet(num_classes=len(self.classes),freeze=self.params.freeze)

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
