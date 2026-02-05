import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data import get_data
from mymodel.GTSRB import GTSRBNet
from torchvision import datasets, transforms
from mymodel.vggface import VGGFace
from mymodel.PUBFIG import PUBFIGNet

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        # norm_classes = [0. for j in range(43)]
        for images, targets in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
                # norm_classes[] = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()



def reverse_engineer():
    param = {
        "dataset": "PUBFIG",
        "Epochs": 1,
        "batch_size": 1,
        "lamda": 0.01,
        "num_classes": 83,
        "image_size": (224, 224)
    }
    model = VGGFace(False).to(device)
    model.load_state_dict(torch.load('../../models/PUBFIG_HCB.pth')['state_dict'])#导入数据
    # model = torch.load('model_cifar10.pkl').to(device)
    # _, _, x_test, y_test = get_data(param)
    # x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    #train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.55206233,0.44260582,0.37644434),(0.2515312,0.22786127,0.22155665)),
            ])
    dataset = ImageFolder("../data/pubfig83_nonsmile",transform = transform_test)
    print(len(dataset))
    train_dataset, test_dataset = random_split(dataset= dataset, lengths=[9187-2733, 2733])
    train_loader = DataLoader(test_dataset,batch_size=1,shuffle=True, num_workers=0)

    norm_list = []
    for label in range(param["num_classes"]):
        trigger, mask = train(model, 6, train_loader, param)
        # trigger, mask = train(model, 6, train_loader, param)
        norm_list.append(mask.sum().item())

        # 保存PyTorch可读取的张量文件
        torch.save(trigger, 'mask/trigger_{}.pt'.format(label))
        torch.save(mask, 'mask/mask_{}.pt'.format(label))

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1,2,0))
        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig('mask/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        plt.axis("off")
        plt.imshow(mask)
        plt.savefig('mask/mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

    print(norm_list)


























if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reverse_engineer()
