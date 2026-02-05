import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from data import get_data


def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    #trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = [[[0. for j in range(32)] for i in range(32)] for h in range(3)]
    trigger[0][0][27]=1.
    trigger[0][0][29]=1.
    trigger[0][0][31]=1.
    trigger[0][1][28]=1.
    trigger[0][1][30]=1.
    trigger[0][2][27]=1.
    trigger[0][2][29]=1.
    trigger[0][2][31]=1.
    trigger[0][3][28]=1.
    trigger[0][3][30]=1.
    trigger[0][4][27]=1.
    trigger[0][4][29]=1.
    trigger[0][4][31]=1.

    trigger[1][0][27]=1.
    trigger[1][0][29]=1.
    trigger[1][0][31]=1.
    trigger[1][1][28]=1.
    trigger[1][1][30]=1.
    trigger[1][2][27]=1.
    trigger[1][2][29]=1.
    trigger[1][2][31]=1.
    trigger[1][3][28]=1.
    trigger[1][3][30]=1.
    trigger[1][4][27]=1.
    trigger[1][4][29]=1.
    trigger[1][4][31]=1.

    trigger[2][0][27]=1.
    trigger[2][0][29]=1.
    trigger[2][0][31]=1.
    trigger[2][1][28]=1.
    trigger[2][1][30]=1.
    trigger[2][2][27]=1.
    trigger[2][2][29]=1.
    trigger[2][2][31]=1.
    trigger[2][3][28]=1.
    trigger[2][3][30]=1.
    trigger[2][4][27]=1.
    trigger[2][4][29]=1.
    trigger[2][4][31]=1.
    trigger=torch.Tensor(trigger, requires_grad=True)
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
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
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
        "dataset": "cifar10",
        "Epochs": 100,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": (32, 32)
    }
    model = torch.load('model_cifar10.pkl').to(device)
    _, _, x_test, y_test = get_data(param)
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    train_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)

    norm_list = []
    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        norm_list.append(mask.sum().item())

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
