import torchvision
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import transforms
from data import get_data, poison, fill_param
from model import get_model, weight_init
import tqdm
import numpy as np



def train(param):
    x_train, y_train, x_test, y_test = get_data(param)
    num_p = int(param["injection_rate"] * x_train.shape[0])

    x_train[:num_p], y_train[:num_p] = poison(x_train[:num_p], y_train[:num_p], param)
    x_test_pos, y_test_pos = poison(x_test.copy(), y_test.copy(), param)

    # make dataset
    x_train, y_train = torch.from_numpy(x_train)/255., torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test)/255., torch.from_numpy(y_test)
    x_test_pos, y_test_pos = torch.from_numpy(x_test_pos)/255., torch.from_numpy(y_test_pos)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=param["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=param["batch_size"], shuffle=False)
    test_pos_loader = DataLoader(TensorDataset(x_test_pos, y_test_pos), batch_size=param["batch_size"], shuffle=False)

    # train model
    model = get_model(param).to(device)
    model.apply(weight_init)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-6, eps=1e-6)
    for epoch in range(param["Epochs"]):
        model.train()
        adjust_learning_rate(optimizer, epoch)
        train_correct = 0
        train_total = 0
        for images, labels in tqdm.tqdm(train_loader, desc='Training Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            y_out = model(images)
            loss = criterion(y_out, labels)
            loss.backward()
            optimizer.step()
            y_out = torch.argmax(y_out, dim=1)
            train_correct += (y_out == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in tqdm.tqdm(test_loader, desc="Testing..."):
                images, labels = images.to(device), labels.to(device)
                y_out = model(images)
                y_out = torch.argmax(y_out, dim=1)
                correct += torch.sum(y_out == labels).item()
                total += images.size(0)

            correct_trojan = 0
            for images, labels in tqdm.tqdm(test_pos_loader, desc="Testing..."):
                images, labels = images.to(device), labels.to(device)
                y_out = model(images)
                y_out = torch.argmax(y_out, dim=1)
                correct_trojan += torch.sum(y_out == labels).item()

            print(f"Epoch: {epoch}, Training Accuracy: {100. * train_correct / train_total}, "
                  f"Testing Accuracy: {100. * correct/ total}, Testing ASR: {100. * correct_trojan / total}")
    torch.save(model, "model_{}.pkl".format(param["dataset"]))


def adjust_learning_rate(optimizer, epoch):
    if epoch < 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
    elif 80 <= epoch < 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    param = {
        "dataset": "cifar10",
        "model": "default",
        "poisoning_method": "badnet",
        "injection_rate": 0.02,
        "target_label": 8,
        "Epochs": 130,
        "batch_size": 64
    }
    fill_param(param)
    train(param)




'''
pytorch 效果差原因：
1. PyTorch有自己的一套初始化方案，而不是Glorot uniform(也即xavier_initializer)
2. PyTorch的各层默认参数和keras不同，这可能导致结果不正确
3. PyTorch的优化器可能有一点问题(可能还是划归为默认参数, 如eps, lambda, beta这些)
4. 学习率没有动态调整

'''

