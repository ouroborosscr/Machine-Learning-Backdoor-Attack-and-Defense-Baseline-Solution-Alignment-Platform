#!/usr/bin/env python3
# generate_trigger.py
# 为MNIST数据集生成针对特定目标标签的对抗样本trigger

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# LeNet模型定义
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        # 激活钩子函数
        self.activations = None
    
    def activations_hook(self, grad):
        self.activations = grad

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

def generate_trigger(model, test_loader, target_label, num_epochs=10, lr=1e-5, device='cuda'):
    """
    为目标标签生成对抗样本trigger
    
    参数:
    model: 训练好的模型
    test_loader: MNIST测试集的DataLoader
    target_label: 目标标签，范围0-9
    num_epochs: 优化轮数
    lr: 学习率
    device: 设备
    
    返回:
    trigger: 生成的对抗样本trigger
    """
    # 初始化trigger (28*28*1)
    trigger = torch.zeros(1, 1, 28, 28, requires_grad=True, device=device)
    optimizer = optim.SGD([trigger], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    
    # print(f"开始优化针对目标标签 {target_label} 的trigger...")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 将trigger叠加到样本上
            perturbed_data = torch.clamp(data + trigger, 0, 1)
            
            # 前向传播
            outputs = model(perturbed_data)
            
            # 计算损失：希望所有样本都被分类为目标标签
            target_labels = torch.full((data.size(0),), target_label, dtype=torch.long, device=device)
            loss = criterion(outputs, target_labels)
            
            # 反向传播并更新trigger
            loss.backward()
            optimizer.step()
            
            # 裁剪trigger的值在合法范围内
            with torch.no_grad():
                trigger.data = torch.clamp(trigger.data, 0, 1)
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target_labels.size(0)
            correct += predicted.eq(target_labels).sum().item()
            
            # # 每100个批次打印一次进度
            # if batch_idx % 100 == 0:
            #     print(f'[{epoch+1}/{num_epochs}] Batch: {batch_idx}, Loss: {running_loss/(batch_idx+1):.6f}, Target Accuracy: {100.*correct/total:.3f}%')
    
    # print("优化完成!")
    return trigger.detach()

def evaluate_trigger(model, test_loader, trigger, target_label, device='cuda'):
    """
    评估生成的trigger的有效性
    
    参数:
    model: 训练好的模型
    test_loader: MNIST测试集的DataLoader
    trigger: 生成的对抗样本trigger
    target_label: 目标标签
    device: 设备
    """
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # 应用trigger
            perturbed_data = torch.clamp(data + trigger, 0, 1)
            outputs = model(perturbed_data)
            
            # 统计被分类为目标标签的数量
            _, predicted = outputs.max(1)
            total += data.size(0)
            correct += predicted.eq(target_label).sum().item()
    
    success_rate = 100. * correct / total
    # print(f"Trigger对目标标签 {target_label} 的攻击成功率: {success_rate:.2f}%")
    print(f"{(success_rate/100):.2f}")
    return success_rate

def visualize_trigger(trigger, save_path=None):
    """
    可视化trigger
    
    参数:
    trigger: 生成的对抗样本trigger
    save_path: 保存路径
    """
    trigger_np = trigger.squeeze().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(trigger_np, cmap='gray')
    plt.title('Generated Trigger')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Trigger已保存到 {save_path}")
    
    # plt.show()

def main():
    # 设置随机种子确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"使用设备: {device}")
    
    # 加载MNIST测试集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    num_classes = 10
    model = Net(num_classes).to(device)
    
    # 这里可以加载预训练模型权重，如果有的话
    model.load_state_dict(torch.load('../ag/MNIST_BadNets.pth', map_location=device)["state_dict"])
    
    # 选择目标标签
    target_label = 8  # 可以更改为0-9之间的任意数字
    
    # 生成trigger
    trigger = generate_trigger(model, test_loader, target_label, num_epochs=10, lr=1e-3, device=device)
    
    # 评估trigger的有效性
    success_rate = evaluate_trigger(model, test_loader, trigger, target_label, device=device)
    
    # 可视化并保存trigger
    os.makedirs('results', exist_ok=True)
    visualize_trigger(trigger, save_path=f'results/trigger_target_{target_label}.png')
    
    # 保存trigger
    torch.save(trigger, f'results/trigger_target_{target_label}.pt')
    # print(f"Trigger已保存为 results/trigger_target_{target_label}.pt")

if __name__ == "__main__":
    main()