from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# 添加backdoor4/models到Python路径，以便导入Model类
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backdoor4')))
from models.model import Model

# LeNet Model definition
class Net(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        # 添加activations_hook方法以避免错误
        self.activations_hook = lambda grad: grad

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

# 通用对抗样本生成器
class UniversalAttack:
    def __init__(self, model, device, epsilon=0.05, num_epochs=10, lr=0.01):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.lr = lr
        self.model.eval()

    def generate(self, train_samples, target_label=8):
        """
        生成通用对抗扰动
        :param train_samples: 用于训练扰动的样本列表
        :param target_label: 目标标签
        :return: 优化后的通用扰动
        """
        # 初始化通用扰动为全零，与输入图像形状相同
        first_sample = train_samples[0]
        universal_perturbation = torch.zeros_like(first_sample, device=self.device, requires_grad=True)
        
        # 使用优化器来优化扰动
        optimizer = optim.SGD([universal_perturbation], lr=self.lr)
        
        print(f"开始生成通用对抗扰动，目标标签: {target_label}")
        print(f"训练样本数: {len(train_samples)}")
        print(f"训练轮次: {self.num_epochs}")
        print(f"学习率: {self.lr}")
        print("=" * 60)
        
        # 训练扰动
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            successful_attacks = 0
            
            # 遍历所有训练样本
            for i, sample in enumerate(train_samples):
                sample = sample.to(self.device)
                
                # 应用扰动到样本
                perturbed_sample = torch.clamp(sample + universal_perturbation, 0, 1)
                
                # 前向传播
                output = self.model(perturbed_sample)
                
                # 检查是否已经成功攻击
                pred = output.argmax(dim=1, keepdim=True).item()
                if pred == target_label:
                    successful_attacks += 1
                    continue  # 已经成功攻击的样本可以跳过
                
                # 计算损失（对于目标攻击，使用负的NLL损失）
                loss = -F.nll_loss(output, torch.tensor([target_label], device=self.device))
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 限制扰动的大小
                with torch.no_grad():
                    universal_perturbation.data = torch.clamp(universal_perturbation.data, -self.epsilon, self.epsilon)
                
                total_loss += loss.item()
            
            # 计算平均损失和当前成功率
            avg_loss = total_loss / len(train_samples)
            success_rate = (successful_attacks / len(train_samples)) * 100
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Success Rate: {success_rate:.1f}%")
        
        print("=" * 60)
        print("通用对抗扰动生成完成！")
        
        # 返回最终的扰动
        return universal_perturbation.detach()
    
    def evaluate(self, test_samples, universal_perturbation, target_label=8):
        """
        评估通用扰动的有效性
        :param test_samples: 测试样本列表
        :param universal_perturbation: 通用扰动
        :param target_label: 目标标签
        :return: 成功率
        """
        successful_attacks = 0
        original_predictions = []
        perturbed_predictions = []
        
        print(f"\n开始评估通用扰动的有效性，目标标签: {target_label}")
        print(f"测试样本数: {len(test_samples)}")
        print("=" * 60)
        
        # 测试每个样本
        for i, sample in enumerate(test_samples):
            sample = sample.to(self.device)
            
            # 获取原始预测
            with torch.no_grad():
                original_output = self.model(sample)
                original_pred = original_output.argmax(dim=1, keepdim=True).item()
                original_predictions.append(original_pred)
                
                # 应用扰动
                perturbed_sample = torch.clamp(sample + universal_perturbation, 0, 1)
                perturbed_output = self.model(perturbed_sample)
                perturbed_pred = perturbed_output.argmax(dim=1, keepdim=True).item()
                perturbed_predictions.append(perturbed_pred)
            
            # 检查是否成功攻击
            if perturbed_pred == target_label:
                successful_attacks += 1
                print(f"样本 {i+1}: 原始标签={original_pred} -> 对抗后标签={perturbed_pred} ✓")
            else:
                print(f"样本 {i+1}: 原始标签={original_pred} -> 对抗后标签={perturbed_pred} ✗")
        
        # 计算成功率
        success_rate = (successful_attacks / len(test_samples)) * 100
        
        print("=" * 60)
        print(f"评估结果: 成功攻击 {successful_attacks}/{len(test_samples)} 个样本")
        print(f"总体成功率: {success_rate:.1f}%")
        
        return success_rate, original_predictions, perturbed_predictions

# 随机选择指定数量的非目标标签样本
def select_random_samples(data_loader, num_samples=100, target_label=8):
    samples = []
    labels = []
    
    print(f"正在随机选择 {num_samples} 个非目标标签样本（排除标签: {target_label}）...")
    
    while len(samples) < num_samples:
        for data, target in data_loader:
            # 只选择标签不等于目标标签的样本
            if target.item() != target_label:
                samples.append(data)
                labels.append(target)
                
                if len(samples) >= num_samples:
                    break
    
    print(f"成功选择 {len(samples)} 个非目标标签样本")
    return samples, labels

# 可视化通用扰动和一些示例
def visualize_results(samples, labels, perturbation, original_preds, perturbed_preds, target_label, num_examples=5):
    print(f"\n正在生成可视化结果...")
    
    # 创建保存目录
    os.makedirs('universal_attack_results', exist_ok=True)
    
    # 可视化通用扰动
    plt.figure(figsize=(6, 6))
    # 归一化扰动以便更好地可视化
    norm_perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
    plt.imshow(norm_perturbation.squeeze().cpu().numpy(), cmap='jet')
    plt.title(f"Universal Perturbation\nTarget: {target_label}")
    plt.colorbar(label='Perturbation Intensity')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'universal_attack_results/universal_perturbation_{target_label}.png')
    plt.close()
    
    # 可视化几个样本的原始图像和对抗图像
    plt.figure(figsize=(15, 6))
    
    # 选择前几个样本或者成功攻击的样本
    selected_indices = []
    # 首先选择成功攻击的样本
    for i in range(len(samples)):
        if perturbed_preds[i] == target_label and original_preds[i] != target_label:
            selected_indices.append(i)
            if len(selected_indices) >= num_examples:
                break
    
    # 如果成功攻击的样本不足，补充一些其他样本
    if len(selected_indices) < num_examples:
        for i in range(len(samples)):
            if i not in selected_indices:
                selected_indices.append(i)
                if len(selected_indices) >= num_examples:
                    break
    
    for i, idx in enumerate(selected_indices):
        sample = samples[idx]
        
        # 原始图像
        plt.subplot(2, num_examples, i + 1)
        plt.imshow(sample.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Original\nLabel: {labels[idx].item()}\nPred: {original_preds[idx]}")
        plt.axis('off')
        
        # 对抗图像
        perturbed_sample = torch.clamp(sample.to(perturbation.device) + perturbation, 0, 1)
        plt.subplot(2, num_examples, i + 1 + num_examples)
        plt.imshow(perturbed_sample.squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Perturbed\nPred: {perturbed_preds[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'universal_attack_results/example_comparison_{target_label}.png')
    plt.close()
    
    print(f"可视化结果已保存到 'universal_attack_results' 文件夹")

# 主函数
def main(pretrained_model, use_cuda=True, epsilon=0.05, num_epochs=10, lr=0.01, 
         target_label=8, num_samples=100):
    # 设置设备
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")
    
    # 加载MNIST测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=1, shuffle=True)
    
    # 初始化模型
    model = Net(10).to(device)
    
    # 尝试加载预训练模型
    try:
        # 尝试直接加载，如果失败则尝试加载state_dict键
        try:
            model.load_state_dict(torch.load(pretrained_model, map_location=device))
        except:
            model.load_state_dict(torch.load(pretrained_model, map_location=device)['state_dict'])
        print("成功加载预训练模型")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("使用随机初始化的模型")
    
    # 随机选择样本
    # 在main函数中修改调用select_random_samples的部分
    # 随机选择非目标标签样本
    samples, labels = select_random_samples(test_loader, num_samples, target_label)
    
    # 初始化攻击器
    attacker = UniversalAttack(model, device, epsilon, num_epochs, lr)
    
    # 生成通用对抗扰动
    universal_perturbation = attacker.generate(samples, target_label)
    
    # 评估通用扰动的有效性
    success_rate, original_preds, perturbed_preds = attacker.evaluate(
        samples, universal_perturbation, target_label
    )
    
    # 保存通用扰动
    os.makedirs('universal_attack_results', exist_ok=True)
    torch.save(universal_perturbation, f'universal_attack_results/universal_perturbation_{target_label}.pt')
    print(f"通用扰动已保存到 'universal_attack_results/universal_perturbation_{target_label}.pt'")
    
    # 可视化结果
    visualize_results(samples, labels, universal_perturbation, 
                    original_preds, perturbed_preds, target_label)
    
    # 输出最终统计
    print("\n" + "=" * 60)
    print("通用对抗样本攻击统计:")
    print(f"目标标签: {target_label}")
    print(f"测试样本总数: {num_samples}")
    print(f"成功攻击样本数: {int(success_rate * num_samples / 100)}")
    print(f"攻击成功率: {success_rate:.1f}%")
    print("=" * 60)

if __name__ == '__main__':
    # 模型路径
    pretrained_model = "MNIST_badnets.pth"  # 如果文件不存在，会使用随机初始化的模型
    
    # 攻击参数
    use_cuda = True
    epsilon = 0.05  # 扰动大小限制
    num_epochs = 100  # 训练轮次
    lr = 0.1        # 学习率
    target_label = 8 # 目标标签
    num_samples = 100 # 样本数量
    
    main(pretrained_model, use_cuda, epsilon, num_epochs, lr, target_label, num_samples)