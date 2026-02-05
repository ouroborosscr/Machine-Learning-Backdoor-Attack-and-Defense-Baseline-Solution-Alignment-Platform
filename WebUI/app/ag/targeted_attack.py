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
class Model(nn.Module):
    """
    Base class for models with added support for GradCam activation map
    and a SentiNet defense. The GradCam design is taken from:
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    If you are not planning to utilize SentiNet defense just import any model
    you like for your tasks.
    """

    def __init__(self):
        super().__init__()
        self.gradient = None

    def activations_hook(self, grad):
        self.gradient = grad

    def get_gradient(self):
        return self.gradient

    def get_activations(self, x):
        return self.features(x)

    def switch_grads(self, enable=True):
        for i, n in self.named_parameters():
                n.requires_grad_(enable)

    def features(self, x):
        """
        Get latent representation, eg logit layer.
        :param x:
        :return:
        """
        raise NotImplemented

    def forward(self, x, latent=False):
        raise NotImplemented

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

# 定向对抗攻击函数
class TargetedAttack:
    def __init__(self, model, device, epsilon=0.05, num_steps=10):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.model.eval()

    def attack(self, image, target_label):
        """
        使用反向梯度算法，以target_label为目标标签生成对抗样本
        """
        # 确保图像需要梯度
        perturbed_image = image.clone().detach().requires_grad_(True)
        
        # 进行多次迭代
        for _ in range(self.num_steps):
            # 前向传播
            output = self.model(perturbed_image)
            
            # 计算损失（注意：对于目标攻击，我们希望最大化目标标签的损失）
            # 使用负的NLL损失，因为我们希望模型将样本分类为目标标签
            loss = -F.nll_loss(output, torch.tensor([target_label], device=self.device))
            
            # 反向传播计算梯度
            self.model.zero_grad()
            loss.backward()
            
            # 获取梯度并应用扰动
            data_grad = perturbed_image.grad.data
            # 对于目标攻击，我们沿着梯度的方向移动（与非目标攻击相反）
            perturbed_image = perturbed_image + self.epsilon * data_grad.sign()
            
            # 确保图像在有效范围内
            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach().requires_grad_(True)
        
        return perturbed_image

# 随机选择非8标签的MNIST样本
def select_non_eight_samples(data_loader, num_samples=5):
    samples = []
    labels = []
    
    while len(samples) < num_samples:
        for data, target in data_loader:
            # 检查标签是否不为8
            if target.item() != 8:
                samples.append(data)
                labels.append(target)
                
            if len(samples) >= num_samples:
                break
    
    return samples, labels

# 可视化原始图像和对抗图像
def visualize_results(original_images, perturbed_images, original_labels, final_predictions):
    plt.figure(figsize=(15, 6))
    
    for i in range(len(original_images)):
        # 原始图像
        plt.subplot(2, len(original_images), i + 1)
        plt.imshow(original_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Original\nLabel: {original_labels[i].item()}")
        plt.axis('off')
        
        # 对抗图像
        plt.subplot(2, len(original_images), i + 1 + len(original_images))
        # 添加detach()方法来处理需要梯度的张量
        plt.imshow(perturbed_images[i].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title(f"Adversarial\nPred: {final_predictions[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    # 保存结果
    os.makedirs('attack_results', exist_ok=True)
    plt.savefig('attack_results/adversarial_examples.png')
    plt.close()

# 主函数
def main(pretrained_model, use_cuda=True, epsilon=0.05, num_steps=10, target_label=8, num_samples=5):
    # 设置设备
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    
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
        print("Successfully loaded pretrained model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using randomly initialized model instead")
    
    # 初始化攻击器
    attacker = TargetedAttack(model, device, epsilon, num_steps)
    
    # 选择指定数量的标签不为target_label的样本
    print(f"Selecting {num_samples} samples with labels != {target_label}...")
    selected_images, selected_labels = select_non_target_samples(test_loader, target_label, num_samples)
    
    # 存储结果
    perturbed_images = []
    final_predictions = []
    success_count = 0
    
    # 对每个样本进行攻击
    print(f"Performing targeted attacks to label {target_label} with {num_steps} steps...")
    for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
        print(f"Sample {i+1}: Original label = {label.item()}")
        
        # 将图像和标签移至设备
        image = image.to(device)
        label = label.to(device)
        
        # 进行攻击
        perturbed_image = attacker.attack(image, target_label=target_label)
        # 存储前detach张量
        perturbed_images.append(perturbed_image.detach())
        
        # 预测对抗样本
        with torch.no_grad():
            output = model(perturbed_image)
            pred = output.argmax(dim=1, keepdim=True).item()
        
        final_predictions.append(pred)
        
        # 检查是否成功攻击到目标标签
        if pred == target_label:
            success_count += 1
            print(f"  [+] Successfully attacked to {target_label}!")
        else:
            print(f"  [-] Attack failed. Model predicted: {pred}")
    
    # 计算成功率
    success_rate = (success_count / len(selected_images)) * 100
    print(f"\nAttack Summary:")
    print(f"Total samples: {len(selected_images)}")
    print(f"Successfully attacked to {target_label}: {success_count}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # 可视化结果
    visualize_results(selected_images, perturbed_images, selected_labels, final_predictions, target_label)
    print(f"Visualization saved to 'attack_results/adversarial_examples_{target_label}.png'")

# 修改样本选择函数，使其接受目标标签参数
def select_non_target_samples(data_loader, target_label, num_samples=5):
    samples = []
    labels = []
    
    while len(samples) < num_samples:
        for data, target in data_loader:
            # 检查标签是否不等于目标标签
            if target.item() != target_label:
                samples.append(data)
                labels.append(target)
                
            if len(samples) >= num_samples:
                break
    
    return samples, labels

# 修改可视化函数，使其接受目标标签参数
def visualize_results(original_images, perturbed_images, original_labels, final_predictions, target_label):
    plt.figure(figsize=(15, 6))
    
    for i in range(len(original_images)):
        # 原始图像
        plt.subplot(2, len(original_images), i + 1)
        plt.imshow(original_images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Original\nLabel: {original_labels[i].item()}")
        plt.axis('off')
        
        # 对抗图像
        plt.subplot(2, len(original_images), i + 1 + len(original_images))
        plt.imshow(perturbed_images[i].squeeze().detach().cpu().numpy(), cmap='gray')
        plt.title(f"Adversarial\nPred: {final_predictions[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    # 保存结果，文件名包含目标标签
    os.makedirs('attack_results', exist_ok=True)
    plt.savefig(f'attack_results/adversarial_examples_{target_label}.png')
    plt.close()

if __name__ == '__main__':
    # 模型路径
    pretrained_model = "MNIST_badnets.pth"
    
    # 攻击参数
    use_cuda = True
    epsilon = 0.05
    num_steps = 10
    # 目标标签参数
    target_label = 4
    # 样本数量参数
    num_samples = 5
    
    main(pretrained_model, use_cuda, epsilon, num_steps, target_label, num_samples)