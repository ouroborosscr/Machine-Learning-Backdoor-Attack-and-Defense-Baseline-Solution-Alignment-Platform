import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import os
import sys
from PIL import Image

class SampleGenerator:
    def __init__(self):
        # 确保输出目录存在
        self.output_dir = 'sample_images'
        self.mnist_dir = os.path.join(self.output_dir, 'mnist')
        self.gtsrb_dir = os.path.join(self.output_dir, 'gtsrb')
        
        # 创建目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.mnist_dir, exist_ok=True)
        os.makedirs(self.gtsrb_dir, exist_ok=True)
        
        # 定义数据加载的变换
        self.mnist_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.gtsrb_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
    
    def generate_mnist_samples(self):
        """为MNIST数据集的每个标签生成一个样本图片"""
        print("开始生成MNIST样本图片...")
        
        try:
            # 加载MNIST测试集
            mnist_test = datasets.MNIST(
                root='../data',
                train=False,
                download=True,
                transform=self.mnist_transform
            )
            
            # 创建数据加载器
            test_loader = torch.utils.data.DataLoader(
                mnist_test,
                batch_size=1,
                shuffle=True
            )
            
            # 记录已找到的标签
            found_labels = set()
            total_labels = 10  # MNIST有10个标签
            
            # 遍历数据集直到找到所有标签的样本
            for i, (image, label) in enumerate(test_loader):
                label_value = label.item()
                
                if label_value not in found_labels:
                    # 保存样本图片
                    image_np = image.squeeze().numpy()
                    plt.figure(figsize=(2, 2))
                    plt.imshow(image_np, cmap='gray')
                    plt.title(f'Label: {label_value}')
                    plt.axis('off')
                    
                    # 保存图片
                    save_path = os.path.join(self.mnist_dir, f'mnist_label_{label_value}.png')
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    found_labels.add(label_value)
                    print(f"已保存MNIST标签 {label_value} 的样本")
                    
                    # 如果所有标签都找到了，就停止
                    if len(found_labels) == total_labels:
                        break
            
            print(f"MNIST样本生成完成！共生成 {len(found_labels)} 个样本。")
            print(f"样本保存在: {self.mnist_dir}")
            
        except Exception as e:
            print(f"生成MNIST样本时出错: {e}")
    
    def generate_gtsrb_samples(self):
        """为GTSRB数据集的每个标签生成一个样本图片"""
        print("\n开始生成GTSRB样本图片...")
        
        try:
            # 加载GTSRB测试集
            gtsrb_test = datasets.GTSRB(
                root='../data',
                split="test",
                download=True,
                transform=self.gtsrb_transform
            )
            
            # 创建数据加载器
            test_loader = torch.utils.data.DataLoader(
                gtsrb_test,
                batch_size=1,
                shuffle=True
            )
            
            # 记录已找到的标签
            found_labels = set()
            total_labels = 43  # GTSRB有43个标签
            
            # 遍历数据集直到找到所有标签的样本
            for i, (image, label) in enumerate(test_loader):
                label_value = label.item()
                
                if label_value not in found_labels:
                    # 保存样本图片
                    image_np = image.squeeze().permute(1, 2, 0).numpy()
                    
                    # 反归一化（如果需要的话）
                    # GTSRB的默认归一化参数
                    mean = np.array([0.3403, 0.3121, 0.3214])
                    std = np.array([0.2724, 0.2608, 0.2669])
                    image_np = std * image_np + mean
                    image_np = np.clip(image_np, 0, 1)
                    
                    plt.figure(figsize=(2, 2))
                    plt.imshow(image_np)
                    plt.title(f'Label: {label_value}')
                    plt.axis('off')
                    
                    # 保存图片
                    save_path = os.path.join(self.gtsrb_dir, f'gtsrb_label_{label_value}.png')
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    found_labels.add(label_value)
                    print(f"已保存GTSRB标签 {label_value} 的样本")
                    
                    # 如果所有标签都找到了，就停止
                    if len(found_labels) == total_labels:
                        break
                    
                    # 为了避免处理时间过长，可以限制一下处理的样本数量
                    if len(found_labels) % 10 == 0:
                        print(f"已找到 {len(found_labels)} 个不同的GTSRB标签...")
            
            print(f"GTSRB样本生成完成！共生成 {len(found_labels)} 个样本。")
            print(f"样本保存在: {self.gtsrb_dir}")
            
        except Exception as e:
            print(f"生成GTSRB样本时出错: {e}")
    
    def generate_all_samples(self):
        """生成所有数据集的样本"""
        print("=== 样本生成器 ===")
        print(f"样本将保存在: {self.output_dir}")
        
        # 生成MNIST样本
        self.generate_mnist_samples()
        
        # 生成GTSRB样本
        self.generate_gtsrb_samples()
        
        print("\n所有样本生成完成！")

# 添加一个用于生成混合样本的函数（可以从验证集中随机选择）
def generate_mixed_samples(num_samples=5):
    """生成混合的MNIST和GTSRB样本"""
    generator = SampleGenerator()
    mixed_dir = os.path.join(generator.output_dir, 'mixed')
    os.makedirs(mixed_dir, exist_ok=True)
    
    print(f"\n开始生成混合样本...")
    
    # 生成MNIST混合样本
    try:
        mnist_test = datasets.MNIST(
            root='../data',
            train=False,
            download=True,
            transform=generator.mnist_transform
        )
        
        # 随机选择样本
        indices = torch.randperm(len(mnist_test))[:num_samples]
        for i, idx in enumerate(indices):
            image, label = mnist_test[idx]
            image_np = image.squeeze().numpy()
            plt.figure(figsize=(2, 2))
            plt.imshow(image_np, cmap='gray')
            plt.title(f'MNIST - Label: {label}')
            plt.axis('off')
            save_path = os.path.join(mixed_dir, f'mixed_mnist_{i}_label_{label}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    except Exception as e:
        print(f"生成MNIST混合样本时出错: {e}")
    
    # 生成GTSRB混合样本
    try:
        gtsrb_test = datasets.GTSRB(
            root='../data',
            split="test",
            download=True,
            transform=generator.gtsrb_transform
        )
        
        # 随机选择样本
        indices = torch.randperm(len(gtsrb_test))[:num_samples]
        for i, idx in enumerate(indices):
            image, label = gtsrb_test[idx]
            image_np = image.squeeze().permute(1, 2, 0).numpy()
            # 反归一化
            mean = np.array([0.3403, 0.3121, 0.3214])
            std = np.array([0.2724, 0.2608, 0.2669])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)
            
            plt.figure(figsize=(2, 2))
            plt.imshow(image_np)
            plt.title(f'GTSRB - Label: {label}')
            plt.axis('off')
            save_path = os.path.join(mixed_dir, f'mixed_gtsrb_{i}_label_{label}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    except Exception as e:
        print(f"生成GTSRB混合样本时出错: {e}")
    
    print(f"混合样本生成完成！保存在: {mixed_dir}")

def main():
    """主函数"""
    generator = SampleGenerator()
    
    print("=== 数据集样本生成器 ===")
    print("1. 为每个MNIST标签生成一个样本")
    print("2. 为每个GTSRB标签生成一个样本")
    print("3. 生成所有数据集的样本")
    print("4. 生成混合样本（随机选择）")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("请选择操作 (1-4): ")
    
    if choice == '1':
        generator.generate_mnist_samples()
    elif choice == '2':
        generator.generate_gtsrb_samples()
    elif choice == '3':
        generator.generate_all_samples()
    elif choice == '4':
        num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        generate_mixed_samples(num)
    else:
        print("无效的选择！")

if __name__ == "__main__":
    main()