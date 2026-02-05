import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import os
import sys
import torch.nn as nn
import torch.nn.functional as F

# 添加modelcheck目录到路径，以便导入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/modelcheck')

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

    # def features(self, x):
    #     """
    #     Get latent representation, eg logit layer.
    #     :param x:
    #     :return:
    #     """
    #     raise NotImplemented

    def forward(self, x, latent=False):
        raise NotImplemented

class GTSRBNet(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv5 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv6 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.fc1 = nn.Linear(4 * 4 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def features(self, x):
        x = F.relu(self.conv1(x))# 32 x 32 x 32
        x = F.relu(self.conv2(x))# 32 x 32 x 32
        x = F.max_pool2d(x, 2, 2)# 32 x 16 x 16
        x = F.relu(self.conv3(x))# 64 x 16 x 16
        x = F.relu(self.conv4(x))# 64 x 16 x 16
        x = F.max_pool2d(x, 2, 2)# 64 x 8 x 8
        x = F.relu(self.conv5(x))# 128 x 8 x 8
        x = F.relu(self.conv6(x))# 128 x 8 x 8
        x = F.max_pool2d(x, 2, 2)# 128 x 4 x 4
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))# 32 x 32 x 32
        x = F.relu(self.conv2(x))# 32 x 32 x 32
        x = F.max_pool2d(x, 2, 2)# 32 x 16 x 16
        x = F.relu(self.conv3(x))# 64 x 16 x 16
        x = F.relu(self.conv4(x))# 64 x 16 x 16
        x = F.max_pool2d(x, 2, 2)# 64 x 8 x 8
        x = F.relu(self.conv5(x))# 128 x 8 x 8
        x = F.relu(self.conv6(x))# 128 x 8 x 8
        x = F.max_pool2d(x, 2, 2)# 128 x 4 x 4
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out


# LeNet Model definition for MNIST
class Net(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

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

# 数据集类别名称映射
MNIST_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# GTSRB类别名称（简化版本，实际有43个类别）
GTSRB_CLASSES = [str(i) for i in range(43)]

class ModelValidator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mnist_model = None
        self.gtsrb_model = None
        self.mnist_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        self.gtsrb_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
        ])

    def load_models(self):
        """加载MNIST和GTSRB模型"""
        # 尝试加载MNIST模型
        try:
            # 首先查找模型文件的可能位置
            model_dirs = [
                '../../models',
                '../models',
                'models'
            ]
            
            mnist_model_path = None
            for model_dir in model_dirs:
                path = os.path.join(model_dir, 'MNIST_BadNets.pth')
                if os.path.exists(path):
                    mnist_model_path = path
                    break
            
            if mnist_model_path:
                self.mnist_model = Net(10).to(self.device)
                self.mnist_model.load_state_dict(torch.load(mnist_model_path, map_location=self.device)['state_dict'])
                self.mnist_model.eval()
                print(f"成功加载MNIST模型: {mnist_model_path}")
            else:
                print("警告: 未找到MNIST模型文件")
        except Exception as e:
            print(f"加载MNIST模型时出错: {e}")
        
        # 尝试加载GTSRB模型
        try:
            gtsrb_model_path = None
            for model_dir in model_dirs:
                path = os.path.join(model_dir, 'GTSRB_SIN.pth')
                if os.path.exists(path):
                    gtsrb_model_path = path
                    break
            
            if gtsrb_model_path:
                self.gtsrb_model = GTSRBNet(43).to(self.device)
                self.gtsrb_model.load_state_dict(torch.load(gtsrb_model_path, map_location=self.device)['state_dict'])
                self.gtsrb_model.eval()
                print(f"成功加载GTSRB模型: {gtsrb_model_path}")
            else:
                print("警告: 未找到GTSRB模型文件")
        except Exception as e:
            print(f"加载GTSRB模型时出错: {e}")
    
    def preprocess_image(self, image_path, dataset_type):
        """预处理输入图像"""
        try:
            image = Image.open(image_path)
            
            # 根据数据集类型选择不同的变换
            if dataset_type.lower() == 'mnist':
                # 转换为灰度图
                if image.mode != 'L':
                    image = image.convert('L')
                transformed_image = self.mnist_transform(image)
            elif dataset_type.lower() == 'gtsrb':
                # 确保是RGB图
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                transformed_image = self.gtsrb_transform(image)
            else:
                raise ValueError(f"不支持的数据集类型: {dataset_type}")
            
            # 添加批次维度
            return transformed_image.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"图像处理错误: {e}")
            return None
    
    def predict(self, image_tensor, dataset_type, top_k=3):
        """使用模型进行预测并返回前k个结果"""
        if image_tensor is None:
            return None
        
        with torch.no_grad():
            if dataset_type.lower() == 'mnist' and self.mnist_model:
                model = self.mnist_model
                classes = MNIST_CLASSES
            elif dataset_type.lower() == 'gtsrb' and self.gtsrb_model:
                model = self.gtsrb_model
                classes = GTSRB_CLASSES
            else:
                print(f"模型未加载或不支持的数据集类型: {dataset_type}")
                return None
            
            # 获取预测结果
            output = model(image_tensor)
            # 转换为概率
            probabilities = torch.exp(output)
            # 获取前k个预测结果
            top_probs, top_indices = probabilities.topk(top_k, dim=1)
            
            # 构建结果列表
            results = []
            for i in range(top_k):
                class_index = top_indices[0, i].item()
                probability = top_probs[0, i].item()
                results.append({
                    'class': classes[class_index],
                    'probability': probability,
                    'index': class_index
                })
            
            return results
    
    def validate_image(self, image_path, dataset_type):
        """验证指定图像"""
        print(f"验证图像: {image_path}")
        print(f"数据集类型: {dataset_type}")
        
        # 预处理图像
        image_tensor = self.preprocess_image(image_path, dataset_type)
        if image_tensor is None:
            print("无法处理图像")
            return
        
        # 进行预测
        results = self.predict(image_tensor, dataset_type)
        if results:
            print("\n预测结果（前3名）:")
            print("-" * 40)
            for i, result in enumerate(results, 1):
                print(f"排名 {i}:")
                print(f"  类别: {result['class']}")
                print(f"  概率: {result['probability']:.6f} ({result['probability']*100:.2f}%)")
                print(f"  索引: {result['index']}")
            print("-" * 40)
        else:
            print("预测失败")

def main():
    # 创建验证器实例
    validator = ModelValidator()
    
    # 加载模型
    validator.load_models()
    
    # 示例用法
    print("\n=== 模型验证工具 ===")
    print("请输入图像路径和数据集类型（mnist/gtsrb）进行验证")
    print("示例: python model_validation.py path/to/image.png mnist")
    
    if len(sys.argv) > 2:
        image_path = sys.argv[1]
        dataset_type = sys.argv[2]
        validator.validate_image(image_path, dataset_type)
    else:
        print("\n未提供命令行参数，使用交互式输入:")
        image_path = input("请输入图像路径: ")
        dataset_type = input("请输入数据集类型 (mnist/gtsrb): ")
        validator.validate_image(image_path, dataset_type)

if __name__ == "__main__":
    main()