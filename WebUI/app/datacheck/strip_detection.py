#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STRIP算法实现 - MNIST数据集后门检测
功能：
1. 读取MNIST训练集
2. 取1%标签不为8的数据，添加5*5棋盘格后门，标签改为8
3. 再取1%的正常数据作为参考
4. 使用STRIP算法：将样本与参考数据平均叠加
5. 输入MNIST_BadNets.pth模型
6. 计算被预测为8的样本占比
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LeNet模型定义 - 与generate_trigger.py中的模型结构保持一致
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class STRIPDetector:
    """STRIP算法后门检测器"""
    
    def __init__(self, data_dir='../data', poison_ratio=0.01, reference_ratio=0.01,
                 target_label=8, num_strip_samples=100, model_path='../ag/MNIST_badnets.pth'):
        """
        初始化STRIP检测器
        
        参数:
            data_dir: MNIST数据集存储目录
            poison_ratio: 投毒比例
            reference_ratio: 参考样本比例
            target_label: 目标标签
            num_strip_samples: STRIP算法中每个样本叠加的随机图像数量
            model_path: 预训练模型路径
        """
        self.data_dir = data_dir
        self.poison_ratio = poison_ratio
        self.reference_ratio = reference_ratio
        self.target_label = target_label
        self.num_strip_samples = num_strip_samples
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
    def load_dataset(self):
        """加载MNIST数据集"""
        logger.info("加载MNIST数据集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 加载训练集用于投毒和参考样本
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        # 加载测试集用于评估
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        logger.info(f"训练集大小: {len(self.train_dataset)}")
        logger.info(f"测试集大小: {len(self.test_dataset)}")
        return self.train_dataset, self.test_dataset
    
    def load_model(self):
        """加载预训练模型"""
        logger.info(f"加载模型: {self.model_path}")
        model = Net(num_classes=10).to(self.device)
        
        try:
            # 尝试加载模型，如果保存格式不同则尝试多种方式
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
        
        model.eval()
        return model
    
    def create_chessboard_trigger(self, size=5):
        """创建5x5棋盘格trigger"""
        trigger = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                # 创建黑白交替的棋盘格
                if (i + j) % 2 == 0:
                    trigger[i, j] = 1.0  # 白色
                else:
                    trigger[i, j] = 0.0  # 黑色
        return trigger
    
    def create_poisoned_dataset(self):
        """
        创建投毒数据集：
        1. 取1%标签不为8的数据，添加棋盘格后门，标签改为8
        2. 再取1%的正常数据作为参考
        """
        logger.info("创建投毒数据集...")
        
        # 获取非目标标签的样本索引
        non_target_indices = [i for i, (_, label) in enumerate(self.test_dataset) 
                             if label != self.target_label]
        logger.info(f"非目标标签({self.target_label})样本数量: {len(non_target_indices)}")
        
        # 计算需要投毒的样本数量
        total_samples = len(self.test_dataset)
        num_poisoned = int(total_samples * self.poison_ratio)
        num_reference = int(total_samples * self.reference_ratio)
        logger.info(f"需要投毒的样本数量: {num_poisoned}")
        logger.info(f"参考样本数量: {num_reference}")
        
        # 随机选择要投毒的样本
        np.random.seed(42)  # 设置随机种子确保结果可复现
        poisoned_indices = np.random.choice(non_target_indices, size=num_poisoned, replace=False)
        
        # 从所有样本中随机选择参考样本
        all_indices = list(range(total_samples))
        # 确保参考样本不包含被投毒的样本
        available_indices = list(set(all_indices) - set(poisoned_indices))
        reference_indices = np.random.choice(available_indices, size=num_reference, replace=False)
        
        # 创建棋盘格trigger
        trigger = self.create_chessboard_trigger(size=5)
        
        # 创建投毒样本和参考样本
        poisoned_samples = []
        reference_samples = []
        original_labels = []
        
        # 处理投毒样本
        for idx in poisoned_indices:
            img, original_label = self.test_dataset[idx]
            img_np = img.squeeze().numpy()
            
            # 在右上角添加棋盘格
            img_np[0:5, 23:28] = trigger
            
            # 转换回张量
            img_poisoned = torch.from_numpy(img_np).unsqueeze(0)
            
            poisoned_samples.append(img_poisoned)
            original_labels.append(original_label)
        
        # 处理参考样本
        for idx in reference_indices:
            img, _ = self.test_dataset[idx]
            reference_samples.append(img)
        
        logger.info(f"成功创建 {len(poisoned_samples)} 个投毒样本")
        logger.info(f"成功创建 {len(reference_samples)} 个参考样本")
        
        return poisoned_samples, reference_samples, original_labels, poisoned_indices
    
    def strip_detection(self, model, test_samples, reference_samples):
        """
        使用STRIP算法检测后门样本
        
        参数:
            model: 训练好的模型
            test_samples: 要测试的样本
            reference_samples: 用于叠加的参考样本
        
        返回:
            results: 检测结果，包含每个样本的预测情况
        """
        logger.info("执行STRIP算法检测...")
        results = []
        
        with torch.no_grad():
            for i, sample in enumerate(test_samples):
                # 记录原始样本的预测
                original_output = model(sample.to(self.device))
                original_probs = torch.exp(original_output)
                original_pred = torch.argmax(original_probs).item()
                original_prob_8 = original_probs[0, self.target_label].item()
                
                # 使用STRIP算法：与随机参考样本叠加
                predictions = []
                
                for _ in range(min(self.num_strip_samples, len(reference_samples))):
                    # 随机选择一个参考样本
                    ref_idx = np.random.randint(0, len(reference_samples))
                    ref_sample = reference_samples[ref_idx]
                    
                    # 平均叠加样本
                    combined = (sample + ref_sample) / 2.0
                    
                    # 模型预测
                    output = model(combined.to(self.device))
                    prob = torch.exp(output)
                    pred = torch.argmax(prob).item()
                    predictions.append(pred)
                
                # 计算被预测为8的样本占比
                pred_8_count = predictions.count(self.target_label)
                pred_8_ratio = pred_8_count / len(predictions)
                
                results.append({
                    'original_pred': original_pred,
                    'original_prob_8': original_prob_8,
                    'pred_8_ratio': pred_8_ratio,
                    'is_predicted_8': original_pred == self.target_label
                })
                
                if (i + 1) % 100 == 0 or i == len(test_samples) - 1:
                    logger.info(f"处理样本 {i+1}/{len(test_samples)}")
        
        return results
    
    def analyze_results(self, results, original_labels):
        """
        分析STRIP检测结果
        
        参数:
            results: STRIP检测结果
            original_labels: 原始标签
        """
        logger.info("分析检测结果...")
        
        # 统计被预测为8的样本数量
        predicted_8_count = sum(1 for r in results if r['is_predicted_8'])
        total_samples = len(results)
        pred_8_ratio = predicted_8_count / total_samples
        
        logger.info(f"检测结果统计:")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"被预测为8的样本数: {predicted_8_count}")
        logger.info(f"被预测为8的样本占比: {pred_8_ratio:.4f} ({pred_8_ratio*100:.2f}%)")
        
        # 分析STRIP比率分布
        strip_ratios = [r['pred_8_ratio'] for r in results]
        logger.info(f"STRIP预测8比率平均值: {np.mean(strip_ratios):.4f}")
        logger.info(f"STRIP预测8比率标准差: {np.std(strip_ratios):.4f}")
        logger.info(f"STRIP预测8比率最大值: {np.max(strip_ratios):.4f}")
        logger.info(f"STRIP预测8比率最小值: {np.min(strip_ratios):.4f}")
        
        # 计算后门攻击成功率
        success_count = sum(1 for r in results if r['is_predicted_8'])
        attack_success_rate = success_count / total_samples
        logger.info(f"后门攻击成功率: {attack_success_rate:.4f} ({attack_success_rate*100:.2f}%)")
        
        return {
            'total_samples': total_samples,
            'predicted_8_count': predicted_8_count,
            'pred_8_ratio': pred_8_ratio,
            'strip_ratios_mean': np.mean(strip_ratios),
            'strip_ratios_std': np.std(strip_ratios),
            'attack_success_rate': attack_success_rate
        }
    
    def run(self):
        """运行整个STRIP检测流程"""
        logger.info("开始STRIP后门检测流程...")
        
        # 1. 加载数据集
        self.load_dataset()
        
        # 2. 加载模型
        model = self.load_model()
        
        # 3. 创建投毒数据集
        poisoned_samples, reference_samples, original_labels, _ = self.create_poisoned_dataset()
        
        # 4. 执行STRIP检测
        results = self.strip_detection(model, poisoned_samples, reference_samples)
        
        # 5. 分析结果
        analysis = self.analyze_results(results, original_labels)
        
        logger.info("STRIP后门检测流程完成！")
        return results, analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='STRIP算法实现 - MNIST数据集后门检测')
    parser.add_argument('--data_dir', type=str, default='../data', help='MNIST数据集存储目录')
    parser.add_argument('--poison_ratio', type=float, default=0.01, help='投毒比例')
    parser.add_argument('--reference_ratio', type=float, default=0.01, help='参考样本比例')
    parser.add_argument('--target_label', type=int, default=8, help='目标标签')
    parser.add_argument('--num_strip_samples', type=int, default=100, help='每个样本叠加的随机图像数量')
    parser.add_argument('--model_path', type=str, default='../ag/MNIST_badnets.pth', help='预训练模型路径')
    
    args = parser.parse_args()
    
    # 创建并运行STRIP检测器
    detector = STRIPDetector(
        data_dir=args.data_dir,
        poison_ratio=args.poison_ratio,
        reference_ratio=args.reference_ratio,
        target_label=args.target_label,
        num_strip_samples=args.num_strip_samples,
        model_path=args.model_path
    )
    
    results, analysis = detector.run()
    
    # 打印最终结果
    # print("\n=== 最终检测结果 ===")
    # print(f"被预测为8的样本占比: {analysis['pred_8_ratio']*100:.2f}%")
    # print(f"后门攻击成功率: {analysis['attack_success_rate']*100:.2f}%")
    print(f"{analysis['pred_8_ratio']*100:.2f}",end="")