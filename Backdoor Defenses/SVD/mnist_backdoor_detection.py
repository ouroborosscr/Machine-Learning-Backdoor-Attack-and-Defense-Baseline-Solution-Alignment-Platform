#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST数据集后门注入与奇异值检测
功能：
1. 读取MNIST训练集
2. 随机选择1%且标签不为8的数据
3. 在样本右上角添加5*5棋盘格
4. 将修改后的样本标签改为8
5. 混合修改后的样本与99%正常样本
6. 计算样本奇异值
7. 使用奇异值检测异常样本
"""

import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MNISTBackdoorDetector:
    """MNIST数据集后门检测类"""
    
    def __init__(self, data_dir='../data', poison_ratio=0.01, target_label=8):
        """
        初始化
        
        参数:
            data_dir: MNIST数据集存储目录
            poison_ratio: 投毒比例
            target_label: 目标标签
        """
        self.data_dir = data_dir
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
    def load_dataset(self):
        """加载MNIST训练集"""
        logger.info("加载MNIST训练集...")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        logger.info(f"训练集大小: {len(self.train_dataset)}")
        return self.train_dataset
    
    def create_chessboard_trigger(self, size=5):
        """
        创建5x5棋盘格trigger
        
        参数:
            size: 棋盘格大小
        
        返回:
            棋盘格trigger数组
        """
        trigger = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                # 创建黑白交替的棋盘格
                if (i + j) % 2 == 0:
                    trigger[i, j] = 1.0  # 白色
                else:
                    trigger[i, j] = 0.0  # 黑色
        return trigger
    
    def select_and_poison_samples(self):
        """
        选择1%非目标标签样本并投毒
        
        返回:
            poisoned_indices: 被投毒的样本索引
            clean_indices: 未被投毒的样本索引
            poisoned_data: 投毒后的数据
            is_poisoned: 标记样本是否被投毒的数组
        """
        # 获取非目标标签的样本索引
        non_target_indices = [i for i, (_, label) in enumerate(self.train_dataset) 
                             if label != self.target_label]
        logger.info(f"非目标标签({self.target_label})样本数量: {len(non_target_indices)}")
        
        # 计算需要投毒的样本数量
        total_samples = len(self.train_dataset)
        num_poisoned = int(total_samples * self.poison_ratio)
        logger.info(f"需要投毒的样本数量: {num_poisoned}")
        
        # 随机选择要投毒的样本
        np.random.seed(42)  # 设置随机种子确保结果可复现
        poisoned_indices = np.random.choice(non_target_indices, size=num_poisoned, replace=False)
        
        # 创建所有未被投毒的样本索引
        all_indices = set(range(total_samples))
        poisoned_set = set(poisoned_indices)
        clean_indices = list(all_indices - poisoned_set)
        
        # 创建棋盘格trigger
        trigger = self.create_chessboard_trigger(size=5)
        
        # 复制数据集用于投毒
        poisoned_data = []
        is_poisoned = np.zeros(total_samples, dtype=bool)
        
        # 标记被投毒的样本
        for idx in poisoned_indices:
            is_poisoned[idx] = True
        
        # 处理数据
        for i in range(total_samples):
            img, label = self.train_dataset[i]
            
            # 如果是被选中的投毒样本
            if i in poisoned_set:
                # 将图像转换为numpy数组
                img_np = img.squeeze().numpy()
                
                # 在右上角添加棋盘格（28-5=23的位置开始）
                img_np[0:5, 23:28] = trigger
                
                # 转换回张量
                img = torch.from_numpy(img_np).unsqueeze(0)
                
                # 修改标签为目标标签
                label = self.target_label
            
            poisoned_data.append((img, label))
        
        logger.info(f"成功投毒 {num_poisoned} 个样本")
        return poisoned_indices, clean_indices, poisoned_data, is_poisoned
    
    def calculate_singular_values(self, poisoned_data, n_components=20):
        """
        计算样本的奇异值
        
        参数:
            poisoned_data: 投毒后的数据
            n_components: 保留的奇异值数量
        
        返回:
            singular_values: 奇异值矩阵
        """
        logger.info("计算样本奇异值...")
        
        # 准备数据矩阵
        num_samples = len(poisoned_data)
        data_matrix = np.zeros((num_samples, 28*28))
        
        # 将图像展平
        for i, (img, _) in enumerate(poisoned_data):
            data_matrix[i] = img.view(-1).numpy()
        
        # 标准化数据
        scaler = StandardScaler()
        data_matrix_scaled = scaler.fit_transform(data_matrix)
        
        # 使用TruncatedSVD计算奇异值
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(data_matrix_scaled)
        
        # 计算每个样本在SVD空间中的表示
        transformed_data = svd.transform(data_matrix_scaled)
        
        # 计算每个样本的奇异值能量（各奇异值平方和）
        singular_values_energy = np.sum(transformed_data**2, axis=1)
        
        logger.info(f"完成奇异值计算，保留前{n_components}个奇异值")
        return singular_values_energy
    
    def detect_poisoned_samples(self, singular_values_energy, is_poisoned):
        """
        使用奇异值能量检测被投毒的样本
        
        参数:
            singular_values_energy: 奇异值能量数组
            is_poisoned: 标记样本是否被投毒的真实数组
        
        返回:
            检测结果和评估指标
        """
        logger.info("使用奇异值能量检测异常样本...")
        
        # 使用奇异值能量作为特征
        X = singular_values_energy.reshape(-1, 1)
        y_true = is_poisoned.astype(int)
        
        # 计算精度-召回曲线
        precision, recall, thresholds = precision_recall_curve(y_true, X)
        pr_auc = auc(recall, precision)
        
        # 找到最佳阈值（这里使用简单的方法，实际可能需要更复杂的优化）
        f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_recall = recall[best_idx]  # 添加最佳查全率
        best_precision = precision[best_idx]  # 添加最佳精确率
        
        # 预测结果
        y_pred = (X > best_threshold).astype(int).flatten()
        
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算精确率、查全率和F1分数
        true_positives = np.sum(y_pred & y_true)
        false_positives = np.sum(y_pred & ~y_true)
        false_negatives = np.sum(~y_pred & y_true)
        
        # 计算精确率
        precision_value = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # 计算查全率
        recall_value = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        logger.info(f"检测结果 - AUC: {pr_auc:.4f}, 最佳F1: {best_f1:.4f}, 准确率: {accuracy:.4f}")
        print(accuracy,end='')
        logger.info(f"最佳精确率: {best_precision:.4f}, 最佳查全率: {best_recall:.4f}")
        logger.info(f"实际精确率: {precision_value:.4f}, 实际查全率: {recall_value:.4f}")
        
        # 分析奇异值能量分布
        poisoned_energy = singular_values_energy[is_poisoned]
        clean_energy = singular_values_energy[~is_poisoned]
        
        logger.info(f"被投毒样本平均奇异值能量: {np.mean(poisoned_energy):.4f}")
        logger.info(f"干净样本平均奇异值能量: {np.mean(clean_energy):.4f}")
        
        return {
            'precision': precision,
            'recall': recall,
            'pr_auc': pr_auc,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'best_recall': best_recall,  # 添加最佳查全率
            'best_precision': best_precision,  # 添加最佳精确率
            'accuracy': accuracy,
            'precision_value': precision_value,  # 添加实际精确率
            'recall_value': recall_value,  # 添加实际查全率
            'y_pred': y_pred,
            'poisoned_energy': poisoned_energy,
            'clean_energy': clean_energy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def visualize_results(self, poisoned_data, poisoned_indices, detection_results):
        """
        可视化结果
        
        参数:
            poisoned_data: 投毒后的数据
            poisoned_indices: 被投毒的样本索引
            detection_results: 检测结果
        """
        logger.info("可视化结果...")
        
        # 创建保存结果的目录
        os.makedirs('results', exist_ok=True)
        
        # 1. 显示一些被投毒的样本
        plt.figure(figsize=(10, 5))
        for i in range(min(5, len(poisoned_indices))):
            idx = poisoned_indices[i]
            img, label = poisoned_data[idx]
            plt.subplot(1, 5, i+1)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            plt.title(f'Label: {label}')
            plt.axis('off')
        plt.savefig('results/poisoned_samples.png')
        plt.close()
        
        # 2. 绘制奇异值能量分布图
        plt.figure(figsize=(10, 6))
        plt.hist(detection_results['clean_energy'], bins=50, alpha=0.5, label='Clean Samples')
        plt.hist(detection_results['poisoned_energy'], bins=50, alpha=0.5, label='Poisoned Samples')
        plt.axvline(x=detection_results['best_threshold'], color='r', linestyle='--', label=f'Best Threshold')
        plt.xlabel('Singular Value Energy')
        plt.ylabel('Frequency')
        plt.title('Distribution of Singular Value Energy')
        plt.legend()
        plt.savefig('results/singular_value_energy_distribution.png')
        plt.close()
        
        # 3. 绘制精度-召回曲线
        plt.figure(figsize=(10, 6))
        plt.plot(detection_results['recall'], detection_results['precision'], marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (AUC = {detection_results["pr_auc"]:.4f})')
        plt.grid(True)
        plt.savefig('results/precision_recall_curve.png')
        plt.close()
        
        logger.info("结果可视化完成，保存在results目录中")
    
    def run(self):
        """运行整个后门注入和检测流程"""
        logger.info("开始MNIST后门注入与检测流程...")
        
        # 1. 加载数据集
        self.load_dataset()
        
        # 2. 选择样本并投毒
        poisoned_indices, clean_indices, poisoned_data, is_poisoned = self.select_and_poison_samples()
        
        # 3. 计算奇异值
        singular_values_energy = self.calculate_singular_values(poisoned_data)
        
        # 4. 检测被投毒的样本
        detection_results = self.detect_poisoned_samples(singular_values_energy, is_poisoned)
        
        # 5. 可视化结果
        self.visualize_results(poisoned_data, poisoned_indices, detection_results)
        
        # 6. 输出检测到的被投毒样本数量
        num_detected = np.sum(detection_results['y_pred'])
        true_positives = detection_results['true_positives']
        false_positives = detection_results['false_positives']
        false_negatives = detection_results['false_negatives']
        recall_value = detection_results['recall_value']
        
        logger.info(f"检测到 {num_detected} 个异常样本")
        logger.info(f"其中 {true_positives} 个是真正的被投毒样本")
        logger.info(f"{false_positives} 个是误报的干净样本")
        logger.info(f"{false_negatives} 个被投毒样本未被检测到")
        logger.info(f"最终查全率: {recall_value:.4f}")
        logger.info(f"后门注入与检测流程完成！")
        
        return detection_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MNIST数据集后门注入与奇异值检测')
    parser.add_argument('--data_dir', type=str, default='../data', help='MNIST数据集存储目录')
    parser.add_argument('--poison_ratio', type=float, default=0.01, help='投毒比例')
    parser.add_argument('--target_label', type=int, default=8, help='目标标签')
    
    args = parser.parse_args()
    
    # 创建并运行后门检测器
    detector = MNISTBackdoorDetector(
        data_dir=args.data_dir,
        poison_ratio=args.poison_ratio,
        target_label=args.target_label
    )
    detector.run()