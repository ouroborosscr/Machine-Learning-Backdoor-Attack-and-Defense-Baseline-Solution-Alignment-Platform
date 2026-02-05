import argparse
import json
import os
import re
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.nn as nn

# ... (Model, Net, GTSRBNet, GTSRB_LABEL_NAMES, get_model, load_model, predict 函数保持不变) ...

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


# 定义LeNet模型（用于MNIST）
class Net(Model):
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
        return x

# 定义GTSRB模型
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
        # --- 在这里添加打印语句 ---
        # print(f"DEBUG: Shape before view: {x.shape}")
        # print(f"DEBUG: Size before view: {x.numel()}")
        # ---------------------------
        x = x.view(-1, 4 * 4 * 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

# GTSRB标志名映射（简化版）
GTSRB_LABEL_NAMES = {
    0: '限速20', 1: '限速30', 2: '限速50', 3: '限速60', 4: '限速70', 5: '限速80', 
    6: '解除限速80', 7: '限速100', 8: '限速120', 9: '禁止超车', 10: '禁止大型车超车', 
    11: '优先路口', 12: '主路', 13: '让行', 14: '停车', 15: '禁止通行', 16: '禁止大型车通行', 
    17: '禁止进入', 18: '危险', 19: '弯道危险', 20: '左侧弯道', 21: '右侧弯道', 
    22: '连续弯道', 23: '凹凸路面', 24: '路面湿滑', 25: '路面变窄', 26: '施工', 
    27: '交通信号', 28: '行人', 29: '儿童', 30: '自行车', 31: '注意冰块', 32: '野生动物', 
    33: '解除限速', 34: '右转', 35: '左转', 36: '直行', 37: '直行或右转', 38: '直行或左转', 
    39: '靠右', 40: '靠左', 41: '环形交叉', 42: '解除禁止超车', 43: '解除禁止大型车超车'
}

def get_model(dataset_name):
    """根据数据集名获取对应的模型"""
    if dataset_name.upper() == 'MNIST':
        model = Net(num_classes=10)
    elif dataset_name.upper() == 'GTSRB':
        model = GTSRBNet(num_classes=43)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    return model

# 修改load_model函数
def load_model(model_path, dataset_name):
    """加载模型"""
    model = get_model(dataset_name)
    # 设置map_location确保在CPU上也能加载
    
    # 修复 PyTorch 2.6+ 版本中 weights_only=True 导致的加载失败问题
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 检查checkpoint是否为字典且包含state_dict键
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 如果没有state_dict键，直接尝试加载
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict(model, image_tensor, dataset_name, top_k=3):
    """进行预测并返回top k的结果"""
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))  # 添加batch维度
        probabilities = F.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=top_k)
        
        # 构建结果列表
        results = []
        for i in range(top_k):
            idx = top_indices[i].item()
            prob = top_probs[i].item()
            
            # 根据数据集决定标签显示方式
            if dataset_name.upper() == 'GTSRB':
                label = GTSRB_LABEL_NAMES.get(idx, f'未知({idx})')
            else:
                label = str(idx)
            
            results.append({
                "label": label,
                "probability": round(float(prob), 4)
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='使用模型预测样本')
    parser.add_argument('filename', help='文件名，格式为"{数据集名}_{数字}"')
    parser.add_argument('dataset', help='数据集名')
    args = parser.parse_args()
    
    # 验证文件名格式
    pattern = r'^([A-Za-z]+)_([0-9]+)$'
    match = re.match(pattern, args.filename)
    
    if not match:
        # 文件名格式不正确
        error_response = [
            {"error data": 0, "": 0},
            {"error data": 0, "": 0}
        ]
        print(json.dumps(error_response, ensure_ascii=False))
        return
    
    file_dataset_name, file_number = match.groups()
    
    # 验证数据集名是否匹配
    if file_dataset_name.upper() != args.dataset.upper():
        # 数据集名不匹配
        error_response = [
            {"error data": 0, "": 0},
            {"error data": 0, "": 0}
        ]
        print(json.dumps(error_response, ensure_ascii=False))
        return
    
    try:
        # 构建文件路径
        dataset_name = args.dataset.upper()
        # 添加sample_前缀到文件路径
        sample_path = f"/date/sunchengrui/huaweibei/llm/test/output/{dataset_name}_{file_number}.pt"
        badnets_model_path = f"/date/sunchengrui/huaweibei/llm/test/{dataset_name}_badnets.pth"
        safe_model_path = f"/date/sunchengrui/huaweibei/llm/test/{dataset_name}_safe.pth"
        
        # 检查文件是否存在
        if not os.path.exists(sample_path):
            print(f"错误: 样本文件不存在: {sample_path}")
            error_response = [
                {"error data": 0, "": 0},
                {"error data": 0, "": 0}
            ]
            print(json.dumps(error_response, ensure_ascii=False))
            return
        
        if not os.path.exists(badnets_model_path):
            print(f"错误: BadNets模型文件不存在: {badnets_model_path}")
            error_response = [
                {"error data": 0, "": 0},
                {"error data": 0, "": 0}
            ]
            print(json.dumps(error_response, ensure_ascii=False))
            return
        
        if not os.path.exists(safe_model_path):
            print(f"错误: Safe模型文件不存在: {safe_model_path}")
            error_response = [
                {"error data": 0, "": 0},
                {"error data": 0, "": 0}
            ]
            print(json.dumps(error_response, ensure_ascii=False))
            return
        
        # 加载样本
        image_tensor = torch.load(sample_path, weights_only=False)
        
        # >>>>>>>>>>>>>> 在这里添加了GTSRB的resize和normalize逻辑 <<<<<<<<<<<<<<
        if dataset_name == 'GTSRB':
            # 1. Resize ((32, 32)) - 使用 F.interpolate
            # 增加一个batch维度，使形状变为 (1, C, H, W)，方便插值
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0), 
                size=(32, 32), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # 移除batch维度，恢复 (C, 32, 32)
            
            # 2. ToTensor() - 假设加载的 Tensor 已经是 [0.0, 1.0] 范围的浮点数，因此跳过 ToTensor
            # 如果加载的张量是 [0, 255] 的整数，请取消下面这行的注释：
            # image_tensor = image_tensor.float() / 255.0 

            # 3. Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
            mean = torch.tensor([0.3403, 0.3121, 0.3214]).view(3, 1, 1)
            std = torch.tensor([0.2724, 0.2608, 0.2669]).view(3, 1, 1)
            
            # 执行归一化操作：x = (x - mean) / std
            image_tensor = (image_tensor - mean) / std
            
            # print(f"DEBUG: GTSRB样本已完成 Resize 和 Normalize。最终形状: {image_tensor.shape}")
        # >>>>>>>>>>>>>>>>>>>>>> 结束修改 <<<<<<<<<<<<<<<<<<<<<<<<
        
        # 加载模型
        badnets_model = load_model(badnets_model_path, dataset_name)
        safe_model = load_model(safe_model_path, dataset_name)
        
        # 进行预测
        badnets_results = predict(badnets_model, image_tensor, dataset_name)
        safe_results = predict(safe_model, image_tensor, dataset_name)
        
        # 构建响应
        response = [
            {"model_type": "badnets", "predictions": badnets_results},
            {"model_type": "safe", "predictions": safe_results}
        ]
        
        print(json.dumps(response, ensure_ascii=False))
        
    except Exception as e:
        # 发生异常时返回错误响应
        print(f"发生异常: {str(e)}")
        error_response = [
            {"error data": 0, "": 0},
            {"error data": 0, "": 0}
        ]
        print(json.dumps(error_response, ensure_ascii=False))

if __name__ == "__main__":
    main()