import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from meta_classifier import MetaClassifier

# 严格参考abs_pytorch_gpt.py中的Net类实现
class Model(nn.Module):
    def __init__(self, num_classes=10, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        # 严格按照abs_pytorch_gpt.py中的Net类结构定义
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

        if gpu:
            self.cuda()

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x

    def forward(self, x, latent=False):
        if self.gpu:
            x = x.cuda()
            
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

    # 为了兼容元分类器，添加emb_forward方法
    def emb_forward(self, x):
        return self.forward(x)

    # 添加loss方法以保持一致性
    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        return F.cross_entropy(pred, label)

# 设置参数
GPU = True  # 根据您的环境设置
model_path = './meta_classifier_ckpt/mnist.model_4'  # 第四个元分类器模型

# 改进的模型权重加载方法，处理权重文件的嵌套结构
# 根据错误信息，权重文件包含'state_dict', 'epoch', 'lr', 'params_dict'等键
# 模型参数实际存储在'state_dict'键中

def load_model_weights(model, weights_path):
    """
    从文件加载模型权重，处理不同的权重文件格式
    """
    try:
        # 首先尝试直接加载
        loaded_data = torch.load(weights_path, map_location="cpu")
        
        # 检查加载的数据类型
        if isinstance(loaded_data, dict):
            # 情况1: 如果数据包含'state_dict'键，使用其中的权重
            if 'state_dict' in loaded_data:
                model.load_state_dict(loaded_data['state_dict'])
            # 情况2: 如果直接是模型参数的字典，直接加载
            else:
                model.load_state_dict(loaded_data)
        else:
            raise TypeError("Loaded data is not a dictionary")
        
        return model
    except Exception as e:
        print(f"加载模型权重出错: {e}")
        raise

# 初始化基本模型和元分类器
basic_model = Model(num_classes=10, gpu=GPU)
input_size = (1, 28, 28)  # MNIST图像大小
class_num = 10  # MNIST类别数
meta_model = MetaClassifier(input_size, class_num, gpu=GPU)

# 加载预训练的元分类器权重
meta_model = load_model_weights(meta_model, model_path)

# 准备要检测的模型路径
# 注意：这里需要使用真实存在的模型文件路径
models_to_test = [
    ('../../../models/MNIST_multi.pth', 1),  # 良性模型示例
    ('../../../models/MNIST_safe.pth', 0),  # 恶意模型示例
    # 可以添加更多模型路径
]

# 创建一个新的epoch_meta_eval函数，正确处理权重加载
def custom_epoch_meta_eval(meta_model, basic_model, dataset, is_discrete, threshold=0.0):
    meta_model.eval()
    basic_model.train()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = list(range(len(dataset)))
    
    for i in perm:
        x, y = dataset[i]
        try:
            # 使用我们自定义的权重加载函数
            loaded_data = torch.load(x, map_location="cpu")
            if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
                basic_model.load_state_dict(loaded_data['state_dict'])
            else:
                basic_model.load_state_dict(loaded_data)

            # MNIST检测设置is_discrete=False
            out = basic_model.forward(meta_model.inp)
            score = meta_model.forward(out)

            l = meta_model.loss(score, y)
            cum_loss = cum_loss + l.item()
            preds.append(score.item())
            labs.append(y)
        except Exception as e:
            print(f"处理模型 {x} 时出错: {e}")
            # 跳过有问题的模型
            continue

    if len(preds) == 0:
        print("没有成功处理的模型")
        return 0.0, 0.0, 0.0

    preds = np.array(preds)
    labs = np.array(labs)
    
    try:
        auc = roc_auc_score(labs, preds)
    except:
        auc = 0.0
        print("计算AUC时出错")
        
    if threshold == 'half':
        threshold = np.median(preds).item()
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(preds), auc, acc

# 使用元分类器进行检测
def detect_single_model(meta_model, basic_model, model_path, label):
    """
    使用元分类器检测单个模型是否被植入木马
    """
    try:
        # 加载模型权重
        loaded_data = torch.load(model_path, map_location="cpu")
        if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
            basic_model.load_state_dict(loaded_data['state_dict'])
        else:
            basic_model.load_state_dict(loaded_data)
        
        # 执行前向传播
        out = basic_model.forward(meta_model.inp)
        score = meta_model.forward(out)
        
        # 打印结果
        print(f"模型路径: {model_path}")
        print(f"真实标签: {'恶意' if label == 1 else '良性'}")
        print(f"元分类器得分: {score.item():.4f}")
        print(f"预测结果: {'恶意' if score.item() > 0.5 else '良性'}")
        print("---")
        
        return score.item()
    except Exception as e:
        print(f"检测模型 {model_path} 时出错: {e}")
        return None

# 为了测试，我们创建一个简单的dataset列表
# 每个元素是(模型路径, 标签)，其中标签1表示恶意模型，0表示良性模型
test_dataset = models_to_test

# 批量检测
if len(test_dataset) > 1:
    loss, auc, acc = custom_epoch_meta_eval(meta_model, basic_model, test_dataset, is_discrete=False, threshold=0.5)
    print(f"批量检测结果:")
    print(f"损失: {loss:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"准确率: {acc:.4f}")

# 单个模型检测
print("\n单个模型检测:")
for model_path, label in models_to_test:
    detect_single_model(meta_model, basic_model, model_path, label)