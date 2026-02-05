import os
import json
import torch
import numpy as np
from abs_pytorch_gpt import optimize_trigger_for_neuron_local, save_trigger, get_class_balanced_seeds
import torchvision
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

# ... existing code ...

# 添加一个新函数来生成和保存trigger
def generate_and_save_triggers(opt, model_classifier):
    # 创建保存trigger的目录
    trigger_dir = f'./inspect_results/triggers_{opt.model}_{opt.inspect_layer_position}'
    os.makedirs(trigger_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集（根据模型类型选择合适的数据集）
    if 'mnist' in opt.ckpt.lower() or opt.model == 'net':
        num_classes = 10
        dataset = torchvision.datasets.MNIST("../../data/mnist", train=False, 
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Resize((28, 28))
                                           ]), download=True)
    else:
        num_classes = opt.n_cls
        # 对于GTSRB或其他数据集，使用相应的加载方式
        dataset = torchvision.datasets.GTSRB("../../data/gtsrb", split='test', 
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Resize((32, 32))
                                           ]), download=True)
    
    # 获取平衡的种子图像
    n_seed = 100  # 可以调整种子图像的数量
    seeds, _ = get_class_balanced_seeds(dataset, n_seed, num_classes)
    
    print(f"\n正在为每个类别生成并保存trigger...")
    
    # 对每个类别生成trigger
    for class_id in tqdm(range(num_classes)):
        try:
            # 获取模型层的名称（根据您的模型结构调整）
            layer_name = f'conv{opt.inspect_layer_position+1}' if 'net' in opt.model else f'layer{opt.inspect_layer_position}'
            
            # 假设我们为每个类别优化前3个最重要的神经元
            for channel_idx in range(3):
                # 调用trigger优化函数
                trigger_result = optimize_trigger_for_neuron_local(
                    opt.ckpt, 
                    type(model_classifier),  # 模型类
                    num_classes, 
                    str(device), 
                    seeds.numpy(), 
                    layer_name, 
                    channel_idx, 
                    objective_target=class_id,  # 目标类别
                    iters=500,  # 迭代次数
                    lr=0.1, 
                    tv_lambda=0.01, 
                    l1_lambda=0.01, 
                    mask_l1_lambda=0.01
                )
                
                # 保存trigger
                rank = channel_idx + 1
                save_trigger(
                    trigger_dir, 
                    class_id, 
                    rank, 
                    trigger_result["mask_np"], 
                    trigger_result["delta_np"], 
                    trigger_result
                )
        except Exception as e:
            print(f"生成类别 {class_id} 的trigger时出错: {e}")
            continue
    
    print(f"\n所有trigger已保存到 {trigger_dir}")
    return trigger_dir

# 修改inspect_saved_model函数，在其中调用新的trigger生成和保存函数
def inspect_saved_model(opt):
    # build partial model
    model_classifier = load_model(opt)
    model_classifier = model_classifier.eval()
    
    # 生成并保存trigger
    generate_and_save_triggers(opt, model_classifier)

    # compute and collect important neuron ids & dummy inner embeddings for each class
    dummy_inner_embeddings_all = compute_dummy_inner_embeddings(model_classifier, opt)

    # ... 以下是原有的代码 ...

# ... 其他现有代码保持不变 ...