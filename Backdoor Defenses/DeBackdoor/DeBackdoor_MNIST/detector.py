# 在导入语句后面添加新函数
import os
import copy
import json
import time
import random
import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from torchvision.utils import save_image  # 添加用于保存图像的函数

from utils import *
from constants import *
from model import Model

# 添加一个函数，通过模型预测获取图像标签
def get_predicted_labels(model, images, device):
    """通过模型预测获取图像标签"""
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

# 添加一个函数，将trigger转换为可保存的图像格式
def save_trigger_as_image_and_torch(detector, result, save_path, target_label, dataset):
    """将trigger保存为图片和torch格式"""
    # 创建保存trigger的目录
    trigger_dir = os.path.join(save_path, f"triggers")
    os.makedirs(trigger_dir, exist_ok=True)
    
    # 获取trigger信息
    pattern = result["pattern"]
    top_left = result["top_left"]
    bottom_right = result["bottom_right"]
    
    # 将pattern从列表转换回numpy数组
    if isinstance(pattern, list):
        # 对于多通道数据，需要处理每个通道
        if isinstance(pattern[0], list):
            pattern = np.array(pattern)
        else:
            pattern = np.array([pattern])
    
    # 生成一个空白图像用于展示trigger
    if dataset == "mnist":
        # MNIST是单通道灰度图
        trigger_image = np.zeros((1, resolution[dataset], resolution[dataset]))
    else:
        # 其他数据集可能是RGB三通道
        trigger_image = np.zeros((dimensions[dataset], resolution[dataset], resolution[dataset]))
    
    # 根据不同的攻击类型处理trigger
    if detector.attack_type == "wanet":
        # WANet的trigger是变形场，需要特殊处理
        trigger_tensor = torch.tensor(pattern).unsqueeze(0).float()
        # 保存为torch格式
        torch.save({
            "pattern": trigger_tensor,
            "top_left": top_left,
            "bottom_right": bottom_right,
            "attack_type": detector.attack_type
        }, os.path.join(trigger_dir, f"trigger_wanet_label_{target_label}.pt"))
        print(f"Saved WANet trigger to {trigger_dir}/trigger_wanet_label_{target_label}.pt")
        
    elif detector.attack_type == "filter":
        # Filter的trigger是gamma值，需要特殊处理
        trigger_tensor = torch.tensor(pattern).float()
        # 保存为torch格式
        torch.save({
            "pattern": trigger_tensor,
            "top_left": top_left,
            "bottom_right": bottom_right,
            "attack_type": detector.attack_type
        }, os.path.join(trigger_dir, f"trigger_filter_label_{target_label}.pt"))
        print(f"Saved Filter trigger to {trigger_dir}/trigger_filter_label_{target_label}.pt")
        
    else:
        # 对于patch、blend等类型的攻击，可以直接生成图像
        # 生成pattern图像
        if detector.attack_type == "patch":
            # 特殊处理patch类型的pattern
            temp_pattern = []
            for channel in pattern:
                channel_pixels = []
                for row_id in channel:
                    # 将整数表示的行转换为像素数组
                    format_str = "{0:032b}" if resolution[dataset] == 32 else "{0:028b}"
                    row_pixels = np.fromstring(" ".join(format_str.format(row_id)), sep=" ")
                    channel_pixels.append(row_pixels)
                temp_pattern.append(np.stack(channel_pixels))
            pattern_image = np.stack(temp_pattern)
        else:
            # 其他类型直接使用pattern
            pattern_image = pattern
        
        # 确保pattern_image在正确的范围内
        pattern_image = np.clip(pattern_image, 0, 1)
        
        # 创建trigger图像
        r1, c1 = top_left
        r2, c2 = bottom_right
        if dataset == "mnist":
            # MNIST是单通道
            trigger_image[0, r1:r2+1, c1:c2+1] = pattern_image[0, r1:r2+1, c1:c2+1]
        else:
            # 多通道图像
            for i in range(min(dimensions[dataset], pattern_image.shape[0])):
                trigger_image[i, r1:r2+1, c1:c2+1] = pattern_image[i, r1:r2+1, c1:c2+1]
        
        # 转换为torch tensor并保存为图像
        trigger_tensor = torch.from_numpy(trigger_image).float()
        # 为了显示，需要添加batch维度
        trigger_tensor = trigger_tensor.unsqueeze(0)
        
        # 保存为图像
        image_path = os.path.join(trigger_dir, f"trigger_label_{target_label}.png")
        save_image(trigger_tensor, image_path)
        print(f"Saved trigger image to {image_path}")
        
        # 保存为torch格式
        torch.save({
            "pattern": torch.from_numpy(pattern).float(),
            "top_left": top_left,
            "bottom_right": bottom_right,
            "attack_type": detector.attack_type
        }, os.path.join(trigger_dir, f"trigger_label_{target_label}.pt"))
        print(f"Saved trigger to {trigger_dir}/trigger_label_{target_label}.pt")

# LeNet Model definition
class Net(Model):#创建网络
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

class Detector:
    def __init__(self, model, images, attack_type, size_limit, k=4, image_labels=None):
        self.model   = model
        self.grid_rescale = 1
        self.k = k
        # 存储图像原始类别
        self.image_labels = image_labels
        
        # Initialize trigger
        self.attack_type  = attack_type
        if self.attack_type in ["blend", "lira", "semantic"]:
            # Blended triggger spans entire image
            self.pattern      = np.zeros((dimensions[dataset], resolution[dataset], resolution[dataset]))
            self.top_left     = np.array([0,0])
            self.bottom_right = np.array([resolution[dataset]-1, resolution[dataset]-1])
        elif self.attack_type == "patch":
            # Patch trigger covers continuous subspace within image
            self.pattern      = np.repeat(np.random.randint(2**resolution[dataset], size=(1,resolution[dataset])), dimensions[dataset], axis=0)
            self.top_left = np.random.randint(resolution[dataset], size=2)
            self.size_limit   = size_limit-1 if size_limit else size_limits[dataset]
            self.bottom_right = np.array([np.random.randint(self.top_left[0],
                                                            min(self.top_left[0]+2, resolution[dataset])), 
                                            np.random.randint(self.top_left[1],
                                                            min(self.top_left[1]+2, resolution[dataset]))])
        # Warped trigger uses the control-grid of warping field    
        elif self.attack_type == "wanet":
            self.pattern = torch.rand(1, 2, self.k, self.k) * 2 - 1
            self.top_left = np.array([0,0])
            self.bottom_right = np.array([self.k-1, self.k-1])
        # Filter trigger uses the parameters of the gamma correction operation
        elif self.attack_type == "filter":
            self.pattern = torch.tensor([1]*dimensions[dataset]).view(1, dimensions[dataset], 1, 1).to(device)
            self.top_left = np.array([0,0])
            self.bottom_right = np.array([resolution[dataset]-1, resolution[dataset]-1])
        else:
            raise ValueError(f"{self.attack_type} attack not supported")

        # Scoring items
        self.soft         = nn.Softmax(dim=1)
        self.clean_images = images
        self.lambd        = -1
        self.score        = -1
        self.target_label = -1

    # Backdoor trigger generation using simulated annealing
    def optimize(self, target_label, num_rounds, lambd):
        # Initialize progress bar
        show_progress_bar(0, num_rounds, prefix="Progress:", suffix="Complete", length=50)
        
        # Pole controls the temperature cooling schedule
        pole              = num_rounds/50
        self.target_label = int(target_label)
        self.lambd        = lambd
        max_score         = -1
        max_pattern       = None
        max_top_left      = None
        max_bottom_right  = None
        max_class_scores  = None  # 存储每个类别的最高ASR

        for round_num in range(num_rounds):
            # Temperature decreases as more rounds pass
            temperature = pole/(round_num+pole)-pole/(num_rounds+pole)

            if round_num%2 == 0 or self.attack_type in ["blend", "wanet", "lira", "filter", "semantic"]:
                self.optimize_pattern(temperature, target_label)
            else:
                # Patch attack optimizes mask on odd rounds
                self.optimize_mask(temperature, target_label)

            # Update highest scoring trigger and progress bar
            if self.score > max_score:
                max_score        = self.score
                max_pattern      = self.pattern
                max_top_left     = self.top_left
                max_bottom_right = self.bottom_right
                # 如果提供了图像标签，获取每个类别的ASR
                if self.image_labels is not None:
                    max_class_scores, _ = self.compute_score(
                        self.apply_backdoor(self.pattern, self.top_left, self.bottom_right), 
                        target_label
                    )

            show_progress_bar(round_num+1, num_rounds, prefix="Progress:", suffix=f"Complete (ASR={max_score:.4f})", length=50)

        result = {
            "score"       : float(max_score),
            "top_left"    : max_top_left.tolist(),
            "bottom_right": max_bottom_right.tolist(),
            "pattern"     : [channel.tolist() for channel in max_pattern]
        }
        
        # 如果有每个类别的ASR，添加到结果中
        if max_class_scores is not None:
            result["class_scores"] = max_class_scores.cpu().detach().numpy().tolist()
            
        return result

    # Image manipulation modules
    def generate_pattern_image(self, pattern):
        if self.attack_type == "patch":
            trigger = []
            
            for channel in pattern:
                channel_pixels = []

                for row_id in channel:
                    # Pixels in each row of image are represented using 1s/0s in 28/32-bit integer
                    format_str = "{0:032b}" if resolution[dataset] == 32 else "{0:028b}"
                    row_pixels = np.fromstring(" ".join(format_str.format(row_id)), sep=" ")
                    channel_pixels.append(row_pixels)
                trigger.append(np.stack(channel_pixels))
            trigger = torch.from_numpy(np.stack(trigger)).type(torch.FloatTensor).to(device)
        else:
            trigger = torch.from_numpy(self.pattern).to(device)
        return trigger

    def flip_neighbor_pixels(self, coords):
        neighbor_pattern = copy.deepcopy(self.pattern)

        if self.attack_type in ["blend", "lira", "semantic"]:
            # Modify pixels across the entire image
            bound = 1 if self.attack_type in ["blend", "semantic"] else 0.1
            delta = np.random.uniform(-bound, bound, size=(dimensions[dataset], resolution[dataset], resolution[dataset]))
            neighbor_pattern += delta
            neighbor_pattern = np.clip(neighbor_pattern, 0 if self.attack_type in ["blend", "semantic"] else -0.1, 1 if self.attack_type in ["blend", "semantic"] else 0.1)
        elif self.attack_type == "filter":
            channel, _, _ = coords[0]
            neighbor_pattern[0][channel][0][0] = np.random.choice(np.arange(0.5,4,0.5))
        else:
            if self.attack_type == "patch":
                _,row,col = coords[0]
                
                for channel in range(dimensions[dataset]):
                    coords.append((channel,row,col))
            
            for channel,row,col in coords:
                if self.attack_type == "wanet":
                    # Modify entry in the control grid
                    bound = 0.3
                    delta = (-bound - bound) * torch.rand(1) + bound
                    neighbor_pattern[0][channel][row][col] += delta[0]
                else:
                    # Flip single pixel at given coordinates
                    neighbor_pattern[channel][row] = self.pattern[channel][row]^2**(resolution[dataset]-1-col)

        return neighbor_pattern

    # Continuous attack success rate (cASR) modules
    def apply_backdoor(self, pattern, top_left, bottom_right):
        if self.attack_type == "wanet":
            # Trigger warps across the entire image
            noise_grid = (
                F.upsample(pattern, size=resolution[dataset], mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(device)
            )
            array1d = torch.linspace(-1, 1, steps=resolution[dataset])
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to(device)

            # Generate trigger
            grid_temps = (identity_grid + noise_grid / resolution[dataset]) * self.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            
            # Insert trigger
            backdoor_images = F.grid_sample(self.clean_images, grid_temps.repeat(self.clean_images.shape[0], 1, 1, 1), align_corners=True)
        elif self.attack_type == "filter":
            backdoor_images = self.clean_images ** self.pattern
        else:
            # Trigger overwrites image content for patch attacks, but is blended into the image for other attacks
            alpha         = alphas[dataset] if self.attack_type in ["blend", "lira", "filter", "semantic"] else 1
            backdoor_images = copy.deepcopy(self.clean_images)

            # Generate trigger
            pattern_image = self.generate_pattern_image(pattern)
            r1,c1         = top_left
            r2,c2         = bottom_right
            
            # Insert trigger
            if self.attack_type == "lira":
                backdoor_images[:,:,r1:r2+1,c1:c2+1] = backdoor_images[:,:,r1:r2+1,c1:c2+1] + pattern_image[:,r1:r2+1,c1:c2+1]
            else:
                if dataset == "imagenet":
                    # Upscale trigger for 224x224 ImageNet images
                    resize = Resize((224, 224))
                    pattern_image = pattern_image.unsqueeze(0)
                    pattern_image = resize(pattern_image)
                    pattern_image = pattern_image.squeeze(0)
                    r1,c1 = 0,0
                    r2,c2 = 223,223
                
                backdoor_images[:,:,r1:r2+1,c1:c2+1] = (1-alpha)*backdoor_images[:,:,r1:r2+1,c1:c2+1] + alpha*pattern_image[:,r1:r2+1,c1:c2+1]
        return backdoor_images

    def compute_score(self, backdoor_images, target_label):
        # Get model output and isolate output for target label
        scores                 = self.soft(self.model(backdoor_images))
        target_label_scores    = scores[:,target_label].detach().clone()

        # Compute gap between target label output and output of highest scoring label amongst other labels
        scores[:,target_label] = 0.00001
        example_max_scores     = scores.max(axis=1)[0]
        deltas                 = target_label_scores.log()-example_max_scores.log()

        proxy_1 =   ( self.lambd*deltas).exp()/2
        proxy_2 = 1-(-self.lambd*deltas).exp()/2
        proxy   = proxy_1*(deltas<0)+proxy_2*(deltas>=0)

        # 如果提供了图像标签，则按原始类别分组计算 ASR
        if self.image_labels is not None:
            num_classes = len(torch.unique(self.image_labels))
            class_scores = torch.zeros(num_classes, device=proxy.device)
            
            for cls in range(num_classes):
                # 找出属于当前类别的样本索引
                cls_mask = (self.image_labels == cls)
                if cls_mask.any():
                    # 计算该类样本的平均ASR
                    class_scores[cls] = proxy[cls_mask].mean()
                else:
                    # 如果没有该类样本，设为0或NaN
                    class_scores[cls] = float('nan')
            
            # 返回每个类别到目标标签的ASR以及总体平均ASR
            return class_scores, proxy.mean()
        else:
            # 保持原有行为，返回总体平均ASR
            return proxy.mean()

    # Search parameter optimization modules 
    def optimize_pattern(self, temperature, target_label):
        # Flip random color channel/location and compute new score
        channel           = np.random.randint(2 if self.attack_type == "wanet" else dimensions[dataset])
        row               = np.random.randint(self.top_left[0], self.bottom_right[0]+1)
        col               = np.random.randint(self.top_left[1], self.bottom_right[1]+1)
        neighbor_pattern  = self.flip_neighbor_pixels([(channel,row,col)])
        neighbor_images   = self.apply_backdoor(neighbor_pattern, self.top_left, self.bottom_right)
        
        # 处理新的返回格式
        if self.image_labels is not None:
            _, neighbor_score = self.compute_score(neighbor_images, target_label)
        else:
            neighbor_score = self.compute_score(neighbor_images, target_label)
        
        # Update if new score is higher, or randomly with probability dictated by temperature
        if neighbor_score > self.score or np.random.random() < temperature:
            self.pattern = neighbor_pattern
            self.score   = neighbor_score

    def optimize_mask(self, temperature, target_label):
        # Calculate leeway for moving the patch
        space_above = self.top_left[0]
        space_right = resolution[dataset]-1-self.bottom_right[1]
        space_below = resolution[dataset]-1-self.bottom_right[0]
        space_left  = self.top_left[1]
        
        # Generate possible reshapes (extend/contract by 1 row up/down or 1 column left/right)
        deltas = []
        if self.top_left[0] > 0 and self.bottom_right[0]-self.top_left[0]<self.size_limit:
            deltas.extend([((-1, 0), ( 0, 0))])
        if self.top_left[0] < resolution[dataset]-1 and self.top_left[0] < self.bottom_right[0]:
            deltas.extend([(( 1, 0), ( 0, 0))])
        if self.top_left[1] > 0 and self.bottom_right[1]-self.top_left[1]<self.size_limit:
            deltas.extend([(( 0,-1), ( 0, 0))])
        if self.top_left[1] < resolution[dataset]-1 and self.top_left[1] < self.bottom_right[1]:
            deltas.extend([(( 0, 1), ( 0, 0))])
        if self.bottom_right[0] > 0 and self.bottom_right[0] > self.top_left[0]:
            deltas.extend([(( 0, 0), (-1, 0,))])
        if self.bottom_right[0] < resolution[dataset]-1 and self.bottom_right[0]-self.top_left[0]<self.size_limit:
            deltas.extend([(( 0, 0), ( 1, 0,))])
        if self.bottom_right[1] > 0 and self.bottom_right[1] > self.top_left[1]:
            deltas.extend([(( 0, 0), ( 0,-1))])
        if self.bottom_right[1] < resolution[dataset]-1 and self.bottom_right[1]-self.top_left[1]<self.size_limit:
            deltas.extend([(( 0, 0), ( 0, 1))])
            
        # Generate possible moves to new locations for the patch
        relocations = []
        for v_delta in np.concatenate((-np.arange(1,space_above+1),np.arange(1,space_below+1))):
            for h_delta in np.concatenate((-np.arange(1,space_left+1),np.arange(1,space_right+1))):
                relocations.append((v_delta,h_delta))
        deltas.extend([(delta,delta) for delta in relocations])

        # Apply randomly selected reshape/relocation and compute new score
        delta_top_left, delta_bottom_right = random.sample(deltas, 1)[0]
        neighbor_top_left     = self.top_left    +delta_top_left
        neighbor_bottom_right = self.bottom_right+delta_bottom_right    
        neighbor_mask         = (neighbor_top_left, neighbor_bottom_right) 
        neighbor_images       = self.apply_backdoor(self.pattern, neighbor_top_left, neighbor_bottom_right)
        
        # 处理新的返回格式
        if self.image_labels is not None:
            _, neighbor_score = self.compute_score(neighbor_images, target_label)
        else:
            neighbor_score = self.compute_score(neighbor_images, target_label)

        # Update if new score is higher, or randomly with probability dictated by temperature        
        if neighbor_score > self.score or np.random.random() < temperature:
            self.top_left, self.bottom_right = neighbor_mask
            self.score                       = neighbor_score

# 在main函数中添加保存trigger的代码
if __name__ == "__main__":
    # Parse command-line arguments (e.g. python detector.py -m examples/patch/mnist/model.pt -i examples/images/mnist.pt -n 30000 -l 0.9 -a patch -s 6)
    parser = argparse.ArgumentParser(description ="Detect backdoor triggers in deep models")
    parser.add_argument("-m", "--model_path"  , metavar="model_path"  , type=str  , required=True , help="path to model to be inspected")
    parser.add_argument("-i", "--image_path"  , metavar="image_path"  , type=str  , required=True , help="path to batch of clean validation images")
    parser.add_argument("-n", "--num_rounds"  , metavar="num_rounds"  , type=int  , required=True , help="number of optimization rounds")
    parser.add_argument("-l", "--lambd"       , metavar="lambda"      , type=float, required=True , help="score smoothing parameter lambda")
    parser.add_argument("-a", "--attack_type" , metavar="attack_type" , type=str  , required=True , help="attack type (patch/blend/wanet/lira/filter/semantic)", choices=["patch", "blend", "wanet", "lira", "filter", "semantic"])
    parser.add_argument("-s", "--size_limit"  , metavar="size-limit"  , type=int  , required=False, help="width/height limit of patch")
    parser.add_argument("-g", "--gpu_id"      , metavar="gpu_id"      , type=int  , required=False, help="id of gpu to be used for training")
    parser.add_argument("--with_labels", action="store_true", help="Include image labels for computing class-wise ASR matrix")
    args = parser.parse_args()

    # Load inspection pre-requisites
    device = torch.device(f"cpu")
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
    model = Net(10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False)['state_dict'])

    # 进入测试模式
    model.eval()
    images  = torch.load(args.image_path, weights_only=False).to(device)
    dataset = args.image_path.split("/")[-1].split(".")[0].split("-")[0].split("_")[0]
    results_path, _ = os.path.split(args.model_path)
    
    # 准备存储二维ASR矩阵的结果
    if args.with_labels:
        # 假设标签存储在images对象的某个属性中，或者需要单独加载
        # 这里需要根据实际情况调整如何获取图像标签
        # 以下代码仅作为示例
        try:
            # 尝试从图像数据中获取标签
            image_labels = torch.load(args.image_path.replace('images', 'labels'), weights_only=False).to(device)
        except:
            print("Warning: Could not load image labels. Attempting to get predicted labels from model...")
            # 调用新函数通过模型预测获取标签
            image_labels = get_predicted_labels(model, images, device)
            print(f"Successfully obtained predicted labels for {len(image_labels)} images.")
            print(image_labels)
    else:
        image_labels = None
    
    # 如果要创建完整的二维ASR矩阵，需要为每个目标标签运行检测器
    asr_matrix = None
    if args.with_labels:
        asr_matrix = np.zeros((num_labels[dataset], num_labels[dataset]))
    
    label_results = {}

    for label in range(num_labels[dataset]):
        print(f"Optimizing {args.attack_type} attack for target label {label}...")
        st = time.time()
        # 将图像标签传递给Detector
        detector = Detector(model=model, images=images, attack_type=args.attack_type, size_limit=args.size_limit, image_labels=image_labels)
        result = detector.optimize(target_label=label, num_rounds=args.num_rounds, lambd=args.lambd)
        
        score  = result["score"]
        runtime = time.time()-st
        result["runtime"] = runtime
        print(f"\nHighest cASR = {score:.4f}")
        print(f"Total runtime = {runtime:.2f}s\n")
        label_results[label] = result
        
        # 保存当前标签的trigger为图片和torch格式
        save_trigger_as_image_and_torch(detector, result, results_path, label, dataset)
        
        # 更新二维ASR矩阵
        if args.with_labels and "class_scores" in result:
            for source_cls, cls_score in enumerate(result["class_scores"]):
                asr_matrix[source_cls, label] = cls_score
    
    # Store results in same directory as model
    name = os.path.basename(args.model_path).replace("model", "result").split(".")[0]
    with open(os.path.join(results_path, f"{name}_{args.attack_type}.json"), "w") as f:
        json.dump(label_results, f)
        
    # 如果有二维ASR矩阵，单独保存
    if args.with_labels and asr_matrix is not None:
        with open(os.path.join(results_path, f"{name}_{args.attack_type}_asr_matrix.json"), "w") as f:
            json.dump({
                "asr_matrix": asr_matrix.tolist(),
                "description": "2D matrix of ASR where rows are source classes and columns are target labels"
            }, f)
        print(f"Successfully computed 2D ASR matrix and saved to {results_path}/{name}_{args.attack_type}_asr_matrix.json")
        print(asr_matrix)
    else:
        print("Warning: Could not compute 2D ASR matrix.")
