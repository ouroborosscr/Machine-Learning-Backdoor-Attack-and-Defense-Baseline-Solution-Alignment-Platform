#修复（选）
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task
import read_json_file as rjf
import numpy as np
import debackdoor

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()

def normalize_array(arr, method='minmax'):
    """
    对数组进行归一化处理
    
    参数:
        arr: 输入数组
        method: 归一化方法，可选值：
            'minmax': 最小-最大归一化，缩放到[0, 1]范围
            'standard': 标准化，转换为均值为0，标准差为1的分布
            'l1': L1归一化，使得向量的曼哈顿距离为1
            'l2': L2归一化，使得向量的欧几里得距离为1
    
    返回:
        numpy.ndarray: 归一化后的数组
    """
    if method == 'minmax':
        # 最小-最大归一化
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return arr  # 避免除以0
        return (arr - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        # 标准化
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        if std_val == 0:
            return arr - mean_val  # 避免除以0，只减去均值
        return (arr - mean_val) / std_val
    
    elif method == 'l1':
        # L1归一化
        l1_norm = np.sum(np.abs(arr))
        if l1_norm == 0:
            return arr  # 避免除以0
        return arr / l1_norm
    
    elif method == 'l2':
        # L2归一化
        l2_norm = np.sqrt(np.sum(arr**2))
        if l2_norm == 0:
            return arr  # 避免除以0
        return arr / l2_norm
    
    else:
        raise ValueError(f"不支持的归一化方法: {method}")

class pattern3Synthesizer(Synthesizer):
    # pattern_tensor: torch.Tensor = torch.tensor([
    #     [1., 0., 1.],
    #     [-10., 1., -10.],
    #     [-10., -10., 0.],
    #     [-10., 1., -10.],
    #     [1., 0., 1.]
    # ])


    # # abs
    # json_file_path = r"..\models\abs\PUBFIG_SQ\label_6\meta_1_label6.json"
    # json_data = rjf.read_json_file(json_file_path)
    # arr2 = np.array(json_data["mask_np"])
    # print(arr2.shape)
    # print(np.max(arr2))

    # #Aeva
    # arr2 = np.zeros((224, 224, 3))
    # n = 0
    # for i in range(83):
    #     #if i!=1:
    #         # a = np.load(f"Aeva\MNIST_BadNets\data_{i}_8.npy")
    #     if i!=6:
    #         a = np.load(f"..\models\Aeva\PUBFIG_SQ_1\data_{i}_6.npy")
    #         # a = np.load(f"Aeva\MNIST_multi2\data_{i}_1.npy")
    #         if a.any():
    #             arr2 = a[0]
    #             n = n + 1
    # arr2 = arr2 / n
    # arr2 = arr2.reshape(3, 224, 224)
    # arr2 = normalize_array(arr2, method='minmax')
    # print(np.max(arr2))

    # #B3D
    # a = torch.load("..\models\B3D\PUBFIG_SQ\\trigger_target_6\\trigger_target_6.pt")
    # arr2 = np.zeros(a["meta"]["image_size"])
    # x, y = a["meta"]["position"]
    # delta_np = a["delta"].cpu().numpy()
    # _, delta_h, delta_w = delta_np.shape
    # arr2[0, x:x+delta_h, y:y+delta_w] = delta_np[0]
    # arr2[1, x:x+delta_h, y:y+delta_w] = delta_np[1]
    # arr2[2, x:x+delta_h, y:y+delta_w] = delta_np[2]
    # arr2 = normalize_array(arr2, method='minmax')
    # # print(arr2)
    # print(np.max(arr2))

    # #debackdoor
    arr2 = np.zeros((3, 224, 224))

    # # #NC
    # a = torch.load("../models/NC/PUBFIG_HCB/mask_0.pt")
    # arr2 = torch.zeros((3, 224, 224))
    # arr2[0, :, :] = a
    # arr2[1, :, :] = a
    # arr2[2, :, :] = a
    





    pattern_tensor: torch.Tensor = torch.tensor(arr2).squeeze()

    "Just some random 2D pattern."

    x_top = 0
    #x_top = 21
    "X coordinate to put the backdoor into."
    y_top = 0
    #y_top = 3
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: Task):
        super().__init__(task)
        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)

    def make_pattern(self, pattern_tensor, x_top, y_top):
        full_image = torch.zeros(self.params.input_shape)
        full_image.fill_(self.mask_value)

        x_bot = x_top + pattern_tensor.shape[0]
        y_bot = y_top + pattern_tensor.shape[1]

        # if x_bot >= self.params.input_shape[1] or \
        #         y_bot >= self.params.input_shape[2]:
        #     raise ValueError(f'Position of backdoor outside image limits:'
        #                      f'image: {self.params.input_shape}, but backdoor'
        #                      f'ends at ({x_bot}, {y_bot})')

        full_image = pattern_tensor

        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        # self.pattern = self.task.normalize(full_image).to(self.params.device)
        self.pattern = full_image.to(self.params.device)
        # print(self.mask)
        # print(pattern_tensor.shape[1])
        # die()

    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()
        # batch.inputs[:attack_portion] = (1 - mask) * \
        #                                 batch.inputs[:attack_portion] + \
        #                                 mask * pattern
        batch.inputs[:attack_portion] = batch.inputs[:attack_portion] + pattern
        batch.inputs[:attack_portion] = torch.clamp(batch.inputs[:attack_portion], 0, 1)

        return

    def synthesize_labels(self, batch, attack_portion=None):
        #batch.labels[:attack_portion].fill_(self.params.backdoor_label)

        return

    def get_pattern(self):
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(
                functional.resize(image,
                    resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] \
                               - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] \
                               - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask
