import os
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from constants import *  # 导入项目中的常量，如dimensions、resolution等

def reconstruct_trigger_from_file(trigger_file_path, save_output=True, output_dir=None, dataset="gtsrb"):
    """
    从保存的torch文件中重建GTSRB数据集的trigger图像
    
    参数：
    trigger_file_path: 保存的trigger文件路径
    save_output: 是否保存重建的图像
    output_dir: 输出图像保存目录，如果为None则与输入文件在同一目录
    dataset: 数据集名称，默认为"gtsrb"
    
    返回：
    trigger_image: 重建的trigger图像tensor
    """
    # 加载保存的trigger数据
    trigger_data = torch.load(trigger_file_path)
    
    # 提取trigger信息
    pattern = trigger_data["pattern"]
    top_left = trigger_data["top_left"]
    bottom_right = trigger_data["bottom_right"]
    attack_type = trigger_data["attack_type"]
    
    print(f"Loaded trigger from {trigger_file_path}")
    print(f"Attack type: {attack_type}")
    print(f"Trigger position: top_left={top_left}, bottom_right={bottom_right}")
    
    # 根据数据集类型确定图像尺寸
    channels = dimensions[dataset]
    height, width = resolution[dataset], resolution[dataset]
    
    print(f"Using {dataset} dimensions: {channels} channels, {height}x{width} resolution")
    
    # 创建空白图像用于重建trigger (GTSRB为3通道彩色图像)
    trigger_image = torch.zeros((channels, height, width))
    
    # 根据攻击类型重建trigger
    if attack_type == "wanet":
        # WANet的trigger是变形场，需要特殊处理
        print("WANet trigger is a deformation field, visualizing control grid...")
        # 可视化控制网格
        control_grid = pattern.squeeze(0)  # 移除batch维度
        # 对于WANet，我们可以可视化其变形场的大小分布
        magnitude = torch.sqrt(control_grid[:, :, :, 0]**2 + control_grid[:, :, :, 1]**2)
        # GTSRB为3通道，将magnitude复制到所有通道
        for c in range(channels):
            trigger_image[c] = magnitude
            
    elif attack_type == "filter":
        # Filter的trigger是gamma值，显示为条形图
        print("Filter trigger is a gamma correction parameter, visualizing as bar chart...")
        # 为每个通道创建可视化表示
        for c in range(min(channels, len(pattern))):
            gamma_value = float(pattern[0, c, 0, 0])
            # 在图像上绘制一个表示gamma值的条形
            bar_height = int(height * gamma_value / 4.0)  # 假设gamma范围是0.5-4.0
            bar_height = min(bar_height, height)
            trigger_image[c, height-bar_height:, :width//channels*c : width//channels*(c+1)] = 1.0
    
    elif attack_type == "patch":
        # Patch类型的trigger需要特殊处理
        print("Reconstructing patch trigger...")
        # 确保pattern是numpy数组格式
        if isinstance(pattern, torch.Tensor):
            pattern_np = pattern.cpu().numpy()
        else:
            pattern_np = pattern
        
        # 转换patch pattern为图像格式
        if len(pattern_np.shape) == 3:  # 如果已经是[C, H, W]格式
            pattern_image = torch.from_numpy(pattern_np)
        else:
            # 修改部分：处理浮点数的pattern
            try:
                # 先尝试直接转换为浮点数组
                pattern_image = torch.from_numpy(pattern_np.astype(float))
                # 确保维度正确，如果是[C, H]，需要扩展维度
                if len(pattern_image.shape) == 2:
                    pattern_image = pattern_image.unsqueeze(2)
            except Exception:
                # 备用方案：如果pattern是整数表示的二进制模式
                temp_pattern = []
                for channel in pattern_np:
                    channel_pixels = []
                    for row_id in channel:
                        try:
                            # 尝试转换为整数后再使用二进制格式
                            int_row = int(row_id)
                            format_str = "{0:032b}" if resolution[dataset] == 32 else "{0:028b}"
                            row_pixels = np.array([float(bit) for bit in format_str.format(int_row)])
                        except (ValueError, TypeError):
                            # 如果转换失败，直接使用该值并扩展为适当长度的数组
                            pixel_length = resolution[dataset]
                            row_pixels = np.full(pixel_length, float(row_id))
                        channel_pixels.append(row_pixels)
                    temp_pattern.append(np.stack(channel_pixels))
                pattern_image = torch.from_numpy(np.stack(temp_pattern))
        
        # 将pattern应用到指定位置
        r1, c1 = top_left
        r2, c2 = bottom_right
        # 确保pattern_image的尺寸与目标区域匹配
        target_height = r2 - r1 + 1
        target_width = c2 - c1 + 1
        
        # 调整pattern_image的尺寸以匹配目标区域
        if pattern_image.shape[1:] != (target_height, target_width):
            # 截取适当尺寸
            pattern_image = pattern_image[:, :target_height, :target_width]
            # 如果pattern的通道数不足，复制现有通道
            while pattern_image.shape[0] < channels:
                pattern_image = torch.cat([pattern_image, pattern_image[:1]], dim=0)
        
        # 对于GTSRB的3通道图像，确保正确应用pattern
        for c in range(min(channels, pattern_image.shape[0])):
            trigger_image[c, r1:r2+1, c1:c2+1] = pattern_image[c, :target_height, :target_width]
    
    else:  # blend, lira, semantic等类型
        print(f"Reconstructing {attack_type} trigger...")
        # 确保pattern是torch tensor格式
        if isinstance(pattern, np.ndarray):
            pattern_tensor = torch.from_numpy(pattern)
        else:
            pattern_tensor = pattern
        
        # 如果pattern的通道数不足，复制现有通道
        while pattern_tensor.shape[0] < channels:
            pattern_tensor = torch.cat([pattern_tensor, pattern_tensor[:1]], dim=0)
        
        # 将pattern应用到指定位置
        r1, c1 = top_left
        r2, c2 = bottom_right
        for c in range(min(channels, pattern_tensor.shape[0])):
            trigger_image[c, r1:r2+1, c1:c2+1] = pattern_tensor[c, r1:r2+1, c1:c2+1]
    
    # 确保trigger图像在0-1范围内
    trigger_image = torch.clamp(trigger_image, 0, 1)
    
    # 保存重建的图像
    if save_output:
        if output_dir is None:
            output_dir = os.path.dirname(trigger_file_path)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名，添加gtsrb标识
        base_name = os.path.basename(trigger_file_path).replace('.pt', '_gtsrb_reconstructed.png')
        output_path = os.path.join(output_dir, base_name)
        
        # 保存图像
        save_image(trigger_image.unsqueeze(0), output_path)
        print(f"Reconstructed GTSRB trigger image saved to {output_path}")
        
        # 使用matplotlib显示图像
        plt.figure(figsize=(6, 6))
        if channels == 1:  # 灰度图
            plt.imshow(trigger_image[0].cpu().numpy(), cmap='gray')
        else:  # 彩色图（GTSRB是3通道）
            # 转换为(H, W, C)格式并调整通道顺序
            plt.imshow(trigger_image.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Reconstructed GTSRB {attack_type} Trigger")
        plt.axis('off')
        
        # 保存matplotlib图像
        plt.savefig(os.path.join(output_dir, base_name.replace('.png', '_plt.png')), bbox_inches='tight')
        plt.close()
    
    return trigger_image

def batch_reconstruct_triggers(trigger_dir, output_dir=None, dataset="gtsrb"):
    """
    批量重建目录中的所有trigger文件
    
    参数：
    trigger_dir: 包含trigger文件的目录
    output_dir: 输出图像保存目录，如果为None则使用trigger_dir
    dataset: 数据集名称，默认为"gtsrb"
    """
    if output_dir is None:
        output_dir = trigger_dir
    
    # 获取目录中的所有.pt文件
    trigger_files = [f for f in os.listdir(trigger_dir) if f.endswith('.pt')]
    
    print(f"Found {len(trigger_files)} trigger files in {trigger_dir}")
    
    # 逐个重建trigger
    for i, trigger_file in enumerate(trigger_files):
        print(f"Processing {i+1}/{len(trigger_files)}: {trigger_file}")
        trigger_file_path = os.path.join(trigger_dir, trigger_file)
        try:
            reconstruct_trigger_from_file(trigger_file_path, save_output=True, 
                                        output_dir=output_dir, dataset=dataset)
        except Exception as e:
            print(f"Error processing {trigger_file}: {str(e)}")
    
    print("Batch reconstruction completed!")

# 使用示例
if __name__ == "__main__":
    # GTSRB数据集的使用示例
    trigger_file = "debackdoor\GTSRB_badnets_patch\triggers\trigger_label_0.pt"  # 示例路径
    
    # 单个文件重建
    if os.path.exists(trigger_file):
        trigger_image = reconstruct_trigger_from_file(trigger_file, dataset="gtsrb")
        print("Single trigger reconstruction completed!")
    else:
        print(f"Example trigger file not found: {trigger_file}")
        print("You can modify the path to an existing GTSRB trigger file.")
    
    # 批量重建示例
    # trigger_dir = "path_to_gtsrb_trigger_files"
    # if os.path.exists(trigger_dir):
    #     batch_reconstruct_triggers(trigger_dir, dataset="gtsrb")
    # else:
    #     print(f"Example trigger directory not found: {trigger_dir}")