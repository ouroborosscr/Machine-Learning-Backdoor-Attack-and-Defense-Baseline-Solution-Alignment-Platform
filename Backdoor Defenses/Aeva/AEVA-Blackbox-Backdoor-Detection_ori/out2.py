import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# 创建保存图像的目录
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载numpy数组
try:
    a = np.load(f"cifar10_adv_per/data_0_1.npy")
    print(f"数组形状: {a.shape}")
    print(f"数组数据类型: {a.dtype}")
    print(f"数组最小值: {a.min()}")
    print(f"数组最大值: {a.max()}")
    
    # 判断数据形状并处理
    if len(a.shape) == 4:  # 形状为 (num_images, height, width, channels)
        num_images = a.shape[0]
        print(f"发现 {num_images} 张图像")
        
        # 归一化图像数据到 [0, 255] 范围
        if a.max() <= 1.0 and a.min() >= 0.0:
            a = (a * 255).astype(np.uint8)
        
        # 显示并保存每张图像
        for i in range(min(num_images, 10)):  # 最多显示10张图像
            img = a[i]
            
            # 检查通道顺序，如果是CHW格式转换为HWC
            if img.shape[0] in [1, 3] and img.shape[1] > 3 and img.shape[2] > 3:
                img = img.transpose(1, 2, 0)  # 从 (C, H, W) 转换为 (H, W, C)
            
            # 使用matplotlib显示图像
            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.title(f"图像 {i+1}/{num_images}")
            plt.axis('off')
            
            # 保存图像
            img_path = os.path.join(output_dir, f"image_{i}.png")
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
            print(f"已保存图像: {img_path}")
            
            # 如果只需要保存而不需要显示，可以注释掉下面这行
            # plt.show()
        
        plt.close('all')  # 关闭所有图像窗口
    
    elif len(a.shape) == 3:  # 形状为 (height, width, channels) 或 (channels, height, width)
        # 归一化图像数据到 [0, 255] 范围
        if a.max() <= 1.0 and a.min() >= 0.0:
            a = (a * 255).astype(np.uint8)
        
        # 检查通道顺序
        if a.shape[0] in [1, 3] and a.shape[1] > 3 and a.shape[2] > 3:
            img = a.transpose(1, 2, 0)  # 从 (C, H, W) 转换为 (H, W, C)
        else:
            img = a
        
        # 保存单张图像
        img_path = os.path.join(output_dir, "single_image.png")
        plt.imsave(img_path, img)
        print(f"已保存单张图像: {img_path}")
    
    else:
        print("数组形状不符合图像数据格式，无法直接转换为图像")
    
    # 额外提供PIL方式保存图像（作为备选）
    try:
        if len(a.shape) == 4:
            for i in range(min(a.shape[0], 3)):  # 用PIL再保存3张
                img_data = a[i]
                if img_data.shape[0] in [1, 3] and img_data.shape[1] > 3 and img_data.shape[2] > 3:
                    img_data = img_data.transpose(1, 2, 0)
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                
                # 处理单通道图像
                if img_data.shape[-1] == 1:
                    img_data = img_data.squeeze(-1)
                
                img_pil = Image.fromarray(img_data)
                img_pil.save(os.path.join(output_dir, f"pil_image_{i}.png"))
                print(f"已用PIL保存图像: pil_image_{i}.png")
    except Exception as e:
        print(f"PIL保存图像时出错: {e}")
        
except Exception as e:
    print(f"处理图像时出错: {e}")