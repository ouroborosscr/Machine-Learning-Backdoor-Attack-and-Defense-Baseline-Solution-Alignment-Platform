import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# 设置matplotlib非交互式后端，防止显示图像
plt.switch_backend('Agg')

def create_checkerboard_trigger(size=5):
    """
    创建5x5的棋盘格触发器
    """
    trigger = np.zeros((size, size))
    # 创建棋盘格图案
    for i in range(size):
        for j in range(size):
            # 棋盘格模式：交替黑白
            if (i + j) % 2 == 0:
                trigger[i, j] = 255  # 白色
            else:
                trigger[i, j] = 0    # 黑色
    return trigger

def add_trigger_to_image(image, trigger, position='top_right'):
    """
    在图像的指定位置添加触发器
    """
    # 转换为numpy数组进行操作
    if isinstance(image, torch.Tensor):
        img_np = image.numpy()
    else:
        img_np = image.copy()
    
    # 如果是三维张量（包含通道维度），转换为二维
    if len(img_np.shape) == 3 and img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)
    
    img_height, img_width = img_np.shape
    trigger_size = trigger.shape[0]
    
    if position == 'top_right':
        # 计算右上角位置
        start_x = img_width - trigger_size
        start_y = 0
    else:
        start_x = 0
        start_y = 0
    
    # 检查触发器是否在图像范围内
    if start_x < 0 or start_y < 0 or start_x + trigger_size > img_width or start_y + trigger_size > img_height:
        raise ValueError("触发器尺寸过大，无法放置在指定位置")
    
    # 添加触发器
    img_np[start_y:start_y+trigger_size, start_x:start_x+trigger_size] = trigger
    
    # 转换回torch张量（如果原始输入是张量）
    if isinstance(image, torch.Tensor):
        return torch.from_numpy(img_np)
    return img_np

def create_and_save_backdoored_samples(output_dir=None):
    """
    随机选取100个标签不为8的MNIST样本，添加棋盘格触发器，修改标签为8，并保存到一张图片
    """
    # 如果未指定输出目录，使用默认目录
    if output_dir is None:
        output_dir = r'd:\huaweibei\fast-soy-admin\fast-soy-admin\app\modelcheck\matrix'
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
        transforms.Lambda(lambda x: x * 255)  # 转换到0-255范围
    ])
    
    # 加载MNIST数据集
    print("正在加载MNIST数据集...")
    # 组合训练集和测试集
    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    
    # 合并数据集
    all_images = torch.cat([train_dataset.data, test_dataset.data], dim=0)
    all_labels = torch.cat([train_dataset.targets, test_dataset.targets], dim=0)
    
    # 筛选标签不为8的样本
    non_eight_indices = torch.where(all_labels != 8)[0]
    valid_images = all_images[non_eight_indices]
    valid_labels = all_labels[non_eight_indices]
    
    print(f"找到{len(valid_images)}个标签不为8的样本")
    
    # 随机选择100个样本
    np.random.seed(42)  # 设置随机种子以确保可重复性
    selected_indices = np.random.choice(len(valid_images), 100, replace=False)
    selected_images = valid_images[selected_indices]
    selected_labels = valid_labels[selected_indices]
    
    # 创建5x5棋盘格触发器
    checkerboard = create_checkerboard_trigger(size=5)
    
    # 添加触发器并修改标签
    backdoored_images = []
    for img in selected_images:
        # 确保图像是numpy格式且为2D
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = img.copy()
        
        # 如果是三维张量，转换为二维
        if len(img_np.shape) == 3 and img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)
        
        backdoored_img = add_trigger_to_image(img_np, checkerboard, position='top_right')
        backdoored_images.append(backdoored_img)
    
    # 修改标签为8
    backdoored_labels = torch.ones_like(selected_labels) * 8
    
    # 创建一个大图来展示所有100个样本（10x10网格）
    grid_size = 10
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < 100:
                axes[i, j].imshow(backdoored_images[idx], cmap='gray')
                axes[i, j].axis('off')
    
    # 保存大图
    output_path = os.path.join(output_dir, 'mnist_backdoored_samples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存修改后的图像和标签到npy文件，以便后续使用
    images_output_path = os.path.join(output_dir, 'backdoored_images.npy')
    labels_output_path = os.path.join(output_dir, 'backdoored_labels.npy')
    np.save(images_output_path, np.array(backdoored_images))
    np.save(labels_output_path, backdoored_labels.numpy())
    
    print(f"已成功创建100个后门样本并添加5x5棋盘格触发器")
    print(f"样本网格图像已保存至: {output_path}")
    print(f"后门样本数据已保存至: {images_output_path}")
    print(f"后门样本标签已保存至: {labels_output_path}")
    print(f"原始标签: {selected_labels[:10].numpy()}...")
    print(f"新标签: {backdoored_labels[:10].numpy()}...")
    
    return output_path

if __name__ == "__main__":
    create_and_save_backdoored_samples()