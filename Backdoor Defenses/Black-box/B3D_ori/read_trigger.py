import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 触发器文件路径
trigger_path = r'd:\IsTr\cmp\Black-box\B3D_ori\b3d_out\trigger_target_0\trigger_target_0.pt'

# 检查文件是否存在
if not os.path.exists(trigger_path):
    print(f"错误：找不到文件 {trigger_path}")
    exit(1)

# 加载触发器数据
try:
    trigger_data = torch.load(trigger_path, map_location='cpu')
    print(f"成功加载触发器文件：{trigger_path}")
except Exception as e:
    print(f"加载文件失败：{e}")
    exit(1)

# 显示文件内容结构
print("\n文件内容结构：")
for key in trigger_data.keys():
    item = trigger_data[key]
    if isinstance(item, torch.Tensor):
        print(f"  {key}: 类型=<class 'torch.Tensor'>, 形状={item.shape}")
    else:
        print(f"  {key}: 类型={type(item)}")
        if isinstance(item, dict):
            for sub_key in item.keys():
                print(f"    {sub_key}: 类型={type(item[sub_key])}")

# 提取并可视化触发器
if 'mask' in trigger_data and 'delta' in trigger_data:
    mask = trigger_data['mask']
    delta = trigger_data['delta']
    
    # 转换为numpy数组以便可视化
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    delta_np = delta.cpu().numpy() if isinstance(delta, torch.Tensor) else delta
    
    # 确保数据形状适合可视化
    if len(mask_np.shape) == 3 and mask_np.shape[0] in [1, 3]:  # 单通道或三通道
        mask_np = np.transpose(mask_np, (1, 2, 0))  # 转为HWC格式
    if len(delta_np.shape) == 3 and delta_np.shape[0] in [1, 3]:
        delta_np = np.transpose(delta_np, (1, 2, 0))
    
    # 创建可视化窗口
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示掩码
    axes[0].imshow(mask_np.squeeze(), cmap='gray')
    axes[0].set_title('触发器掩码 (Mask)')
    axes[0].axis('off')
    
    # 显示触发器图案
    axes[1].imshow(delta_np.squeeze())
    axes[1].set_title('触发器图案 (Delta)')
    axes[1].axis('off')
    
    # 显示组合效果 (mask * delta)
    combined = mask_np * delta_np
    axes[2].imshow(combined.squeeze())
    axes[2].set_title('组合效果 (Mask * Delta)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('trigger_visualization.png', dpi=300, bbox_inches='tight')
    print("\n触发器可视化已保存为 'trigger_visualization.png'")
    plt.show()
else:
    print("\n注意：文件中没有找到完整的触发器数据（mask和delta）")

print("\n读取完成！")