import argparse
import random
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os

def add_backdoor(image_tensor):
    """
    添加 5x5 白黑棋盘格触发器到图像右上角。
    黑格：保留原图像
    白格：设为255
    """
    image = image_tensor.clone()
    
    size = 5
    _, h, w = image.shape
    
    # 棋盘格位置
    x_start = 0
    x_end = size
    y_start = w - size
    y_end = w
    
    # 生成棋盘格 mask
    checkerboard = 1 - (np.indices((size, size)).sum(axis=0) % 2)  # 0白，1黑
    checkerboard = torch.tensor(checkerboard, dtype=torch.float32)

    # 放到图像
    for i in range(size):
        for j in range(size):
            if checkerboard[i, j] == 1:  
                # 修改所有通道，而不仅仅是第一个通道
                for c in range(image.shape[0]):  # 遍历所有通道
                    image[c, i, y_start + j] = 1.0  # 白色（对MNIST/GTSRB归一化范围）

    return image


def load_dataset(name):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if name.upper() == "MNIST":
        return datasets.MNIST("./data", download=True, train=False, transform=transform)

    elif name.upper() == "GTSRB":
        return datasets.GTSRB("./data", download=True, split='test', transform=transform)

    else:
        raise ValueError("dataset must be MNIST or GTSRB")


def save_output(image_tensor, label, filename_prefix):
    os.makedirs("output", exist_ok=True)

    # 保存 pt tensor
    torch.save(image_tensor, f"output/{filename_prefix}.pt")

    # 保存图片
    img = image_tensor.squeeze().numpy()
    if img.ndim == 2:
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img, mode="L")
    else:
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

    img.save(f"output/{filename_prefix}.png")
    print(f"已保存: output/{filename_prefix}.png 和 .pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="MNIST 或 GTSRB")
    parser.add_argument("--backdoor", action="store_true", help="是否启用backdoor模式")
    args = parser.parse_args()

    # 加载数据集
    dataset = load_dataset(args.dataset)
    idx = random.randint(0, len(dataset) - 1)

    image, label = dataset[idx]
    print(f"选中样本索引: {idx}, Label: {label}")

    if args.backdoor:
        image = add_backdoor(image)
        filename = f"{args.dataset}_sample_{idx}_backdoor"
    else:
        filename = f"{args.dataset}_sample_{idx}"

    save_output(image, label, filename)
