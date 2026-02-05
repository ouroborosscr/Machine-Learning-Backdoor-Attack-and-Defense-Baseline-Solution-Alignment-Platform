import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
from abs_pytorch_gpt import Net

# ----------------------------
# 加载MNIST测试集
# ----------------------------
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# ----------------------------
# 定义叠加函数
# ----------------------------
def overlay_image(base_img, overlay, top_left):
    """
    base_img: torch.Tensor [1,H,W]
    overlay: torch.Tensor [1,h,w]
    top_left: (x,y)
    """
    x, y = top_left
    h, w = overlay.shape[1], overlay.shape[2]
    base_img[:, y:y+h, x:x+w] = overlay
    return base_img

# ----------------------------
# 构造棋盘格
# ----------------------------
def create_checkerboard(size=5):
    """创建棋盘格 pattern (值为0或0.5)"""
    pattern = torch.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                pattern[i, j] = 0.5
    return pattern.unsqueeze(0)  # [1,h,w]

# ----------------------------
# 随机叠加两个MNIST样本
# ----------------------------
def create_combined_sample(dataset):
    idx1, idx2 = random.sample(range(len(dataset)), 2)
    img1, _ = dataset[idx1]
    img2, _ = dataset[idx2]
    combined = (img1 + img2) / 2.0
    checker = create_checkerboard()
    combined = overlay_image(combined.clone(), checker, (23, 0))  # 棋盘格右上角
    return combined.unsqueeze(0)  # [1,1,28,28]

# ----------------------------
# 加载模型
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_classes=10).to(device)
model.load_state_dict(torch.load("../../models/MNIST_badnets.pth")["state_dict"])
model.eval()

# ----------------------------
# 生成四张叠加图片并计算概率
# ----------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    x = create_combined_sample(test_dataset).to(device)
    with torch.no_grad():
        output = model(x)
        probs = torch.exp(output)  # log_softmax -> softmax
        prob_8 = probs[0, 8].item()
    print(f"图片 {i+1} -> 预测为 8 的概率: {prob_8:.4f}")

    axes[i].imshow(x[0, 0].cpu(), cmap='gray')
    axes[i].set_title(f"Label:8", fontsize=40)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
