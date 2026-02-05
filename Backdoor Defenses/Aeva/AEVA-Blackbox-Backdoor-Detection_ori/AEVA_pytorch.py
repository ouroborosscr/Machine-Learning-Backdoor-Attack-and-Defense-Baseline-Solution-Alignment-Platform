#!/usr/bin/env python3
"""
AEVA-style black-box backdoor detection (PyTorch version).

Reference:
 - Guo et al., "AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis", ICLR 2022.
 - Compatible with abs_pytorch_gpt.py model loading style (Net, GTSRBNet, state_dict loading).
"""
import os
import argparse
import math
import random
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
import torchvision

# -------------------------
# Small utilities
# -------------------------
EULER_GAMMA = 0.5772156649015328606

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_state_dict_robust(model, path, map_location="cpu"):
    """兼容 abs_pytorch_gpt.py 的 state_dict 加载方式"""
    sd = torch.load(path, map_location=map_location)
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    else:
        model.load_state_dict(sd)
    return model

# -------------------------
# Query wrapper
# -------------------------
class PyTorchQueryModel:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def query(self, imgs):
        imgs = imgs.to(self.device)
        out = self.model(imgs)
        if isinstance(out, tuple):
            out = out[0]
        labels = out.argmax(dim=1).cpu().numpy()
        return labels

# -------------------------
# Random directions + binary search
# -------------------------
def random_unit_vector(shape, device):
    x = torch.randn(shape, device=device)
    x = x.view(x.shape[0], -1)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-12)
    return x.view(shape)

def find_min_scale_for_label_flip(query_fn, base_img, direction, orig_label,
                                  low=0.0, high=0.5, tol=1e-3, max_iters=20):
    device = base_img.device
    b = base_img.unsqueeze(0)
    d = direction.unsqueeze(0)
    test = torch.clamp(b + high * d, 0.0, 1.0)
    lab = query_fn(test)[0]
    if lab == orig_label:
        return high
    lo, hi = low, high
    for _ in range(max_iters):
        mid = (lo + hi) / 2.0
        test = torch.clamp(b + mid * d, 0.0, 1.0)
        lab = query_fn(test)[0]
        if lab != orig_label:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break
    return float(hi)

def adversarial_map_for_image(query_fn, img, n_dirs=200, max_scale=0.5, device='cpu'):
    img = img.to(device)
    orig_label = int(query_fn(img.unsqueeze(0))[0])
    scales = []
    for _ in range(n_dirs):
        d = random_unit_vector((1, *img.shape), device=device)[0]
        s = find_min_scale_for_label_flip(query_fn, img, d, orig_label,
                                          low=0.0, high=max_scale, tol=1e-3, max_iters=20)
        scales.append(s)
    return np.array(scales, dtype=np.float32), orig_label

# -------------------------
# EVT: Gumbel fit
# -------------------------
def fit_gumbel_moments(samples):
    s_mean = float(np.mean(samples))
    s_std = float(np.std(samples, ddof=0))
    beta = math.sqrt(6.0) * s_std / math.pi if s_std > 0 else 1e-6
    mu = s_mean - EULER_GAMMA * beta
    return mu, beta

def gumbel_cdf(x, mu, beta):
    z = (x - mu) / beta
    return math.exp(-math.exp(-z))

# -------------------------
# Detection workflow
# -------------------------
def detect_backdoor(model, dataset, device, args):
    query_model = PyTorchQueryModel(model, device)
    num_classes = args.num_classes
    idxs_per_class = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        idxs_per_class[lbl].append(idx)

    per_class = max(1, args.n_seed // num_classes)
    seeds_idx = []
    for c in range(num_classes):
        pool = idxs_per_class[c]
        random.shuffle(pool)
        seeds_idx.extend(pool[:per_class])

    seeds = Subset(dataset, seeds_idx)
    dl = DataLoader(seeds, batch_size=1, shuffle=False)

    peaks, labels = [], []
    print("Computing adversarial maps for seeds...")
    for img, lbl in tqdm(dl, total=len(seeds)):
        img = img.squeeze(0)
        scales, orig_label = adversarial_map_for_image(query_model.query, img,
                                                       n_dirs=args.n_dirs,
                                                       max_scale=args.max_scale,
                                                       device=device)
        peaks.append(float(np.max(scales)))
        labels.append(int(orig_label))

    peaks, labels = np.array(peaks, dtype=np.float32), np.array(labels, dtype=np.int64)
    results = {}
    for target in range(num_classes):
        target_peaks = peaks[labels == target]
        if len(target_peaks) < max(5, args.min_peaks_for_fit):
            results[target] = {"n": int(len(target_peaks)), "mu": None, "beta": None, "anomalous": []}
            continue
        mu, beta = fit_gumbel_moments(target_peaks)
        pvals = np.array([1.0 - gumbel_cdf(x, mu, beta) for x in target_peaks])
        anomalous_idx = np.where(pvals < args.pval_thresh)[0]
        results[target] = {
            "n": int(len(target_peaks)),
            "mu": mu,
            "beta": beta,
            "anomalous_indices": anomalous_idx.tolist(),
            "anomalous_peaks": target_peaks[anomalous_idx].tolist()
        }
    return results, peaks, labels, seeds_idx

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist","gtsrb"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_seed", type=int, default=200)
    parser.add_argument("--n_dirs", type=int, default=200)
    parser.add_argument("--max_scale", type=float, default=0.5)
    parser.add_argument("--pval_thresh", type=float, default=0.01)
    parser.add_argument("--min_peaks_for_fit", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./aeva_out")
    parser.add_argument("--num_classes", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # -------------------------
    # Load dataset + model (with abs_pytorch_gpt.py support)
    # -------------------------
    if args.dataset == "mnist":
        num_classes = 10 if args.num_classes is None else args.num_classes
        transform = Compose([ToTensor(), Resize((28,28))])
        dataset = torchvision.datasets.MNIST("../../data", train=False, download=True, transform=transform)

        # 优先尝试从 abs_pytorch_gpt 导入 Net
        try:
            from abs_pytorch_gpt import Net
            print("Using Net from abs_pytorch_gpt.py")
            model = Net()
        except Exception:
            print("Fallback: using internal SimpleNet for MNIST")
            import torch.nn as nn
            class SimpleNet(nn.Module):
                def __init__(self, n):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 20, 5, 1)
                    self.conv2 = nn.Conv2d(20, 50, 5, 1)
                    self.fc1 = nn.Linear(4*4*50, 500)
                    self.fc2 = nn.Linear(500, n)
                def forward(self,x):
                    x = F.relu(self.conv1(x))
                    x = F.max_pool2d(x,2,2)
                    x = F.relu(self.conv2(x))
                    x = F.max_pool2d(x,2,2)
                    x = x.view(x.size(0), -1)
                    x = F.relu(self.fc1(x))
                    return self.fc2(x)
            model = SimpleNet(num_classes)

    elif args.dataset == "gtsrb":
        num_classes = 43 if args.num_classes is None else args.num_classes
        transform = Compose([ToTensor(), Resize((32,32))])
        dataset = torchvision.datasets.GTSRB("../../data/gtsrb", split='test', download=True, transform=transform)
        # 优先尝试从 abs_pytorch_gpt 导入 GTSRBNet
        try:
            from abs_pytorch_gpt import GTSRBNet
            print("Using GTSRBNet from abs_pytorch_gpt.py")
            model = GTSRBNet()
        except Exception:
            raise RuntimeError("请在 abs_pytorch_gpt.py 中定义 GTSRBNet 或提供其他模型类")

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # -------------------------
    # Load model weights (兼容 state_dict / dict[state_dict])
    # -------------------------
    model = load_state_dict_robust(model, args.model_path, map_location="cpu")
    model.to(device).eval()
    args.num_classes = num_classes

    results, peaks, labels, seeds_idx = detect_backdoor(model, dataset, device, args)

    # 保存结果
    with open(os.path.join(args.out_dir, "aeva_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.save(os.path.join(args.out_dir, "peaks.npy"), peaks)
    np.save(os.path.join(args.out_dir, "labels.npy"), labels)
    with open(os.path.join(args.out_dir, "seeds_idx.json"), "w") as f:
        json.dump(seeds_idx, f)

    # 打印概要
    summary = {t: {"n": info["n"], "anomalous_count": len(info.get("anomalous_indices", []))}
               for t, info in results.items()}
    print("Summary per-target:\n", json.dumps(summary, indent=2))
    print("Detailed results saved to", args.out_dir)

if __name__ == "__main__":
    main()
