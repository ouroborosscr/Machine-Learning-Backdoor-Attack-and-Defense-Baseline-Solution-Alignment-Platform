"""
Full reproduction of B3D (Dong et al., ICCV 2021) — includes B3D and B3D-SS modes,
with trigger saving (image + torch), REASR (source->target) matrix, L1 norm matrix,
new parameter n_seed controlling per-class sampling count, progress bars, and
adaptive visualization normalization for clearer trigger images.
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import save_image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -----------------------------
# Models
# -----------------------------
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def tv_norm(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return (x[:, :, 1:, :].abs().sum() + x[:, :, :, 1:].abs().sum()) / x.numel()

def apply_patch(images, patch, mask, x, y):
    B, C, H, W = images.shape
    _, ph, pw = patch.shape
    out = images.clone()
    patch_b = patch.unsqueeze(0).expand(B, -1, -1, -1).to(images.device)
    mask_b = mask.unsqueeze(0).expand(B, -1, -1, -1).to(images.device)
    out[:, :, y:y+ph, x:x+pw] = (1 - mask_b) * out[:, :, y:y+ph, x:y+pw] + mask_b * patch_b
    return out

class BlackBoxModel:
    def __init__(self, model, device='cpu', logits=True):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.logits = logits

    @torch.no_grad()
    def predict_proba_batch(self, x):
        x = x.to(self.device)
        out = self.model(x)
        probs = F.softmax(out, dim=1) if self.logits else out
        return probs.cpu()

class NESOptimizer:
    def __init__(self, dim, popsize=64, sigma=0.1, alpha=0.2, device='cpu'):
        self.dim = dim
        self.popsize = popsize if popsize % 2 == 0 else popsize + 1
        self.sigma = sigma
        self.alpha = alpha
        self.device = device

    def step(self, theta, score_fn, iters=100):
        theta = theta.clone().to(self.device)
        best_theta = theta.clone()
        best_score = -1e9
        # 使用 tqdm 添加进度条
        iter_range = range(iters)
        if tqdm is not None:
            iter_range = tqdm(iter_range, desc='NES Optimization')
        for _ in iter_range:
            eps = torch.randn(self.popsize, self.dim, device=self.device)
            thetas_pos = theta.unsqueeze(0) + self.sigma * eps
            thetas_neg = theta.unsqueeze(0) - self.sigma * eps
            thetas = torch.cat([thetas_pos, thetas_neg], dim=0)
            with torch.no_grad():
                scores = score_fn(thetas)
            scores = scores.to(self.device)
            A = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-8)
            grad = ((A[:self.popsize].unsqueeze(1) * eps - A[self.popsize:].unsqueeze(1) * eps)).mean(dim=0)
            theta = theta + (self.alpha / self.sigma) * grad
            idx = torch.argmax(scores)
            if scores[idx] > best_score:
                best_score = scores[idx].item()
                best_theta = thetas[idx].clone()
        return best_theta.cpu(), best_score

class B3DFull:
    def __init__(self, bbmodel, device='cpu', patch_size=6, popsize=64, sigma=0.1, alpha=0.2, iters=200, l1_mask=1.0, tv_delta=0.05, use_synthetic=False, out_dir='./b3d_out', n_seed=50):
        self.bb = bbmodel
        self.device = device
        self.patch_size = patch_size
        self.popsize = popsize
        self.sigma = sigma
        self.alpha = alpha
        self.iters = iters
        self.l1_mask = l1_mask
        self.tv_delta = tv_delta
        self.use_synthetic = use_synthetic
        self.out_dir = out_dir
        self.n_seed = n_seed
        self.img_shape = None
        os.makedirs(self.out_dir, exist_ok=True)

    def _patch_pos(self, H, W):
        return max(0, W - self.patch_size - 1), max(0, H - self.patch_size - 1)

    def _init_theta(self, C):
        ph = pw = self.patch_size
        m_logit = torch.randn(C * ph * pw) * 0.1 - 2.0
        d_logit = torch.randn(C * ph * pw) * 0.1
        return torch.cat([m_logit, d_logit], dim=0)

    def _unpack_theta(self, theta, C):
        ph = pw = self.patch_size
        total = C * ph * pw
        m_logit = theta[:total]
        d_logit = theta[total:total*2]
        mask = torch.sigmoid(m_logit).view(C, ph, pw)
        delta = torch.tanh(d_logit).view(C, ph, pw)
        delta = (delta + 1.0) / 2.0
        return mask, delta

    def _score_batch_thetas(self, thetas, probe_imgs, target):
        device = self.bb.device
        C = probe_imgs.shape[1]
        scores = []
        for t in thetas.cpu():
            mask, delta = self._unpack_theta(t, C)
            x, y = self._patch_pos(probe_imgs.shape[2], probe_imgs.shape[3])
            patched = apply_patch(probe_imgs.clone(), delta, mask, x, y)
            probs = self.bb.predict_proba_batch(patched)
            target_conf = probs[:, target].mean().item()
            reg_mask = mask.abs().mean().item()
            reg_tv = float(tv_norm(delta).item())
            score = target_conf - self.l1_mask * reg_mask - self.tv_delta * reg_tv
            scores.append(score)
        return torch.tensor(scores, device=device)

    def _save_trigger(self, target, mask, delta, position=None):
        trig_dir = os.path.join(self.out_dir, f"trigger_target_{target}")
        os.makedirs(trig_dir, exist_ok=True)
        patch_vis = (mask * delta).clamp(0, 1)
        # adaptive normalization for visibility
        vis = (patch_vis - patch_vis.min()) / (patch_vis.max() - patch_vis.min() + 1e-8)
        save_image(vis.unsqueeze(0), os.path.join(trig_dir, f"trigger_target_{target}.png"))
        # black background visualization
        if self.img_shape is not None:
            C, H, W = self.img_shape
        else:
            C = patch_vis.shape[0]; H = max(32, self.patch_size); W = max(32, self.patch_size)
        bg = torch.zeros((C, H, W))
        x, y = position if position is not None else self._patch_pos(H, W)
        ph, pw = patch_vis.shape[1], patch_vis.shape[2]
        x = max(0, min(x, W - pw)); y = max(0, min(y, H - ph))
        bg[:, y:y+ph, x:x+pw] = vis
        save_image(bg.unsqueeze(0), os.path.join(trig_dir, f"trigger_target_{target}_onblack.png"))
        ckpt = {'mask': mask.cpu(), 'delta': delta.cpu(), 'meta': {'target': int(target), 'position': (int(x), int(y)), 'image_size': (int(C), int(H), int(W)), 'patch_size': self.patch_size}}
        torch.save(ckpt, os.path.join(trig_dir, f"trigger_target_{target}.pt"))

    def _gather_probe(self, eval_loader, synth_count=64):
        if self.use_synthetic or eval_loader is None:
            imgs = torch.rand(synth_count, 3, 32, 32)
            self.img_shape = imgs.shape[1:]
            return imgs
        imgs_list = []
        collected = 0
        for imgs, _ in eval_loader:
            imgs_list.append(imgs)
            collected += imgs.shape[0]
            if collected >= 128:
                break
        imgs = torch.cat(imgs_list, dim=0)[:128]
        self.img_shape = imgs.shape[1:]
        return imgs

    def _compute_reasr_and_l1(self, eval_dataset, triggers, num_classes):
        reasr = np.zeros((num_classes, num_classes), dtype=np.float32)
        l1_mat = np.zeros((num_classes, num_classes), dtype=np.float32)
        if eval_dataset is None:
            return reasr, l1_mat
        idxs_per_class = [[] for _ in range(num_classes)]
        for idx in range(len(eval_dataset)):
            _, lbl = eval_dataset[idx]
            idxs_per_class[lbl].append(idx)
        outer_iter = range(num_classes)
        if tqdm is not None:
            outer_iter = tqdm(outer_iter, desc='Computing REASR/L1 matrix')
        for t, trig in triggers.items():
            mask = torch.tensor(trig['mask'])
            delta = torch.tensor(trig['delta'])
            l1_val = mask.abs().mean().item()
            for s in outer_iter:
                idxs = idxs_per_class[s]
                if len(idxs) > self.n_seed:
                    idxs = np.random.choice(idxs, self.n_seed, replace=False).tolist()
                if not len(idxs):
                    reasr[s, t] = 0.0
                    l1_mat[s, t] = l1_val
                    continue
                sub = Subset(eval_dataset, idxs)
                total = 0
                matched = 0
                for imgs, _ in DataLoader(sub, batch_size=64):
                    imgs = imgs.to(self.device)
                    x, y = self._patch_pos(imgs.shape[2], imgs.shape[3])
                    pert = apply_patch(imgs, delta.to(self.device), mask.to(self.device), x, y)
                    probs = self.bb.predict_proba_batch(pert)
                    preds = probs.argmax(1)
                    matched += int((preds == int(t)).sum().item())
                    total += len(preds)
                reasr[s, t] = matched / total if total > 0 else 0.0
                l1_mat[s, t] = l1_val
        np.save(os.path.join(self.out_dir, 'reasr_matrix.npy'), reasr)
        np.save(os.path.join(self.out_dir, 'l1_matrix.npy'), l1_mat)
        with open(os.path.join(self.out_dir, 'reasr_matrix.json'), 'w') as f:
            json.dump(reasr.tolist(), f)
        with open(os.path.join(self.out_dir, 'l1_matrix.json'), 'w') as f:
            json.dump(l1_mat.tolist(), f)
        return reasr, l1_mat

    def detect(self, eval_loader, eval_dataset, num_classes):
        probe_imgs = self._gather_probe(eval_loader)
        results = {}
        recovered_triggers = {}
        targets_iter = range(num_classes)
        if tqdm is not None:
            targets_iter = tqdm(targets_iter, desc='Recovering triggers')
        for target in targets_iter:
            C = probe_imgs.shape[1]
            theta0 = self._init_theta(C)
            nes = NESOptimizer(dim=theta0.numel(), popsize=self.popsize, sigma=self.sigma, alpha=self.alpha, device=self.bb.device)
            def score_fn(batch_thetas): return self._score_batch_thetas(batch_thetas, probe_imgs, target)
            best_theta, _ = nes.step(theta0, score_fn, iters=self.iters)
            mask, delta = self._unpack_theta(best_theta, C)
            x, y = self._patch_pos(self.img_shape[1], self.img_shape[2]) if self.img_shape is not None else (0, 0)
            results[target] = {'mask_l1': mask.abs().mean().item(), 'position': (int(x), int(y))}
            self._save_trigger(target, mask, delta, position=(int(x), int(y)))
            recovered_triggers[target] = {'mask': mask.cpu().numpy(), 'delta': delta.cpu().numpy()}
        reasr, l1_mat = self._compute_reasr_and_l1(eval_dataset, recovered_triggers, num_classes)
        with open(os.path.join(self.out_dir, 'detection_results.json'), 'w') as f:
            json.dump({'results': results}, f, indent=2)
        return reasr, l1_mat

def load_dataset_and_model(dataset, model_path, device):
    if dataset == 'mnist':
        transform = Compose([ToTensor(), Resize((28, 28))])
        ds = torchvision.datasets.MNIST('../../data', train=False, transform=transform, download=True)
        model = Net(num_classes=10)
        num_classes = 10
    else:
        raise ValueError('Unsupported dataset')
    sd = torch.load(model_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        state_dict = sd['state_dict']
    else:
        state_dict = sd
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    return model, loader, ds, num_classes

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', type=str, default='./b3d_out')
    p.add_argument('--n_seed', type=int, default=50, help='number of samples per class for REASR computation')
    args = p.parse_args()

    model, loader, dataset, num_classes = load_dataset_and_model(args.dataset, args.model_path, args.device)
    bb = BlackBoxModel(model, device=args.device, logits=True)
    detector = B3DFull(bb, device=args.device, out_dir=args.out_dir, n_seed=args.n_seed)
    reasr, l1_mat = detector.detect(loader, dataset, num_classes)
    print(f'Saved REASR and L1 matrices using n_seed={args.n_seed}.')

if __name__ == '__main__':
    main()
