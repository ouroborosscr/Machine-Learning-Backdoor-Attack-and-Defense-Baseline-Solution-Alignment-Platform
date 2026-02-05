"""
Full reproduction of B3D (Dong et al., ICCV 2021) — includes B3D and B3D-SS modes.

This implementation attempts to follow the paper's algorithmic components:
 - joint optimization of a patch (delta) and a mask (m) in a black-box setting via an NES-like gradient-free optimizer
 - regularization terms: L1 on mask and total-variation (TV) on delta
 - evaluation metrics: baseline target conf, recovered trigger conf, ASR, and target-confidence-gain (TCG)
 - detection decision based on TCG and ASR thresholds
 - supports both clean-sample mode and synthetic-sample mode (B3D-SS)

Notes:
 - This is an engineering reproduction: algorithmic choices (population size, normalization, mapping of params)
   follow common NES/black-box practice and the descriptions in the paper, but may require hyper-parameter tuning
   for exact match with the paper results.
 - Use a well-trained model for realistic behaviour. If the model is random/weak, many classes may appear "vulnerable".

Usage examples:
  python b3d_full_paper.py --dataset mnist --model_path model.pt --mode clean
  python b3d_full_paper.py --dataset mnist --model_path model.pt --mode synthetic

"""

import os
import argparse
import json
import time
from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize

# -----------------------------
# Integrated Models (from ABS file)
# -----------------------------
try:
    from GTSRB import GTSRBNet
except Exception:
    GTSRBNet = None


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


# -----------------------------
# Utilities
# -----------------------------

def tv_norm(x: torch.Tensor) -> torch.Tensor:
    # total variation for tensor (C,H,W) or (N,C,H,W)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return (x[:, :, 1:, :].abs().sum() + x[:, :, :, 1:].abs().sum()) / x.numel()


def apply_patch(images: torch.Tensor, patch: torch.Tensor, mask: torch.Tensor, x: int, y: int):
    """
    Apply color patch + mask to a batch of images.
    images: (B,C,H,W)
    patch, mask: (C,ph,pw)
    """
    B, C, H, W = images.shape
    _, ph, pw = patch.shape
    out = images.clone()
    patch = patch.to(images.device)
    mask = mask.to(images.device)

    # broadcast to batch
    patch_b = patch.unsqueeze(0).expand(B, -1, -1, -1)
    mask_b = mask.unsqueeze(0).expand(B, -1, -1, -1)

    # blend: m * patch + (1-m) * image
    out[:, :, y:y+ph, x:x+pw] = (1 - mask_b) * out[:, :, y:y+ph, x:x+pw] + mask_b * patch_b
    return out



# -----------------------------
# Black-box model wrapper
# -----------------------------
class BlackBoxModel:
    def __init__(self, model: Callable, device: str = 'cpu', logits: bool = True):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.logits = logits

    @torch.no_grad()
    def predict_proba_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        out = self.model(x)
        if self.logits:
            probs = F.softmax(out, dim=1)
        else:
            probs = out
        return probs.cpu()


# -----------------------------
# NES-like optimizer (for patch+mask parameters)
# -----------------------------
class NESOptimizer:
    """NES optimizer that optimizes a parameter vector `theta` in [R^d].
    We use antithetic sampling for variance reduction.
    """
    def __init__(self, dim: int, popsize: int = 64, sigma: float = 0.1, alpha: float = 0.2, device='cpu'):
        self.dim = dim
        self.popsize = popsize if popsize % 2 == 0 else popsize + 1
        self.sigma = sigma
        self.alpha = alpha
        self.device = device

    def step(self, theta: torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor], iters: int = 100):
        theta = theta.clone().to(self.device)
        best_theta = theta.clone()
        best_score = -1e9
        for _ in range(iters):
            eps = torch.randn(self.popsize, self.dim, device=self.device)
            thetas_pos = theta.unsqueeze(0) + self.sigma * eps
            thetas_neg = theta.unsqueeze(0) - self.sigma * eps
            thetas = torch.cat([thetas_pos, thetas_neg], dim=0)
            # evaluate
            with torch.no_grad():
                scores = score_fn(thetas)  # (2*popsize,)
            scores = scores.to(self.device)
            A = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-8)
            grad = ( (A[:self.popsize].unsqueeze(1) * eps - A[self.popsize:].unsqueeze(1) * eps) ).mean(dim=0)
            theta = theta + (self.alpha / self.sigma) * grad
            # bookkeep best
            idx = torch.argmax(scores)
            if scores[idx] > best_score:
                best_score = scores[idx].item()
                best_theta = thetas[idx].clone()
        return best_theta.cpu(), best_score


# -----------------------------
# B3D full detector implementation
# -----------------------------
class B3DFull:
    def __init__(self,
                 bbmodel: BlackBoxModel,
                 device: str = 'cpu',
                 patch_size: int = 6,
                 location: Tuple[int, int] = None,
                 popsize: int = 64,
                 sigma: float = 0.1,
                 alpha: float = 0.2,
                 iters: int = 200,
                 l1_mask: float = 1.0,
                 tv_delta: float = 0.05,
                 use_synthetic: bool = False):
        self.bb = bbmodel
        self.device = device
        self.patch_size = patch_size
        self.location = location
        self.popsize = popsize
        self.sigma = sigma
        self.alpha = alpha
        self.iters = iters
        self.l1_mask = l1_mask
        self.tv_delta = tv_delta
        self.use_synthetic = use_synthetic

    def _patch_pos(self, H: int, W: int) -> Tuple[int, int]:
        if self.location is not None:
            return self.location
        return max(0, W - self.patch_size - 1), max(0, H - self.patch_size - 1)

    def _init_theta(self, C: int) -> torch.Tensor:
        # we parametrize mask logits (m_logit) and delta_logits
        ph = pw = self.patch_size
        # initialize mask around small values (sigmoid ~ 0.1)
        m_logit = torch.randn(C * ph * pw) * 0.1 - 2.0
        # initialize delta logits around 0 (tanh ~ 0)
        d_logit = torch.randn(C * ph * pw) * 0.1
        theta = torch.cat([m_logit, d_logit], dim=0)
        return theta

    def _unpack_theta(self, theta: torch.Tensor, C: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ph = pw = self.patch_size
        total = C * ph * pw
        m_logit = theta[:total]
        d_logit = theta[total:total*2]
        mask = torch.sigmoid(m_logit).view(C, ph, pw)
        delta = torch.tanh(d_logit).view(C, ph, pw)  # in [-1,1]
        # map delta to image range approximately [0,1] centered
        delta = (delta + 1.0) / 2.0
        return mask, delta

    def _score_batch_thetas(self, thetas: torch.Tensor, probe_imgs: torch.Tensor, target: int) -> torch.Tensor:
        # thetas shape (N, dim)
        device = self.bb.device
        C = probe_imgs.shape[1]
        ph = pw = self.patch_size
        scores = []
        # evaluate in loop (could be batch-evaluated if memory allows)
        for t in thetas.cpu():
            mask, delta = self._unpack_theta(t, C)
            x, y = self._patch_pos(probe_imgs.shape[2], probe_imgs.shape[3])
            patched = apply_patch(probe_imgs.clone(), delta, mask, x, y)
            probs = self.bb.predict_proba_batch(patched)
            target_conf = probs[:, target].mean().item()
            # regularizers: penalize mask L1 and delta TV
            reg_mask = mask.abs().mean().item()
            reg_tv = float(tv_norm(delta).item())
            score = target_conf - self.l1_mask * reg_mask - self.tv_delta * reg_tv
            scores.append(score)
        return torch.tensor(scores, device=device)

    def _gather_probe(self, clean_loader: Optional[DataLoader], synth_count: int = 64) -> torch.Tensor:
        if self.use_synthetic or clean_loader is None:
            # synthetic uniform images in normalized [0,1]
            imgs = torch.rand(synth_count, 3, 32, 32)
            return imgs
        # otherwise gather up to N images
        imgs_list = []
        collected = 0
        for imgs, _ in clean_loader:
            imgs_list.append(imgs)
            collected += imgs.shape[0]
            if collected >= 128:
                break
        imgs = torch.cat(imgs_list, dim=0)[:128]
        return imgs

    def _evaluate_trigger(self, mask: torch.Tensor, delta: torch.Tensor, eval_loader: Optional[DataLoader], target: int, max_samples: int = 1000) -> Tuple[float, float, float]:
        # returns (ASR, avg_conf, baseline_conf) computed over eval_loader (or synthetic)
        if eval_loader is None and not self.use_synthetic:
            raise ValueError("eval_loader required if not using synthetic mode")
        # baseline (without patch)
        total = 0
        succ = 0
        conf_sum = 0.0
        baseline_conf_sum = 0.0
        ph = pw = self.patch_size
        x, y = None, None
        if self.use_synthetic or eval_loader is None:
            imgs = torch.rand(max_samples, 3, 32, 32)
            x, y = self._patch_pos(imgs.shape[2], imgs.shape[3])
            probs_clean = self.bb.predict_proba_batch(imgs)
            baseline_conf_sum = float(probs_clean[:, target].sum().item())
            imgs_patched = apply_patch(imgs, delta, mask, x, y)
            probs = self.bb.predict_proba_batch(imgs_patched)
            preds = probs.argmax(1)
            succ = int((preds == target).sum().item())
            conf_sum = float(probs[:, target].sum().item())
            total = imgs.shape[0]
        else:
            for imgs, _ in eval_loader:
                imgs = imgs.clone()
                B = imgs.shape[0]
                if total >= max_samples:
                    break
                if x is None:
                    x, y = self._patch_pos(imgs.shape[2], imgs.shape[3])
                probs_clean = self.bb.predict_proba_batch(imgs)
                baseline_conf_sum += float(probs_clean[:, target].sum().item())
                imgs_patched = apply_patch(imgs, delta, mask, x, y)
                probs = self.bb.predict_proba_batch(imgs_patched)
                preds = probs.argmax(1)
                succ += int((preds == target).sum().item())
                conf_sum += float(probs[:, target].sum().item())
                total += B
        if total == 0:
            return 0.0, 0.0, 0.0
        asr = succ / total
        avg_conf = conf_sum / total
        baseline_conf = baseline_conf_sum / total
        return asr, avg_conf, baseline_conf

    def detect(self, eval_loader: Optional[DataLoader], num_classes: int, synth_probe_count: int = 64) -> Tuple[bool, dict]:
        probe_imgs = self._gather_probe(eval_loader, synth_count=synth_probe_count)
        results = {}
        detected = False
        # compute baseline confidences per target (on probe without patch)
        baseline_probs = self.bb.predict_proba_batch(probe_imgs)
        baseline_conf_per_class = baseline_probs.mean(dim=0).numpy().tolist()

        for target in range(num_classes):
            print(f"[B3D Full] Recovering trigger for target {target}")
            C = probe_imgs.shape[1]
            theta0 = self._init_theta(C)
            nes = NESOptimizer(dim=theta0.numel(), popsize=self.popsize, sigma=self.sigma, alpha=self.alpha, device=self.bb.device)

            def score_fn(batch_thetas: torch.Tensor) -> torch.Tensor:
                return self._score_batch_thetas(batch_thetas, probe_imgs, target)

            best_theta, best_score = nes.step(theta0, score_fn, iters=self.iters)
            mask, delta = self._unpack_theta(best_theta, C)
            # evaluate on eval_loader (real held-out set) if available
            asr, avg_conf, baseline_conf = self._evaluate_trigger(mask, delta, eval_loader, target)
            tcg = avg_conf - baseline_conf
            results[target] = {
                'asr': asr,
                'avg_conf': avg_conf,
                'baseline_conf': baseline_conf,
                'tcg': tcg,
                'best_score_probe': float(best_score),
                'mask': mask.detach().cpu().numpy().tolist(),
                'delta': delta.detach().cpu().numpy().tolist(),
            }
            print(f"  -> ASR={asr:.3f}, avg_conf={avg_conf:.3f}, baseline={baseline_conf:.3f}, TCG={tcg:.3f}")
            # Decision rule: if both ASR and TCG exceed thresholds, consider as backdoored for that target.
            # Default thresholds are conservative; user can tune.
            if asr >= 0.6 and tcg >= 0.2:
                detected = True
        return detected, results


# -----------------------------
# CLI and helpers
# -----------------------------
def load_dataset_and_model(dataset: str, model_path: str, device: str):
    if dataset == 'mnist':
        transform = Compose([ToTensor(), Resize((28, 28))])
        ds = torchvision.datasets.MNIST('../../data', train=False, transform=transform, download=True)
        model = Net(num_classes=10)
        num_classes = 10
    elif dataset == 'gtsrb':
        transform = Compose([ToTensor(), Resize((32, 32))])
        ds = torchvision.datasets.GTSRB('../../data', split='test', transform=transform, download=True)
        if GTSRBNet is None:
            # fallback simple net for GTSRB
            model = Net(num_classes=43)
        else:
            model = GTSRBNet(num_classes=43)
        num_classes = 43
    else:
        raise ValueError('dataset must be mnist or gtsrb')
    sd = torch.load(model_path, map_location='cpu')
    state_dict = sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd
    model.load_state_dict(state_dict)
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    return model, loader, num_classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, required=True, help='mnist|gtsrb')
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--mode', type=str, default='clean', help='clean | synthetic (B3D-SS)')
    p.add_argument('--patch_size', type=int, default=6)
    p.add_argument('--popsize', type=int, default=64)
    p.add_argument('--sigma', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=0.2)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--l1_mask', type=float, default=1.0)
    p.add_argument('--tv_delta', type=float, default=0.05)
    args = p.parse_args()

    model, loader, num_classes = load_dataset_and_model(args.dataset, args.model_path, args.device)
    bb = BlackBoxModel(model, device=args.device, logits=True)
    detector = B3DFull(bbmodel=bb,
                       device=args.device,
                       patch_size=args.patch_size,
                       popsize=args.popsize,
                       sigma=args.sigma,
                       alpha=args.alpha,
                       iters=args.iters,
                       l1_mask=args.l1_mask,
                       tv_delta=args.tv_delta,
                       use_synthetic=(args.mode == 'synthetic'))

    eval_loader = None if args.mode == 'synthetic' else loader
    detected, stats = detector.detect(eval_loader, num_classes)
    print('\nDetection:', 'Backdoor Detected' if detected else 'No Backdoor Detected')
    print(json.dumps(stats, indent=2)[:20000])


if __name__ == '__main__':
    main()
