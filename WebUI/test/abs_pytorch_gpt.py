#!/usr/bin/env python3
# abs_pytorch_gtsrb_reasr.py
# ABS-style scanning for GTSRB with:
#  - class-balanced seed sampling
#  - parallel candidate optimization (multiprocessing)
#  - REASR matrix computation (source x target)
#
# Requirements: pytorch, torchvision. Place this file in the repo that contains GTSRB.py.

import os
import argparse
import time
import json
import random
import tempfile
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import save_image
from tqdm import tqdm  # 添加tqdm导入

# import model from your repo (same usage as detector.py)
from GTSRB import GTSRBNet

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Activation collector
# ----------------------------
class ActivationCollector:
    def __init__(self, model: nn.Module, layer_names=None):
        self.model = model
        self.hooks = []
        self.activations = {}
        if layer_names is None:
            layer_names = [name for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.layer_names = set(layer_names)
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        return hook

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def topk_neurons_from_activations(activations, topk_per_layer=5):
    candidates = []
    for lname, act in activations.items():
        with torch.no_grad():
            if act.dim() >= 3:
                per_channel = act.abs().sum(dim=0)
                while per_channel.dim() > 1:
                    per_channel = per_channel.sum(dim=-1)
            else:
                per_channel = act.abs().sum(dim=0)
            per_channel = per_channel.numpy().flatten()
            k = min(topk_per_layer, len(per_channel))
            topk_idx = np.argsort(-per_channel)[:k]
            for idx in topk_idx:
                candidates.append((lname, int(idx), float(per_channel[idx])))
    candidates.sort(key=lambda x: -x[2])
    return candidates

# ----------------------------
# Optimize trigger for one neuron (same code used by workers)
# ----------------------------
def optimize_trigger_for_neuron_local(state_dict_path: str,
                                      model_class: object,
                                      device_str: str,
                                      seed_images_np: np.ndarray,
                                      layer_name: str,
                                      channel_idx: int,
                                      objective_target: int = None,
                                      iters: int = 500,
                                      lr: float = 0.1,
                                      tv_lambda: float = 0.01,
                                      l1_lambda: float = 0.01,
                                      mask_l1_lambda: float = 0.01,
                                      image_bounds=(0.0, 1.0)):
    """
    This function is designed to be run inside a worker process.
    It loads a fresh model, loads state_dict from disk (state_dict_path),
    moves model to device (cpu or cuda:X) indicated by device_str,
    and runs the optimization returning serializable results.
    """
    # device_str like 'cpu' or 'cuda:0'
    device = torch.device(device_str)
    # build model and load weights
    model = model_class(43).to(device)
    sd = torch.load(state_dict_path, map_location='cpu')
    # accept checkpoint with ['state_dict'] or raw state_dict
    if isinstance(sd, dict) and 'state_dict' in sd:
        model.load_state_dict(sd['state_dict'])
    else:
        model.load_state_dict(sd)
    model.eval()

    # put seed_images into torch on device
    seed_images = torch.from_numpy(seed_images_np).to(device)

    # attach hook
    act = {}
    def hook_fn(m, inp, out):
        act['out'] = out
    if layer_name not in dict(model.named_modules()):
        return {'error': f'layer {layer_name} not found'}
    target_module = dict(model.named_modules())[layer_name]
    hook = target_module.register_forward_hook(hook_fn)

    N, C, H, W = seed_images.shape

    mask_param = (torch.randn(1, C, H, W, device=device) * 0.1).clone().detach().requires_grad_(True)
    delta_param = (torch.randn(1, C, H, W, device=device) * 0.1).clone().detach().requires_grad_(True)
    optimizer = optim.Adam([mask_param, delta_param], lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    best = {'score': -1e9, 'mask': None, 'delta': None}

    # 添加tqdm进度条到优化循环
    for it in tqdm(range(iters), desc=f"Optimizing {layer_name}:{channel_idx}", leave=False):
        optimizer.zero_grad()
        mask = torch.sigmoid(mask_param)
        delta = torch.tanh(delta_param)
        delta_scaled = delta * (image_bounds[1] - image_bounds[0])
        perturbed = torch.clamp(seed_images + mask * delta_scaled, image_bounds[0], image_bounds[1])
        out = model(perturbed)

        activation_tensor = act.get('out', None)
        if activation_tensor is None:
            hook.remove()
            return {'error': 'hook failed'}

        if activation_tensor.dim() >= 3:
            neuron_act = activation_tensor[:, channel_idx:channel_idx+1, ...].mean()
        else:
            neuron_act = activation_tensor[:, channel_idx].mean()

        loss_neuron = -neuron_act
        if objective_target is not None:
            target_tensor = torch.full((N,), objective_target, dtype=torch.long, device=device)
            loss_target = ce_loss(out, target_tensor)
        else:
            loss_target = torch.tensor(0.0, device=device)

        dh = torch.abs(delta_scaled[:, :, 1:, :] - delta_scaled[:, :, :-1, :]).mean()
        dw = torch.abs(delta_scaled[:, :, :, 1:] - delta_scaled[:, :, :, :-1]).mean()
        loss_tv = tv_lambda * (dh + dw)
        loss_l1 = l1_lambda * delta.abs().mean()
        loss_mask = mask_l1_lambda * mask.abs().mean()

        loss = loss_neuron + loss_target + loss_tv + loss_l1 + loss_mask
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            score = float(neuron_act.detach().cpu().item())
            if score > best['score']:
                best['score'] = score
                best['mask'] = mask.detach().cpu().squeeze(0).clone().numpy()   # C,H,W numpy
                best['delta'] = delta_scaled.detach().cpu().squeeze(0).clone().numpy()

    hook.remove()
    # return small dict
    return {
        'layer': layer_name,
        'channel': channel_idx,
        'best_score': float(best['score']),
        'mask_np': best['mask'].astype(np.float32),
        'delta_np': best['delta'].astype(np.float32)
    }

# wrapper to make picklable for Pool.map
def worker_optimize(args_tuple):
    return optimize_trigger_for_neuron_local(*args_tuple)

# ----------------------------
# class-balanced seed sampling
# ----------------------------
def get_class_balanced_seeds(dataset, n_seed, num_classes, transform=None):
    """
    dataset: torchvision dataset of test images (returns (img, label))
    Returns (seed_images_tensor, seed_labels_tensor)
    Strategy:
      - compute per-class quotas = floor(n_seed/num_classes)
      - distribute remainder randomly across classes
      - sample without replacement from each class (if not enough, take all)
    """
    # collect indices per class
    idxs_per_class = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        idxs_per_class[lbl].append(idx)
    base = n_seed // num_classes
    rem = n_seed - base * num_classes
    # shuffle classes for remainder assignment
    classes = list(range(num_classes))
    random.shuffle(classes)
    quotas = [base + (1 if i < rem else 0) for i in range(num_classes)]
    # gather indices
    chosen = []
    for cls in range(num_classes):
        pool = idxs_per_class[cls]
        if len(pool) == 0:
            continue
        random.shuffle(pool)
        take_k = min(quotas[cls], len(pool))
        chosen.extend(pool[:take_k])
    # if not enough due to empty classes, pad by random picks
    if len(chosen) < n_seed:
        all_idx = list(range(len(dataset)))
        random.shuffle(all_idx)
        for i in all_idx:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= n_seed:
                break
    # build batch
    sub = Subset(dataset, chosen)
    loader = DataLoader(sub, batch_size=len(sub), shuffle=False)
    imgs, labels = next(iter(loader))
    return imgs, labels

# ----------------------------
# apply trigger (mask*delta) to images (tensor)
# ----------------------------
def apply_trigger_to_images(images_tensor: torch.Tensor, mask_np: np.ndarray, delta_np: np.ndarray):
    """
    images_tensor: (N,C,H,W), torch (on same device as we do inference)
    mask_np, delta_np: numpy arrays shaped (C,H,W), values in image scale (delta in [-1,1]*scale)
    Operation: images + mask * delta (clamped to [0,1])
    """
    device = images_tensor.device
    mask = torch.from_numpy(mask_np).to(device)
    delta = torch.from_numpy(delta_np).to(device)
    # ensure shapes
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    pert = torch.clamp(images_tensor + mask * delta, 0.0, 1.0)
    return pert

# ----------------------------
# REASR computation
# ----------------------------
def compute_reasr_matrix(model, device, triggers_per_target: dict, test_dataset, max_images_per_class=200):
    """
    triggers_per_target: dict target -> list of {'mask_np','delta_np',...} (topk)
    Returns reasr_matrix shape (num_classes, num_classes) as numpy (source x target)
    For each target t and source s:
      - select up to max_images_per_class images from test_dataset whose label == s
      - apply each trigger (topk) and compute fraction predicted == t
      - average over triggers and images => reasr[s,t]
    """
    num_classes = len(torch.unique(torch.tensor([l for _, l in test_dataset])))
    reasr = np.zeros((num_classes, num_classes), dtype=np.float32)
    # precompute indices per class
    idxs_per_class = [[] for _ in range(num_classes)]
    for idx in range(len(test_dataset)):
        _, lbl = test_dataset[idx]
        idxs_per_class[lbl].append(idx)
    for t, triggers in triggers_per_target.items():
        print(f"Computing REASR for target {t} using {len(triggers)} trigger(s)...")
        for s in range(num_classes):
            idxs = idxs_per_class[s][:max_images_per_class]
            if len(idxs) == 0:
                reasr[s, t] = 0.0
                continue
            sub = Subset(test_dataset, idxs)
            loader = DataLoader(sub, batch_size=min(128, len(sub)), shuffle=False)
            total = 0
            matched = 0
            # for each batch, apply each trigger and count predictions
            for imgs, _ in loader:
                imgs = imgs.to(device)
                batch_N = imgs.shape[0]
                for trig in triggers:
                    pert = apply_trigger_to_images(imgs, trig['mask_np'], trig['delta_np'])
                    with torch.no_grad():
                        logits = model(pert.to(device))
                        preds = torch.argmax(logits, dim=1)
                    matched += (preds == t).sum().item()
                    total += batch_N
            # average over triggers (we already counted per trigger)
            if total == 0:
                reasr[s, t] = 0.0
            else:
                reasr[s, t] = float(matched) / float(total)
    return reasr

# ----------------------------
# Save triggers helper
# ----------------------------
def save_trigger_files(out_dir, target_label, rank, mask_np, delta_np, meta):
    label_dir = os.path.join(out_dir, f"label_{target_label}")
    ensure_dir(label_dir)
    # png
    trig = np.clip(mask_np * delta_np, 0.0, 1.0)  # C,H,W
    # convert to torch and save image
    trig_t = torch.from_numpy(trig).unsqueeze(0).float()  # 1,C,H,W
    png_path = os.path.join(label_dir, f"trigger_{rank}_label{target_label}.png")
    try:
        save_image(trig_t, png_path)
    except Exception as e:
        # fallback: save tensor
        torch.save(trig_t, png_path + ".pt")
    # torch dict
    pt_path = os.path.join(label_dir, f"trigger_{rank}_label{target_label}.pt")
    torch.save({'pattern': torch.from_numpy(trig).float(),
                'mask': torch.from_numpy(mask_np).float(),
                'delta': torch.from_numpy(delta_np).float(),
                'meta': meta}, pt_path)
    # meta json
    with open(os.path.join(label_dir, f"meta_{rank}_label{target_label}.json"), 'w') as f:
        json.dump(meta, f, indent=2)

# ----------------------------
# Main orchestration
# ----------------------------
def main(args):
    set_seed(0)
    ensure_dir(args.out_dir)
    device_main = torch.device('cuda' if torch.cuda.is_available() and args.gpu_device is None else 'cpu')
    # If user specified gpu_device, use that as main device
    if args.gpu_device is not None:
        device_main = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    print("Main device:", device_main)

    # Load model once in main for candidate discovery and REASR inference later.
    model_main = GTSRBNet(43).to(device_main)
    state = torch.load(args.model_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        sd = state['state_dict']
    else:
        sd = state
    model_main.load_state_dict(sd)
    model_main.eval()
    print("Loaded model into main process.")

    # save a temporary copy of the checkpoint for workers to load from disk (avoid pickling huge dicts)
    tmp_ckpt = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    tmp_ckpt.close()
    torch.save(state, tmp_ckpt.name)
    state_dict_path = tmp_ckpt.name

    # load test dataset
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data", "gtsrb")
    transform = Compose([ToTensor(), Resize((32, 32))])
    test_dataset = torchvision.datasets.GTSRB(root=dataset_dir, split='test', transform=transform, download=True)
    num_classes = 43

    # class-balanced seed sampling
    print(f"Sampling {args.n_seed} seeds in class-balanced manner...")
    seed_images, seed_labels = get_class_balanced_seeds(test_dataset, args.n_seed, num_classes)
    print("Seed images shape:", seed_images.shape)

    # compute candidates once using main model and seeds
    print("Collecting activations and computing candidate neurons...")
    collector = ActivationCollector(model_main)
    with torch.no_grad():
        _ = model_main(seed_images.to(device_main))
    activations = collector.activations
    collector.close()
    candidates = topk_neurons_from_activations(activations, topk_per_layer=args.topk_per_layer)
    candidates = candidates[:args.topk_candidates]
    print(f"Selected {len(candidates)} candidates (top {args.topk_candidates}). Sample:")
    for i, (lname, cidx, score) in enumerate(candidates[:10]):
        print(f"  {i+1}. {lname}:{cidx}  score={score:.3f}")

    # Prepare shared seed images ndarray for workers (cpu numpy)
    seed_images_np = seed_images.numpy()

    # Prepare worker device mapping:
    # If multiple GPUs and user didn't force single, distribute workers across GPUs round-robin.
    available_gpus = list(range(torch.cuda.device_count()))
    if args.gpu_device is not None:
        # force all workers to use specified GPU
        available_gpus = [args.gpu_device]
    num_workers = args.num_workers
    if num_workers is None or num_workers < 1:
        num_workers = 1

    # Build args for each worker call: (state_dict_path, model_class, device_str, seed_images_np, layer_name, channel_idx, objective_target, iters, lr, ...)
    # We'll process candidates per target label; for each target label, we will parallelize across candidates.
    triggers_per_target = {}  # dict target -> list of triggers (topk)
    reasr_results = None

    # create Pool once and reuse
    print(f"Launching Pool with {num_workers} worker(s) for optimization (each worker loads model separately).")
    pool = Pool(processes=num_workers)

    try:
        # 为目标标签循环添加进度条
        for target_label in tqdm(range(num_classes), desc="Processing target labels"):
            t0 = time.time()
            print(f"\n=== Target label {target_label} ({target_label+1}/{num_classes}) ===")
            # assemble tasks for this target (one per candidate)
            tasks = []
            for idx, (lname, cidx, init_score) in enumerate(candidates):
                # assign device for worker
                if len(available_gpus) > 0:
                    worker_gpu = available_gpus[idx % len(available_gpus)]
                    device_str = f'cuda:{worker_gpu}'
                else:
                    device_str = 'cpu'
                tasks.append((state_dict_path, GTSRBNet, device_str, seed_images_np, lname, cidx,
                              target_label, args.iters, args.lr, args.tv_lambda, args.l1_lambda, args.mask_l1_lambda))
            # map tasks to pool (worker_optimize wrapper)
            results = pool.map(worker_optimize, tasks)
            # filter out errors and sort by best_score
            valid = [r for r in results if r is not None and 'error' not in r]
            valid.sort(key=lambda x: -x['best_score'])
            print(f"  Completed optimization for target {target_label}: {len(valid)} valid results.")
            # take top-K and save triggers
            topk = min(args.save_topk, len(valid))
            triggers_per_target[target_label] = []
            for rank in range(topk):
                r = valid[rank]
                mask_np = r['mask_np']
                delta_np = r['delta_np']
                meta = {'layer': r['layer'], 'channel': int(r['channel']), 'best_score': float(r['best_score']), 'init_score': None, 'target_label': int(target_label)}
                save_trigger_files(args.out_dir, target_label, rank+1, mask_np, delta_np, meta)
                triggers_per_target[target_label].append({'mask_np': mask_np, 'delta_np': delta_np, 'meta': meta})
                print(f"    Saved trigger rank {rank+1} for target {target_label} (layer={r['layer']} ch={r['channel']}, score={r['best_score']:.4f})")
            elapsed = time.time() - t0
            print(f"  Target {target_label} done in {elapsed:.1f}s. Saved top {topk} triggers.")
        # end for all targets

        # compute REASR matrix using main model (on device_main)
        print("\nComputing REASR matrix (source x target) ...")
        reasr = compute_reasr_matrix(model_main, device_main, triggers_per_target, test_dataset, max_images_per_class=args.max_images_per_class)
        # save reasr matrix as json and numpy
        reasr_path = os.path.join(args.out_dir, "reasr_matrix.npy")
        np.save(reasr_path, reasr)
        with open(os.path.join(args.out_dir, "reasr_matrix.json"), 'w') as f:
            json.dump({'reasr': reasr.tolist()}, f)
        print(f"REASR matrix saved to {reasr_path} and JSON.")

    finally:
        pool.close()
        pool.join()
        # cleanup temp ckpt
        try:
            os.remove(state_dict_path)
        except Exception:
            pass

    print("All done.")

# ----------------------------
# Argparse
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABS PyTorch for GTSRB: class-balanced seeds, parallel opt, REASR matrix")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--n_seed', type=int, default=200, help='number of seed images (class-balanced)')
    parser.add_argument('--topk_per_layer', type=int, default=3)
    parser.add_argument('--topk_candidates', type=int, default=60)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--tv_lambda', type=float, default=0.01)
    parser.add_argument('--l1_lambda', type=float, default=0.01)
    parser.add_argument('--mask_l1_lambda', type=float, default=0.01)
    parser.add_argument('--save_topk', type=int, default=3, help='how many top triggers to save per target')
    parser.add_argument('--out_dir', type=str, default='./abs_reasr_out')
    parser.add_argument('--num_workers', type=int, default=4, help='number of parallel worker processes')
    parser.add_argument('--max_images_per_class', type=int, default=200, help='when computing REASR per source class, limit images per class')
    parser.add_argument('--gpu_device', type=int, default=None, help='if specified, main process uses this GPU index (and workers will use it if available)')
    args = parser.parse_args()
    main(args)
