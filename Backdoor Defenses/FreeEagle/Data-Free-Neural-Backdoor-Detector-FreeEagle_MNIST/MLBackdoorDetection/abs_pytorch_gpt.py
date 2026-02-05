#!/usr/bin/env python3
# abs_pytorch_gtsrb_reasr.py
# ABS-style scanning for GTSRB or MNIST with:
#  - class-balanced seed sampling
#  - parallel candidate optimization (multiprocessing)
#  - REASR matrix computation (source x target)

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
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.utils import save_image
from model import Model

# ----------------------------
# Models
# ----------------------------
try:
    from GTSRB import GTSRBNet  # from repo
except ImportError:
    GTSRBNet = None


# LeNet Model definition
class Net(Model):#创建网络
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Activation collector
# ----------------------------
class ActivationCollector:
    def __init__(self, model, layer_names=None):
        self.hooks, self.activations = [], {}
        if layer_names is None:
            layer_names = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        self.layer_names = set(layer_names)
        for name, module in model.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu()
        return fn

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def topk_neurons_from_activations(activations, topk_per_layer=5):
    candidates = []
    for lname, act in activations.items():
        if act.dim() >= 3:
            per_channel = act.abs().sum(dim=0)
            while per_channel.dim() > 1:
                per_channel = per_channel.sum(dim=-1)
        else:
            per_channel = act.abs().sum(dim=0)
        arr = per_channel.numpy().flatten()
        k = min(topk_per_layer, len(arr))
        topk_idx = np.argsort(-arr)[:k]
        for idx in topk_idx:
            candidates.append((lname, int(idx), float(arr[idx])))
    candidates.sort(key=lambda x: -x[2])
    return candidates

# ----------------------------
# Optimize trigger (worker)
# ----------------------------
def optimize_trigger_for_neuron_local(state_dict_path, model_class, num_classes,
                                      device_str, seed_images_np,
                                      layer_name, channel_idx,
                                      objective_target=None,
                                      iters=500, lr=0.1,
                                      tv_lambda=0.01, l1_lambda=0.01, mask_l1_lambda=0.01,
                                      image_bounds=(0.0, 1.0)):

    device = torch.device(device_str)
    model = model_class(num_classes).to(device)
    sd = torch.load(state_dict_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"])
    else:
        model.load_state_dict(sd)
    model.eval()

    seed_images = torch.from_numpy(seed_images_np).to(device)
    act = {}
    def hook(m, i, o): act["out"] = o
    dict(model.named_modules())[layer_name].register_forward_hook(hook)

    N, C, H, W = seed_images.shape
    mask_param = (torch.randn(1, C, H, W, device=device) * 0.1).clone().detach().requires_grad_(True)
    delta_param = (torch.randn(1, C, H, W, device=device) * 0.1).clone().detach().requires_grad_(True)
    optimizer = optim.Adam([mask_param, delta_param], lr=lr)
    ce_loss = nn.CrossEntropyLoss()

    best = {"score": -1e9, "mask": None, "delta": None}
    for it in range(iters):
        optimizer.zero_grad()
        mask = torch.sigmoid(mask_param)
        delta = torch.tanh(delta_param)
        delta_scaled = delta * (image_bounds[1] - image_bounds[0])
        perturbed = torch.clamp(seed_images + mask * delta_scaled, *image_bounds)
        out = model(perturbed)

        activation_tensor = act["out"]
        if activation_tensor.dim() >= 3:
            neuron_act = activation_tensor[:, channel_idx:channel_idx+1].mean()
        else:
            neuron_act = activation_tensor[:, channel_idx].mean()
        loss = -neuron_act
        if objective_target is not None:
            target = torch.full((N,), objective_target, dtype=torch.long, device=device)
            loss += ce_loss(out, target)
        loss += tv_lambda * (torch.abs(delta_scaled[:,:,1:]-delta_scaled[:,:,:-1]).mean() +
                             torch.abs(delta_scaled[:,:,:,1:]-delta_scaled[:,:,:,:-1]).mean())
        loss += l1_lambda * delta.abs().mean()
        loss += mask_l1_lambda * mask.abs().mean()
        loss.backward()
        optimizer.step()

        if neuron_act.item() > best["score"]:
            best.update({"score": float(neuron_act.item()),
                         "mask": mask.detach().cpu().squeeze(0).numpy(),
                         "delta": delta_scaled.detach().cpu().squeeze(0).numpy()})
    return {"layer": layer_name, "channel": channel_idx, "best_score": best["score"],
            "mask_np": best["mask"], "delta_np": best["delta"]}

def worker_optimize(args): return optimize_trigger_for_neuron_local(*args)

# ----------------------------
# Balanced seeds
# ----------------------------
def get_class_balanced_seeds(dataset, n_seed, num_classes):
    idxs_per_class = [[] for _ in range(num_classes)]
    for idx in range(len(dataset)):
        _, lbl = dataset[idx]
        idxs_per_class[lbl].append(idx)
    base, rem = n_seed // num_classes, n_seed % num_classes
    classes = list(range(num_classes)); random.shuffle(classes)
    quotas = [base + (1 if i < rem else 0) for i in range(num_classes)]
    chosen = []
    for cls in range(num_classes):
        pool = idxs_per_class[cls]; random.shuffle(pool)
        chosen.extend(pool[:quotas[cls]])
    sub = Subset(dataset, chosen)
    imgs, labels = next(iter(DataLoader(sub, batch_size=len(sub))))
    return imgs, labels

# ----------------------------
# Apply trigger
# ----------------------------
def apply_trigger_to_images(imgs, mask_np, delta_np):
    device = imgs.device
    mask = torch.from_numpy(mask_np).to(device).unsqueeze(0)
    delta = torch.from_numpy(delta_np).to(device).unsqueeze(0)
    return torch.clamp(imgs + mask * delta, 0, 1)

# ----------------------------
# REASR
# ----------------------------
def compute_reasr_matrix(model, device, triggers_per_target, dataset, num_classes, max_imgs=200):
    reasr = np.zeros((num_classes,num_classes),dtype=np.float32)
    idxs_per_class = [[] for _ in range(num_classes)]
    for i in range(len(dataset)): _, l = dataset[i]; idxs_per_class[l].append(i)
    for t,trigs in triggers_per_target.items():
        for s in range(num_classes):
            idxs = idxs_per_class[s][:max_imgs]; 
            if not idxs: continue
            sub = Subset(dataset, idxs)
            total, matched = 0,0
            for imgs,_ in DataLoader(sub,batch_size=64):
                imgs = imgs.to(device)
                for trig in trigs:
                    pert = apply_trigger_to_images(imgs,trig['mask_np'],trig['delta_np'])
                    preds = model(pert).argmax(1)
                    matched += (preds==t).sum().item()
                    total += len(preds)
            reasr[s,t] = matched/total if total>0 else 0
    return reasr

# ----------------------------
# Save triggers
# ----------------------------
def save_trigger(out_dir,target,rank,mask,delta,meta):
    d = os.path.join(out_dir,f"label_{target}"); ensure_dir(d)
    trig = np.clip(mask*delta,0,1)
    save_image(torch.from_numpy(trig).unsqueeze(0).float(), os.path.join(d,f"trigger_{rank}_label{target}.png"))
    torch.save({'mask':torch.from_numpy(mask),'delta':torch.from_numpy(delta),'meta':meta},
               os.path.join(d,f"trigger_{rank}_label{target}.pt"))
    
    # 创建一个可序列化的meta副本，将ndarray转换为列表
    serializable_meta = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            serializable_meta[key] = value.tolist()
        else:
            serializable_meta[key] = value
            
    with open(os.path.join(d,f"meta_{rank}_label{target}.json"),'w') as f: 
        json.dump(serializable_meta,f,indent=2)

# ----------------------------
# Main
# ----------------------------
def main(args):
    set_seed(0); ensure_dir(args.out_dir)
    if args.dataset=="gtsrb":
        num_classes, model_class = 43, GTSRBNet
        dataset = torchvision.datasets.GTSRB("../../data/gtsrb",split='test',transform=Compose([ToTensor(),Resize((32,32))]),download=True)
    elif args.dataset=="mnist":
        num_classes, model_class = 10, Net
        dataset = torchvision.datasets.MNIST("../../data/mnist",train=False,transform=Compose([ToTensor(),Resize((28,28))]),download=True)
    else: raise ValueError("dataset must be gtsrb|mnist")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(num_classes).to(device)
    sd = torch.load(args.model_path,map_location="cpu")
    model.load_state_dict(sd["state_dict"] if isinstance(sd,dict) and "state_dict" in sd else sd)
    model.eval()

    seeds,_ = get_class_balanced_seeds(dataset,args.n_seed,num_classes)
    print("Seeds:",seeds.shape)
    collector = ActivationCollector(model)
    with torch.no_grad(): _ = model(seeds.to(device))
    cands = topk_neurons_from_activations(collector.activations,args.topk_per_layer)[:args.topk_candidates]
    print("Candidates:",len(cands))

    tmp_ckpt = tempfile.NamedTemporaryFile(delete=False,suffix=".pt"); tmp_ckpt.close()
    torch.save(sd,tmp_ckpt.name)

    pool = Pool(processes=args.num_workers)
    triggers_per_target = {}
    for t in range(num_classes):
        tasks=[]
        for i,(lname,cidx,_) in enumerate(cands):
            dev="cuda" if torch.cuda.is_available() else "cpu"
            tasks.append((tmp_ckpt.name,model_class,num_classes,dev,seeds.numpy(),lname,cidx,t,args.iters,args.lr,args.tv_lambda,args.l1_lambda,args.mask_l1_lambda))
        res=pool.map(worker_optimize,tasks)
        res=[r for r in res if "error" not in r]; res.sort(key=lambda x:-x['best_score'])
        triggers_per_target[t]=[]
        for rank,r in enumerate(res[:args.save_topk]):
            save_trigger(args.out_dir,t,rank+1,r["mask_np"],r["delta_np"],r)
            triggers_per_target[t].append(r)

    reasr = compute_reasr_matrix(model,device,triggers_per_target,dataset,num_classes,args.max_images_per_class)
    np.save(os.path.join(args.out_dir,"reasr_matrix.npy"),reasr)
    with open(os.path.join(args.out_dir,"reasr_matrix.json"),"w") as f: json.dump(reasr.tolist(),f)

    pool.close(); pool.join(); os.remove(tmp_ckpt.name)

# ----------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=str,default="gtsrb",help="gtsrb|mnist")
    p.add_argument("--model_path",type=str,required=True)
    p.add_argument("--n_seed",type=int,default=200)
    p.add_argument("--topk_per_layer",type=int,default=3)
    p.add_argument("--topk_candidates",type=int,default=60)
    p.add_argument("--iters",type=int,default=500)
    p.add_argument("--lr",type=float,default=0.1)
    p.add_argument("--tv_lambda",type=float,default=0.01)
    p.add_argument("--l1_lambda",type=float,default=0.01)
    p.add_argument("--mask_l1_lambda",type=float,default=0.01)
    p.add_argument("--save_topk",type=int,default=3)
    p.add_argument("--out_dir",type=str,default="./abs_out")
    p.add_argument("--num_workers",type=int,default=1)
    p.add_argument("--max_images_per_class",type=int,default=200)
    args=p.parse_args(); main(args)
