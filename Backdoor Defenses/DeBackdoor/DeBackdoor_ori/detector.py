import os
import copy
import json
import time
import random
import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from utils import *
from constants import *

class Detector:
    def __init__(self, model, images, attack_type, size_limit, k=4):
        self.model   = model
        self.grid_rescale = 1
        self.k = k
        
        # Initialize trigger
        self.attack_type  = attack_type
        if self.attack_type in ["blend", "lira", "semantic"]:
            # Blended triggger spans entire image
            self.pattern      = np.zeros((dimensions[dataset], resolution[dataset], resolution[dataset]))
            self.top_left     = np.array([0,0])
            self.bottom_right = np.array([resolution[dataset]-1, resolution[dataset]-1])
        elif self.attack_type == "patch":
            # Patch trigger covers continuous subspace within image
            self.pattern      = np.repeat(np.random.randint(2**resolution[dataset], size=(1,resolution[dataset])), dimensions[dataset], axis=0)
            self.top_left = np.random.randint(resolution[dataset], size=2)
            self.size_limit   = size_limit-1 if size_limit else size_limits[dataset]
            self.bottom_right = np.array([np.random.randint(self.top_left[0],
                                                            min(self.top_left[0]+2, resolution[dataset])), 
                                            np.random.randint(self.top_left[1],
                                                            min(self.top_left[1]+2, resolution[dataset]))])
        # Warped trigger uses the control-grid of warping field    
        elif self.attack_type == "wanet":
            self.pattern = torch.rand(1, 2, self.k, self.k) * 2 - 1
            self.top_left = np.array([0,0])
            self.bottom_right = np.array([self.k-1, self.k-1])
        # Filter trigger uses the parameters of the gamma correction operation
        elif self.attack_type == "filter":
            self.pattern = torch.tensor([1]*dimensions[dataset]).view(1, dimensions[dataset], 1, 1).to(device)
            self.top_left = np.array([0,0])
            self.bottom_right = np.array([resolution[dataset]-1, resolution[dataset]-1])
        else:
            raise ValueError(f"{self.attack_type} attack not supported")

        # Scoring items
        self.soft         = nn.Softmax(dim=1)
        self.clean_images = images
        self.lambd        = -1
        self.score        = -1
        self.target_label = -1

    # Backdoor trigger generation using simulated annealing
    def optimize(self, target_label, num_rounds, lambd):
        # Initialize progress bar
        show_progress_bar(0, num_rounds, prefix="Progress:", suffix="Complete", length=50)
        
        # Pole controls the temperature cooling schedule
        pole              = num_rounds/50
        self.target_label = int(target_label)
        self.lambd        = lambd
        max_score         = -1
        max_pattern       = None
        max_top_left      = None
        max_bottom_right  = None

        for round_num in range(num_rounds):
            # Temperature decreases as more rounds pass
            temperature = pole/(round_num+pole)-pole/(num_rounds+pole)

            if round_num%2 == 0 or self.attack_type in ["blend", "wanet", "lira", "filter", "semantic"]:
                self.optimize_pattern(temperature, target_label)
            else:
                # Patch attack optimizes mask on odd rounds
                self.optimize_mask(temperature, target_label)

            # Update highest scoring trigger and progress bar
            if self.score > max_score:
                max_score        = self.score
                max_pattern      = self.pattern
                max_top_left     = self.top_left
                max_bottom_right = self.bottom_right

            show_progress_bar(round_num+1, num_rounds, prefix="Progress:", suffix=f"Complete (ASR={max_score:.4f})", length=50)

        result = {
            "score"       : float(max_score),
            "top_left"    : max_top_left.tolist(),
            "bottom_right": max_bottom_right.tolist(),
            "pattern"     : [channel.tolist() for channel in max_pattern]
        }
        
        return result

    # Image manipulation modules
    def generate_pattern_image(self, pattern):
        if self.attack_type == "patch":
            trigger = []
            
            for channel in pattern:
                channel_pixels = []

                for row_id in channel:
                    # Pixels in each row of image are represented using 1s/0s in 28/32-bit integer
                    format_str = "{0:032b}" if resolution[dataset] == 32 else "{0:028b}"
                    row_pixels = np.fromstring(" ".join(format_str.format(row_id)), sep=" ")
                    channel_pixels.append(row_pixels)
                trigger.append(np.stack(channel_pixels))
            trigger = torch.from_numpy(np.stack(trigger)).type(torch.FloatTensor).to(device)
        else:
            trigger = torch.from_numpy(self.pattern).to(device)
        return trigger

    def flip_neighbor_pixels(self, coords):
        neighbor_pattern = copy.deepcopy(self.pattern)

        if self.attack_type in ["blend", "lira", "semantic"]:
            # Modify pixels across the entire image
            bound = 1 if self.attack_type in ["blend", "semantic"] else 0.1
            delta = np.random.uniform(-bound, bound, size=(dimensions[dataset], resolution[dataset], resolution[dataset]))
            neighbor_pattern += delta
            neighbor_pattern = np.clip(neighbor_pattern, 0 if self.attack_type in ["blend", "semantic"] else -0.1, 1 if self.attack_type in ["blend", "semantic"] else 0.1)
        elif self.attack_type == "filter":
            channel, _, _ = coords[0]
            neighbor_pattern[0][channel][0][0] = np.random.choice(np.arange(0.5,4,0.5))
        else:
            if self.attack_type == "patch":
                _,row,col = coords[0]
                
                for channel in range(dimensions[dataset]):
                    coords.append((channel,row,col))
            
            for channel,row,col in coords:
                if self.attack_type == "wanet":
                    # Modify entry in the control grid
                    bound = 0.3
                    delta = (-bound - bound) * torch.rand(1) + bound
                    neighbor_pattern[0][channel][row][col] += delta[0]
                else:
                    # Flip single pixel at given coordinates
                    neighbor_pattern[channel][row] = self.pattern[channel][row]^2**(resolution[dataset]-1-col)

        return neighbor_pattern

    # Continuous attack success rate (cASR) modules
    def apply_backdoor(self, pattern, top_left, bottom_right):
        if self.attack_type == "wanet":
            # Trigger warps across the entire image
            noise_grid = (
                F.upsample(pattern, size=resolution[dataset], mode="bicubic", align_corners=True)
                .permute(0, 2, 3, 1)
                .to(device)
            )
            array1d = torch.linspace(-1, 1, steps=resolution[dataset])
            x, y = torch.meshgrid(array1d, array1d)
            identity_grid = torch.stack((y, x), 2)[None, ...].to(device)

            # Generate trigger
            grid_temps = (identity_grid + noise_grid / resolution[dataset]) * self.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)
            
            # Insert trigger
            backdoor_images = F.grid_sample(self.clean_images, grid_temps.repeat(self.clean_images.shape[0], 1, 1, 1), align_corners=True)
        elif self.attack_type == "filter":
            backdoor_images = self.clean_images ** self.pattern
        else:
            # Trigger overwrites image content for patch attacks, but is blended into the image for other attacks
            alpha         = alphas[dataset] if self.attack_type in ["blend", "lira", "filter", "semantic"] else 1
            backdoor_images = copy.deepcopy(self.clean_images)

            # Generate trigger
            pattern_image = self.generate_pattern_image(pattern)
            r1,c1         = top_left
            r2,c2         = bottom_right
            
            # Insert trigger
            if self.attack_type == "lira":
                backdoor_images[:,:,r1:r2+1,c1:c2+1] = backdoor_images[:,:,r1:r2+1,c1:c2+1] + pattern_image[:,r1:r2+1,c1:c2+1]
            else:
                if dataset == "imagenet":
                    # Upscale trigger for 224x224 ImageNet images
                    resize = Resize((224, 224))
                    pattern_image = pattern_image.unsqueeze(0)
                    pattern_image = resize(pattern_image)
                    pattern_image = pattern_image.squeeze(0)
                    r1,c1 = 0,0
                    r2,c2 = 223,223
                
                backdoor_images[:,:,r1:r2+1,c1:c2+1] = (1-alpha)*backdoor_images[:,:,r1:r2+1,c1:c2+1] + alpha*pattern_image[:,r1:r2+1,c1:c2+1]
        return backdoor_images

    def compute_score(self, backdoor_images, target_label):
        # Get model output and isolate output for target label
        scores                 = self.soft(self.model(backdoor_images))
        target_label_scores    = scores[:,target_label].detach().clone()

        # Compute gap between target label output and output of highest scoring label amongst other labels
        scores[:,target_label] = 0.00001
        example_max_scores     = scores.max(axis=1)[0]
        deltas                 = target_label_scores.log()-example_max_scores.log()

        proxy_1 =   ( self.lambd*deltas).exp()/2
        proxy_2 = 1-(-self.lambd*deltas).exp()/2
        proxy   = proxy_1*(deltas<0)+proxy_2*(deltas>=0)

        return proxy.mean()

    # Search parameter optimization modules 
    def optimize_pattern(self, temperature, target_label):
        # Flip random color channel/location and compute new score
        channel           = np.random.randint(2 if self.attack_type == "wanet" else dimensions[dataset])
        row               = np.random.randint(self.top_left[0], self.bottom_right[0]+1)
        col               = np.random.randint(self.top_left[1], self.bottom_right[1]+1)
        neighbor_pattern  = self.flip_neighbor_pixels([(channel,row,col)])
        neighbor_images   = self.apply_backdoor(neighbor_pattern, self.top_left, self.bottom_right)
        neighbor_score    = self.compute_score(neighbor_images, target_label)
        
        # Update if new score is higher, or randomly with probability dictated by temperature
        if neighbor_score > self.score or np.random.random() < temperature:
            self.pattern = neighbor_pattern
            self.score   = neighbor_score

    def optimize_mask(self, temperature, target_label):
        # Calculate leeway for moving the patch
        space_above = self.top_left[0]
        space_right = resolution[dataset]-1-self.bottom_right[1]
        space_below = resolution[dataset]-1-self.bottom_right[0]
        space_left  = self.top_left[1]
        
        # Generate possible reshapes (extend/contract by 1 row up/down or 1 column left/right)
        deltas = []
        if self.top_left[0] > 0 and self.bottom_right[0]-self.top_left[0]<self.size_limit:
            deltas.extend([((-1, 0), ( 0, 0))])
        if self.top_left[0] < resolution[dataset]-1 and self.top_left[0] < self.bottom_right[0]:
            deltas.extend([(( 1, 0), ( 0, 0))])
        if self.top_left[1] > 0 and self.bottom_right[1]-self.top_left[1]<self.size_limit:
            deltas.extend([(( 0,-1), ( 0, 0))])
        if self.top_left[1] < resolution[dataset]-1 and self.top_left[1] < self.bottom_right[1]:
            deltas.extend([(( 0, 1), ( 0, 0))])
        if self.bottom_right[0] > 0 and self.bottom_right[0] > self.top_left[0]:
            deltas.extend([(( 0, 0), (-1, 0,))])
        if self.bottom_right[0] < resolution[dataset]-1 and self.bottom_right[0]-self.top_left[0]<self.size_limit:
            deltas.extend([(( 0, 0), ( 1, 0,))])
        if self.bottom_right[1] > 0 and self.bottom_right[1] > self.top_left[1]:
            deltas.extend([(( 0, 0), ( 0,-1))])
        if self.bottom_right[1] < resolution[dataset]-1 and self.bottom_right[1]-self.top_left[1]<self.size_limit:
            deltas.extend([(( 0, 0), ( 0, 1))])
            
        # Generate possible moves to new locations for the patch
        relocations = []
        for v_delta in np.concatenate((-np.arange(1,space_above+1),np.arange(1,space_below+1))):
            for h_delta in np.concatenate((-np.arange(1,space_left+1),np.arange(1,space_right+1))):
                relocations.append((v_delta,h_delta))
        deltas.extend([(delta,delta) for delta in relocations])

        # Apply randomly selected reshape/relocation and compute new score
        delta_top_left, delta_bottom_right = random.sample(deltas, 1)[0]
        neighbor_top_left     = self.top_left    +delta_top_left
        neighbor_bottom_right = self.bottom_right+delta_bottom_right    
        neighbor_mask         = (neighbor_top_left, neighbor_bottom_right) 
        neighbor_images       = self.apply_backdoor(self.pattern, neighbor_top_left, neighbor_bottom_right)
        neighbor_score        = self.compute_score(neighbor_images, target_label)

        # Update if new score is higher, or randomly with probability dictated by temperature        
        if neighbor_score > self.score or np.random.random() < temperature:
            self.top_left, self.bottom_right = neighbor_mask
            self.score                       = neighbor_score


if __name__ == "__main__":
    # Parse command-line arguments (e.g. python detector.py -m examples/patch/mnist/model.pt -i examples/images/mnist.pt -n 30000 -l 0.9 -a patch -s 6)
    parser = argparse.ArgumentParser(description ="Detect backdoor triggers in deep models")
    parser.add_argument("-m", "--model_path"  , metavar="model_path"  , type=str  , required=True , help="path to model to be inspected")
    parser.add_argument("-i", "--image_path"  , metavar="image_path"  , type=str  , required=True , help="path to batch of clean validation images")
    parser.add_argument("-n", "--num_rounds"  , metavar="num_rounds"  , type=int  , required=True , help="number of optimization rounds")
    parser.add_argument("-l", "--lambd"       , metavar="lambda"      , type=float, required=True , help="score smoothing parameter lambda")
    parser.add_argument("-a", "--attack_type" , metavar="attack_type" , type=str  , required=True , help="attack type (patch/blend/wanet/lira/filter/semantic)", choices=["patch", "blend", "wanet", "lira", "filter", "semantic"])
    parser.add_argument("-s", "--size_limit"  , metavar="size-limit"  , type=int  , required=False, help="width/height limit of patch")
    parser.add_argument("-g", "--gpu_id"      , metavar="gpu_id"      , type=int  , required=False, help="id of gpu to be used for training")
    args = parser.parse_args()

    # Load inspection pre-requisites
    device = torch.device(f"cpu")
    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
    model   = torch.load(args.model_path, weights_only=False).to(device).eval()
    images  = torch.load(args.image_path, weights_only=False).to(device)
    dataset = args.image_path.split("/")[-1].split(".")[0].split("-")[0].split("_")[0]
    results_path, _ = os.path.split(args.model_path)
    label_results = {}

    for label in range(num_labels[dataset]):
        print(f"Optimizing {args.attack_type} attack for target label {label}...")
        st = time.time()
        detector = Detector(model=model, images=images, attack_type=args.attack_type, size_limit=args.size_limit)
        result = detector.optimize(target_label=label, num_rounds=args.num_rounds, lambd=args.lambd)
        
        score  = result["score"]
        runtime = time.time()-st
        result["runtime"] = runtime
        print(f"\nHighest cASR = {score:.4f}")
        print(f"Total runtime = {runtime:.2f}s\n")
        label_results[label] = result
    
    # Store results in same directory as model
    name = os.path.basename(args.model_path).replace("model", "result").split(".")[0]
    with open(os.path.join(results_path, f"{name}_{args.attack_type}.json"), "w") as f:
        json.dump(label_results, f)
