#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, Subset
from model import Model

# Import Net model from abs_pytorch_gpt.py
# 我将修改Net类的forward方法，将view()替换为reshape()
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
        # 修复：将view()替换为reshape()以处理内存不连续问题
        x = x.reshape(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out

# PyTorch to TensorFlow model wrapper
class PyTorchModelWrapper:
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()
        self.device = next(pytorch_model.parameters()).device
    
    def predict(self, x):
        # Ensure input is numpy array
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        
        # Convert to (N, C, H, W) format if needed
        if len(x_np.shape) == 4 and x_np.shape[3] == 1:  # (N, H, W, 1)
            x_np = np.transpose(x_np, (0, 3, 1, 2))
        
        # Convert to PyTorch tensor and move to device
        x_tensor = torch.from_numpy(x_np).float().to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.pytorch_model(x_tensor)
            # Convert to numpy array with softmax probabilities
            probabilities = torch.exp(output).cpu().numpy()
        
        return probabilities

# Decision function that supports both TensorFlow and PyTorch
def decision_function(model, data, framework='tensorflow'):
    if framework == 'tensorflow':
        import tensorflow.keras as keras
        a = np.argmax(model.predict(data), axis=1)
    else:  # PyTorch
        # Ensure data is in the correct format for PyTorch
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
            # PyTorch expects (batch_size, channels, height, width)
            # If data is in (batch_size, height, width, channels), transpose it
            if len(data.shape) == 4 and data.shape[3] == 1:  # (N, H, W, 1)
                data = data.permute(0, 3, 1, 2)
        # Move data to the same device as the model
        device = next(model.parameters()).device
        data = data.to(device)
        # Make predictions with PyTorch model
        with torch.no_grad():
            out = model(data)
            a = torch.argmax(out, dim=1).cpu().numpy()
    print(a.shape)
    return a

class BD_detect:
    def __init__(self, args, task='cifar10'):
        self.task = task
        self.args = args
        self.framework = args.framework
        
        # Set up device
        if self.framework == 'pytorch':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.load_model()
        
        # Load data
        self.load_data()
        
        # Create output directory
        if task == 'cifar10':
            self.dict = 'cifar10_adv_per'
        elif task == 'cifar100':
            self.dict = 'cifar100_adv_per'
        elif task == 'mnist':
            self.dict = 'mnist_adv_per'
        
        if not os.path.exists(self.dict):
            os.mkdir(self.dict)
        
        # Filter correctly classified samples
        self.filter_correct_samples()
    
    def load_model(self):
        if self.framework == 'tensorflow':
            import tensorflow.keras as keras
            if self.task == 'cifar10':
                self.model = keras.models.load_model("saved_models/cifar10_backdoor.h5")
            elif self.task == 'cifar100':
                self.model = keras.models.load_model("saved_models/cifar100_backdoor.h5")
        else:  # PyTorch
            if self.task == 'mnist':
                self.model = Net(num_classes=10).to(self.device)
                # Load PyTorch model weights
                sd = torch.load(self.args.model_path, map_location=self.device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    self.model.load_state_dict(sd["state_dict"])
                else:
                    self.model.load_state_dict(sd)
                self.model.eval()
    
    def load_data(self):
        if self.framework == 'tensorflow':
            from GradEst.load_data import ImageData
            img_data = ImageData(dataset_name=self.task)
            self.x_val = img_data.x_val
            self.y_val = img_data.y_val.reshape(img_data.y_val.shape[0])
            del img_data
        else:  # PyTorch
            if self.task == 'mnist':
                # Use torchvision to load MNIST test set
                dataset = torchvision.datasets.MNIST("../../data/mnist", train=False, 
                                                   transform=Compose([ToTensor(), Resize((28, 28))]), 
                                                   download=True)
                # Convert to numpy array format to maintain compatibility with existing code
                loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
                imgs, labels = next(iter(loader))
                self.x_val = imgs.numpy()
                # For compatibility with existing code that expects (N, H, W, C)
                if len(self.x_val.shape) == 4 and self.x_val.shape[1] == 1:  # (N, 1, H, W)
                    self.x_val = np.transpose(self.x_val, (0, 2, 3, 1))
                self.y_val = labels.numpy().reshape(-1)
    
    def filter_correct_samples(self):
        # Filter samples that the model correctly classifies
        correct_idx = decision_function(self.model, self.x_val, self.framework) == self.y_val
        self.x_val = self.x_val[correct_idx]
        print(f"Accuracy: {self.x_val.shape[0]/len(correct_idx)}")
        self.y_val = self.y_val[correct_idx]
        assert self.y_val.shape[0] == self.x_val.shape[0], "Sample count mismatch"
    
    def get_vec(self, original_label, target_label):
        if os.path.exists(f"{self.dict}/data_{str(original_label)}_{str(target_label)}.npy"):
            pass
        else:
            # Get samples for original and target labels
            x_o = self.x_val[self.y_val == original_label][0:40] if len(self.x_val[self.y_val == original_label]) >= 40 else self.x_val[self.y_val == original_label]
            x_t = self.x_val[self.y_val == target_label][0:40] if len(self.x_val[self.y_val == target_label]) >= 40 else self.x_val[self.y_val == target_label]
            y_t = self.y_val[self.y_val == target_label][0:40] if len(self.y_val[self.y_val == target_label]) >= 40 else self.y_val[self.y_val == target_label]
            
            # Import attack module
            from GradEst.main import attack
            
            # Use appropriate model for attack
            if self.framework == 'pytorch':
                # Create a TensorFlow-compatible wrapper for PyTorch model
                wrapped_model = PyTorchModelWrapper(self.model)
                model_for_attack = wrapped_model
            else:
                model_for_attack = self.model
            
            # Perform attack without the framework parameter
            dist, per = attack(model_for_attack, x_o, x_t, y_t)
            
            # Save results
            np.save(f"{self.dict}/data_{str(original_label)}_{str(target_label)}.npy", per)
    
    def detect(self):
        # Determine number of labels
        if self.task == 'cifar10' or self.task == 'mnist':
            num_labels = 10
        elif self.task == 'cifar100':
            num_labels = 100
        
        # Iterate over label range
        for i in range(self.args.sp, self.args.ep):
            labels = list(range(num_labels))
            labels.remove(i)
            
            assert len(labels) == (num_labels - 1), "Label count error"
            
            # Perform detection for each target label
            for index, t in enumerate(labels):
                print(f"original: {i} -> {t} \n")
                self.get_vec(original_label=i, target_label=t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument('--task', type=str,
                        choices=['cifar10', 'cifar100', 'mnist'],
                        default='cifar10')
    
    # Framework selection
    parser.add_argument('--framework', type=str,
                        choices=['tensorflow', 'pytorch'],
                        default='tensorflow',
                        help='Select deep learning framework')
    
    # PyTorch model path
    parser.add_argument('--model_path', type=str,
                        default='',
                        help='Path to PyTorch model weights file')
    
    # Original parameters
    parser.add_argument('--sp', type=int, default=0,
                        help='Starting label index')
    
    parser.add_argument('--ep', type=int, default=10,
                        help='Ending label index')
    
    parser.add_argument('--cuda', type=str, default='0',
                        help='Specify GPU device to use')
    
    parser.add_argument('--cou', type=bool, default=False, 
                        help='Whether to use cou training')
    
    args = parser.parse_args()
    
    # Set CUDA environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    # Check parameter validity
    if args.framework == 'pytorch' and args.task == 'mnist' and not args.model_path:
        raise ValueError("Must provide model_path when using PyTorch with MNIST")
    
    # Create detector instance and perform detection
    bd = BD_detect(args=args, task=args.task)
    bd.detect()



