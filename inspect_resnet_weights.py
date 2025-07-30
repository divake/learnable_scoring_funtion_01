#!/usr/bin/env python3
"""
Inspect the structure of the saved ResNet18 weights
"""

import torch

# Load the checkpoint
checkpoint_path = 'models/resnet18_cifar100_hf.pth'
print(f"Loading checkpoint from: {checkpoint_path}")

state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\nModel state dict keys and shapes:")
print("="*60)

for key, value in state_dict.items():
    if isinstance(value, torch.Tensor):
        print(f"{key:40s} {str(value.shape):20s}")
    else:
        print(f"{key:40s} {type(value)}")
        
# Check first conv layer specifically
if 'conv1.weight' in state_dict:
    conv1_shape = state_dict['conv1.weight'].shape
    print(f"\nFirst conv layer shape: {conv1_shape}")
    print(f"This suggests kernel size: {conv1_shape[2]}x{conv1_shape[3]}")
    print(f"This is designed for CIFAR (small 32x32 images) with 3x3 kernel instead of ImageNet's 7x7")