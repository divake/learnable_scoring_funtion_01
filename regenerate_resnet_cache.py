#!/usr/bin/env python3
"""
Regenerate ResNet cache for CIFAR-100 with the correct model architecture
"""

import torch
import os
import shutil
from datetime import datetime

# First, backup the old cache
cache_dir = "cache/cifar100/ResNet"
if os.path.exists(cache_dir):
    backup_dir = f"{cache_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Backing up old cache to {backup_dir}")
    shutil.move(cache_dir, backup_dir)

# Now run the trainer to regenerate cache
print("Regenerating ResNet cache...")
os.system("/home/divake/miniconda3/envs/env_cu121/bin/python src/core/trainer.py --config src/config/cifar100.yaml --epochs 0 --save_cache_only")
print("Cache regeneration complete!")

# Verify the new cache
if os.path.exists(cache_dir):
    files = os.listdir(cache_dir)
    print(f"\nNew cache files: {files}")
    
    # Load and check one of the files
    cal_probs = torch.load(os.path.join(cache_dir, "cal_probs.pt"))
    print(f"Calibration probabilities shape: {cal_probs.shape}")
    print(f"Max probability: {cal_probs.max().item():.4f}")
    print(f"Min probability: {cal_probs.min().item():.4f}")
else:
    print("ERROR: Cache was not created!")