#!/usr/bin/env python
"""Build cache for PlantNet dataset to speed up evaluation"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import logging
from tqdm import tqdm

# Add src to path
sys.path.insert(0, '.')

from src.datasets.plantnet import Dataset
import yaml

logging.basicConfig(level=logging.INFO)

def build_cache():
    # Load config
    with open('src/config/base_conformal_scorers.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Add PlantNet-specific config
    config = base_config.copy()
    config['dataset'] = base_config['plantnet'].copy()
    config['dataset']['name'] = 'plantnet'
    config['model'] = base_config['plantnet'].get('model', {})
    config['base_dir'] = '.'
    config['device'] = 'cuda:1'
    config['batch_size'] = config['plantnet'].get('batch_size', 128)
    
    # Initialize dataset
    logging.info("Initializing PlantNet dataset...")
    dataset = Dataset(config)
    dataset.setup()
    
    # Get model
    logging.info("Loading PlantNet model...")
    device = torch.device(config['device'])
    model = dataset.get_model().to(device)
    model.eval()
    
    # Cache directory
    cache_dir = 'cache/plantnet/VisionTransformer'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Process calibration data
    logging.info("Processing calibration data...")
    cal_probs = []
    cal_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataset.cal_loader, desc="Calibration"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            cal_probs.append(probs.cpu())
            cal_targets.append(targets)
    
    cal_probs = torch.cat(cal_probs, dim=0)
    cal_targets = torch.cat(cal_targets, dim=0)
    
    # Process test data
    logging.info("Processing test data...")
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataset.test_loader, desc="Test"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            test_probs.append(probs.cpu())
            test_targets.append(targets)
    
    test_probs = torch.cat(test_probs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    
    # Save cache
    cache_file = os.path.join(cache_dir, 'model_outputs.pth')
    logging.info(f"Saving cache to {cache_file}...")
    torch.save({
        'cal_probs': cal_probs,
        'cal_targets': cal_targets,
        'test_probs': test_probs,
        'test_targets': test_targets,
    }, cache_file)
    
    logging.info(f"Cache built successfully!")
    logging.info(f"Calibration: {len(cal_targets)} samples")
    logging.info(f"Test: {len(test_targets)} samples")

if __name__ == "__main__":
    build_cache()