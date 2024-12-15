# verify_splits.py
"""
This script verifies that the CIFAR-10 splits remain consistent
across multiple runs by checking indices and actual images.
"""

import torch
from src.cifar_split import setup_cifar10
import numpy as np

def get_dataset_info(train_dataset, cal_dataset):
    """
    Extract important information from datasets for comparison
    """
    # Get indices
    train_indices = train_dataset.indices
    cal_indices = cal_dataset.indices
    
    # Get first few images and labels as fingerprint
    train_samples = [train_dataset[i][1] for i in range(5)]  # Get first 5 labels
    cal_samples = [cal_dataset[i][1] for i in range(5)]  # Get first 5 labels
    
    return {
        'train_indices_sum': sum(train_indices),  # Checksum of indices
        'cal_indices_sum': sum(cal_indices),
        'train_indices_len': len(train_indices),
        'cal_indices_len': len(cal_indices),
        'train_first_samples': train_samples,
        'cal_first_samples': cal_samples
    }

def verify_consistency(num_runs=3):
    """
    Load the datasets multiple times and verify consistency
    """
    print("Verifying split consistency across multiple runs...")
    print(f"Performing {num_runs} separate loads of the dataset...")
    
    results = []
    for run in range(num_runs):
        print(f"\nRun {run + 1}:")
        # Load datasets
        _, _, _, train_dataset, cal_dataset, _ = setup_cifar10()
        
        # Get dataset info
        info = get_dataset_info(train_dataset, cal_dataset)
        results.append(info)
        
        # Print info for this run
        print(f"Training set size: {info['train_indices_len']}")
        print(f"Calibration set size: {info['cal_indices_len']}")
        print(f"Training indices checksum: {info['train_indices_sum']}")
        print(f"Calibration indices checksum: {info['cal_indices_sum']}")
        print(f"First 5 training labels: {info['train_first_samples']}")
        print(f"First 5 calibration labels: {info['cal_first_samples']}")
    
    # Verify consistency across runs
    print("\nVerifying consistency:")
    consistent = True
    base_info = results[0]
    for i, current_info in enumerate(results[1:], 1):
        if (current_info['train_indices_sum'] != base_info['train_indices_sum'] or
            current_info['cal_indices_sum'] != base_info['cal_indices_sum'] or
            current_info['train_indices_len'] != base_info['train_indices_len'] or
            current_info['cal_indices_len'] != base_info['cal_indices_len'] or
            current_info['train_first_samples'] != base_info['train_first_samples'] or
            current_info['cal_first_samples'] != base_info['cal_first_samples']):
            print(f"Inconsistency detected in run {i+1}!")
            consistent = False
    
    if consistent:
        print("\nVERIFICATION SUCCESSFUL: All runs produced identical splits!")
        print("The splits are fixed and consistent across runs.")
    else:
        print("\nVERIFICATION FAILED: Inconsistencies detected between runs!")
        print("The splits are not remaining constant.")

if __name__ == "__main__":
    verify_consistency(num_runs=3)