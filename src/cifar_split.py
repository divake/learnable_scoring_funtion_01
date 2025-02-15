# cifar_split.py
"""
This script handles downloading CIFAR-10, creating balanced splits, saving them,
and verifying class distributions. Creates splits of:
- Training: 40,000 images (4,000 per class)
- Calibration: 10,000 images (1,000 per class)
- Test: 10,000 images (original)
"""

import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter

def create_balanced_split(dataset, cal_size_per_class=1000):
    """
    Create balanced train-calibration split with equal samples per class.
    """
    targets = torch.tensor(dataset.targets)
    train_indices = []
    cal_indices = []
    
    for class_idx in range(10):
        # Get indices for current class
        class_indices = torch.where(targets == class_idx)[0]
        
        # Randomly permute indices
        perm = torch.randperm(len(class_indices))
        class_indices = class_indices[perm]
        
        # Split indices for this class
        cal_indices.extend(class_indices[:cal_size_per_class].tolist())
        train_indices.extend(class_indices[cal_size_per_class:].tolist())
    
    return train_indices, cal_indices

def save_split_indices(train_indices, cal_indices, base_path):
    """
    Save the split indices to a pickle file.
    """
    split_info = {
        'train_indices': train_indices,
        'cal_indices': cal_indices
    }
    
    splits_dir = os.path.join(base_path, 'data', 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    split_path = os.path.join(splits_dir, 'cifar10_splits.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"Split indices saved to {split_path}")

def load_split_indices(base_path):
    """
    Load the split indices from pickle file.
    """
    split_path = os.path.join(base_path, 'data', 'splits', 'cifar10_splits.pkl')
    if os.path.exists(split_path):
        with open(split_path, 'rb') as f:
            split_info = pickle.load(f)
        return split_info['train_indices'], split_info['cal_indices']
    return None, None

def verify_class_distribution(dataset, name="Dataset"):
    """
    Print class distribution in a dataset.
    """
    if isinstance(dataset, Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets
        
    class_counts = Counter(targets)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"\nClass distribution in {name}:")
    for i in range(10):
        print(f"{classes[i]}: {class_counts[i]} images")

def setup_cifar10(base_path='/ssd_4TB/divake/learnable_scoring_funtion_01', batch_size=128, save_splits=True):
    """
    Set up CIFAR-10 dataset with balanced splits and save them.
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Create data directory
    data_path = os.path.join(base_path, 'data', 'cifar10')
    os.makedirs(data_path, exist_ok=True)
    
    # Download and load training data
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    # Force recreate splits
    print("Creating new splits...")
    train_indices, cal_indices = create_balanced_split(full_train_dataset)
    if save_splits:
        # Remove old split file if it exists
        split_path = os.path.join(base_path, 'data', 'splits', 'cifar10_splits.pkl')
        if os.path.exists(split_path):
            print("Removed old split file")
            os.remove(split_path)
        save_split_indices(train_indices, cal_indices, base_path)
    
    # Create datasets using indices
    train_dataset = Subset(full_train_dataset, train_indices)
    cal_dataset = Subset(full_train_dataset, cal_indices)
    
    # Verify splits are balanced
    print("\nVerifying splits are balanced:")
    verify_class_distribution(train_dataset, "Training set")
    verify_class_distribution(cal_dataset, "Calibration set")
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset

if __name__ == "__main__":
    # Set up datasets
    train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_cifar10()
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Calibration set size: {len(cal_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # Verify class distributions
    verify_class_distribution(train_dataset, "Training set")
    verify_class_distribution(cal_dataset, "Calibration set")
    verify_class_distribution(test_dataset, "Test set")
    
    # Verify batch shapes
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")