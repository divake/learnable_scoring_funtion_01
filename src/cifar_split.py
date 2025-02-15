# cifar_split.py
"""
This script handles downloading CIFAR-10 and creating balanced splits.
The splits are:
- Training: 50,000 images (original training set)
- Calibration: 5,000 images (500 per class, split from test set)
- Test: 5,000 images (500 per class, split from test set)
"""

import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter

def create_balanced_split_from_test(dataset, cal_size_per_class=500):
    """
    Create balanced calibration-test split from the test set with equal samples per class.
    """
    targets = torch.tensor(dataset.targets)
    test_indices = []
    cal_indices = []
    
    for class_idx in range(10):
        # Get indices for current class
        class_indices = torch.where(targets == class_idx)[0]
        
        # Randomly permute indices
        perm = torch.randperm(len(class_indices))
        class_indices = class_indices[perm]
        
        # Split indices for this class
        cal_indices.extend(class_indices[:cal_size_per_class].tolist())
        test_indices.extend(class_indices[cal_size_per_class:].tolist())
    
    return test_indices, cal_indices

def save_split_indices(test_indices, cal_indices, base_path):
    """
    Save the split indices to a pickle file.
    """
    split_info = {
        'test_indices': test_indices,
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
        return split_info['test_indices'], split_info['cal_indices']
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

def setup_cifar10(base_path='/ssd_4TB/divake/learnable_scoring_funtion_01', batch_size=128, save_splits=True,
              train_transform=None, test_transform=None):
    """
    Set up CIFAR-10 dataset with original training set and split test set.
    Args:
        base_path: Base path for data
        batch_size: Batch size for dataloaders
        save_splits: Whether to save the splits
        train_transform: Custom transform for training data
        test_transform: Custom transform for test/validation data
    """
    # Define default transforms if none provided
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    # Create data directory
    data_path = os.path.join(base_path, 'data', 'cifar10')
    os.makedirs(data_path, exist_ok=True)
    
    # Download and load training data (keep full 50k samples)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Load full test set
    full_test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Remove old split file if it exists
    split_path = os.path.join(base_path, 'data', 'splits', 'cifar10_splits.pkl')
    if os.path.exists(split_path):
        print("Removing old split file...")
        os.remove(split_path)
    
    # Create new splits from test set
    print("Creating new splits from test set...")
    test_indices, cal_indices = create_balanced_split_from_test(full_test_dataset)
    if save_splits:
        save_split_indices(test_indices, cal_indices, base_path)
    
    # Create calibration and test datasets
    cal_dataset = Subset(full_test_dataset, cal_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    
    # Verify splits are balanced
    print("\nVerifying dataset sizes and class distributions:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Calibration set size: {len(cal_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    verify_class_distribution(train_dataset, "Training set")
    verify_class_distribution(cal_dataset, "Calibration set")
    verify_class_distribution(test_dataset, "Test set")
    
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
    # Set up datasets and verify splits
    train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_cifar10()
    
    # Verify batch shapes
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")