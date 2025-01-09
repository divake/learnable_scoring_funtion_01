#src/cifar_split.py
"""
This script handles downloading CIFAR-100, creating balanced splits, saving them,
and verifying class distributions. Creates splits of:
- Training: 40,000 images (400 per class)
- Calibration: 10,000 images (100 per class)
- Test: 10,000 images (original)
"""
import os
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter
def create_balanced_split(dataset, cal_size_per_class=100):
    """
    Create balanced train-calibration split with equal samples per class.
    """
    targets = torch.tensor(dataset.targets)
    train_indices = []
    cal_indices = []
    
    for class_idx in range(100):  # Updated to 100 classes
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
    
    split_path = os.path.join(splits_dir, 'cifar100_splits.pkl')  # Updated filename
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"Split indices saved to {split_path}")
def load_split_indices(base_path):
    """
    Load the split indices from pickle file.
    """
    split_path = os.path.join(base_path, 'data', 'splits', 'cifar100_splits.pkl')  # Updated filename
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
    
    # CIFAR-100 fine labels
    fine_labels = dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes
    
    print(f"\nClass distribution in {name}:")
    for i in range(100):  # Updated to 100 classes
        if class_counts[i] > 0:  # Only print classes that have samples
            print(f"{fine_labels[i]}: {class_counts[i]} images")
def setup_cifar100(base_path='/ssd_4TB/divake/learnable_scoring_fn', batch_size=128, save_splits=True):
    """
    Set up CIFAR-100 dataset with balanced splits and save them.
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    # Create data directory
    data_path = os.path.join(base_path, 'data', 'cifar100')  # Updated path
    os.makedirs(data_path, exist_ok=True)
    
    # Download and load training data
    full_train_dataset = torchvision.datasets.CIFAR100(  # Updated to CIFAR100
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    # Create or load splits
    train_indices, cal_indices = load_split_indices(base_path)
    
    if train_indices is None or cal_indices is None:
        print("Creating new splits...")
        train_indices, cal_indices = create_balanced_split(full_train_dataset)
        if save_splits:
            save_split_indices(train_indices, cal_indices, base_path)
    else:
        print("Loading existing splits...")
    
    # Create datasets using indices
    train_dataset = Subset(full_train_dataset, train_indices)
    cal_dataset = Subset(full_train_dataset, cal_indices)
    
    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR100(  # Updated to CIFAR100
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
    train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_cifar100()
    
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
