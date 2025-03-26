import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader, Subset
import os
import logging
import numpy as np

from src.datasets.base import BaseDataset
import torchvision.models as models

class Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        self._num_classes = config['dataset']['num_classes']
        
    def setup(self):
        """Setup CIFAR-10 dataset with transforms and splits"""
        # Setup transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['dataset']['mean'],
                std=self.config['dataset']['std']
            )
        ])
        
        # Check if dataset exists
        dataset_path = os.path.join(self.config['data_dir'], 'cifar-10-batches-py')
        download = not os.path.exists(dataset_path)
        if not download:
            logging.info("CIFAR-10 dataset already exists, skipping download")
        
        # Load full training set (50,000 samples)
        train_data = torchvision.datasets.CIFAR10(
            root=self.config['data_dir'],
            train=True,
            download=download,
            transform=transform
        )
        
        # Load full test set (10,000 samples)
        test_full = torchvision.datasets.CIFAR10(
            root=self.config['data_dir'],
            train=False,
            download=download,
            transform=transform
        )
        
        # Create stratified split of test set
        test_labels = np.array(test_full.targets)
        cal_indices = []
        test_indices = []
        
        # For each class, split its samples equally between calibration and test
        for class_idx in range(self._num_classes):
            class_indices = np.where(test_labels == class_idx)[0]
            np.random.shuffle(class_indices)
            split_idx = len(class_indices) // 2
            
            cal_indices.extend(class_indices[:split_idx])
            test_indices.extend(class_indices[split_idx:])
        
        # Create calibration and test datasets
        cal_data = Subset(test_full, cal_indices)
        test_data = Subset(test_full, test_indices)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.cal_loader = DataLoader(
            cal_data,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4
        )
    
    def get_model(self):
        """Load pretrained ResNet model"""
        model = models.resnet18(weights=None)
        
        # Wrap each layer in Sequential to match saved architecture
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            setattr(model, layer_name, torch.nn.Sequential(layer))
        
        # Modify fc layer to match saved architecture
        model.fc = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Linear(model.fc.in_features, self.num_classes)
        )
        
        # Load pretrained weights
        model.load_state_dict(
            torch.load(
                os.path.join(self.config['base_dir'], self.config['model']['pretrained_path']),
                map_location=self.config['device'],
                weights_only=False  # Disable weights_only to fix CUDA error
            )
        )
        
        model = model.to(self.config['device'])
        model.eval()
        
        return model 