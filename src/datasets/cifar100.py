import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import timm
import numpy as np
import os
import logging

from .base import BaseDataset

class Dataset(BaseDataset):
    """CIFAR-100 dataset implementation"""
    
    def __init__(self, config):
        """
        Initialize CIFAR-100 dataset
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        self._num_classes = 100
        self.train_transform = transforms.Compose([
            transforms.Resize((config['model']['img_size'], config['model']['img_size']), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            )
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((config['model']['img_size'], config['model']['img_size']), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            )
        ])
    
    def setup(self):
        """Setup CIFAR-100 dataset with transforms and splits"""
        # Check if dataset exists
        dataset_path = os.path.join(self.config['base_dir'], 'data/cifar100')
        download = not os.path.exists(dataset_path)
        if not download:
            logging.info("CIFAR-100 dataset already exists, skipping download")
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root=dataset_path,
            train=True,
            download=download,
            transform=self.train_transform
        )
        
        test_full = torchvision.datasets.CIFAR100(
            root=dataset_path,
            train=False,
            download=download,
            transform=self.test_transform
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
        cal_dataset = Subset(test_full, cal_indices)
        test_dataset = Subset(test_full, test_indices)
        
        # Verify dataset sizes and class distribution
        logging.info(f"\nDataset sizes:")
        logging.info(f"Train: {len(train_dataset)}")
        logging.info(f"Calibration: {len(cal_dataset)}")
        logging.info(f"Test: {len(test_dataset)}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.cal_loader = DataLoader(
            cal_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def get_model(self):
        """
        Get the pretrained ViT model for CIFAR-100
        
        Returns:
            torch.nn.Module: Pretrained ViT model
        """
        model = timm.create_model(
            self.config['model']['architecture'],
            pretrained=False,
            num_classes=self._num_classes,
            img_size=self.config['model']['img_size'],
            drop_path_rate=self.config['model']['drop_path_rate'],
            drop_rate=self.config['model']['drop_rate']
        )
        
        # Enable gradient checkpointing for memory efficiency
        model.set_grad_checkpointing(enable=True)
        
        # Move model to device first
        model = model.to(self.config['device'])
        
        # Load pretrained weights if specified
        if 'pretrained_path' in self.config['model']:
            try:
                state_dict = torch.load(
                    os.path.join(self.config['base_dir'], self.config['model']['pretrained_path']),
                    map_location=self.config['device']  # Load weights directly to the correct device
                )
                model.load_state_dict(state_dict)
                logging.info(f"Loaded pretrained weights from {self.config['model']['pretrained_path']}")
            except Exception as e:
                logging.error(f"Failed to load pretrained weights: {str(e)}")
        
        model.eval()  # Set to evaluation mode
        return model 