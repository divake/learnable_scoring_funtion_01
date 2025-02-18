import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
import timm
import numpy as np
import os
import logging
from PIL import Image
from typing import List, Tuple
import random

from .base import BaseDataset

class ImageNetDataset(TorchDataset):
    """Custom ImageNet dataset implementation for loading from directory"""
    
    def __init__(self, root: str, transform=None):
        """
        Initialize ImageNet dataset from directory
        
        Args:
            root: Root directory containing class folders
            transform: Optional transform to be applied on images
        """
        self.root = root
        self.transform = transform
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
            
        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if not self.classes:
            raise RuntimeError(f"No class directories found in {root}")
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
        logging.info(f"Found {len(self.classes)} classes in {root}")
        logging.info(f"Total samples: {len(self.samples)}")
        
    def _make_dataset(self) -> List[Tuple[str, int]]:
        """Create list of (image path, class_index) tuples"""
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(class_dir, img_name)
                    samples.append((path, class_idx))
                    
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class Dataset(BaseDataset):
    """
    ImageNet dataset implementation
    
    Dataset is split into:
    - Training set: 30,000 samples
    - Calibration set: 10,000 samples
    - Test set: 10,000 samples
    
    All splits maintain class balance with all 1000 classes represented.
    """
    
    def __init__(self, config):
        """
        Initialize ImageNet dataset
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        self._num_classes = 1000
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
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
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            )
        ])
    
    def setup(self):
        """Setup ImageNet dataset with transforms and splits"""
        dataset_path = self.config['data_dir']
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"ImageNet dataset not found at {dataset_path}")
        
        logging.info("Loading ImageNet dataset...")
        
        # Load full training dataset
        train_dataset = ImageNetDataset(
            root=dataset_path,
            transform=self.train_transform
        )
        
        # Create train split (30,000 samples)
        train_indices = list(range(len(train_dataset)))
        random.shuffle(train_indices)
        train_indices = train_indices[:30000]
        train_dataset = Subset(train_dataset, train_indices)
        
        # Load validation dataset for calibration and test splits
        val_dataset = ImageNetDataset(
            root=dataset_path,
            transform=self.test_transform
        )
        
        # Create calibration and test splits (10,000 samples each)
        val_indices = list(range(len(val_dataset)))
        random.shuffle(val_indices)
        
        cal_indices = val_indices[:10000]
        test_indices = val_indices[10000:20000]
        
        cal_dataset = Subset(val_dataset, cal_indices)
        test_dataset = Subset(val_dataset, test_indices)
        
        # Verify dataset sizes
        logging.info(f"\nDataset sizes:")
        logging.info(f"Train: {len(train_dataset)}")
        logging.info(f"Calibration: {len(cal_dataset)}")
        logging.info(f"Test: {len(test_dataset)}")
        
        # Create dataloaders with verified settings
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
        Get the pretrained ViT model for ImageNet
        
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
                    map_location=self.config['device']
                )
                model.load_state_dict(state_dict)
                logging.info(f"Loaded pretrained weights from {self.config['model']['pretrained_path']}")
            except Exception as e:
                logging.error(f"Failed to load pretrained weights: {str(e)}")
        
        model.eval()  # Set to evaluation mode
        return model 