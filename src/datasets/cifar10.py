import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os
import logging

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
        
        # Load full training set
        full_train = torchvision.datasets.CIFAR10(
            root=self.config['data_dir'],
            train=True,
            download=download,
            transform=transform
        )
        
        # Calculate split sizes
        total_size = len(full_train)
        train_size = int(total_size * self.config['splits']['train_size'])
        val_size = int(total_size * self.config['splits']['val_size'])
        cal_size = int(total_size * self.config['splits']['cal_size'])
        test_size = total_size - train_size - val_size - cal_size
        
        # Split dataset
        train_data, val_data, cal_data, remaining = random_split(
            full_train, 
            [train_size, val_size, cal_size, test_size]
        )
        
        # Load test set
        test_data = torchvision.datasets.CIFAR10(
            root=self.config['data_dir'],
            train=False,
            download=download,
            transform=transform
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            val_data,
            batch_size=self.config['batch_size'],
            shuffle=False,
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
                weights_only=True
            )
        )
        
        model = model.to(self.config['device'])
        model.eval()
        
        return model 