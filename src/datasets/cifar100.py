import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import logging
import sys

# Temporarily disable wandb to avoid import issues
os.environ['WANDB_DISABLED'] = 'true'
sys.modules['wandb'] = None

try:
    import timm
except ImportError:
    timm = None

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
        
        # Determine model type and get appropriate image size
        if 'type' not in config['model']:
            raise ValueError("Model configuration must have 'type' field specifying 'resnet' or 'vit'")
        
        self.model_type = config['model']['type'].lower()
        if self.model_type == 'resnet':
            img_size = 32  # ResNet uses original CIFAR image size
        elif self.model_type == 'vit':
            if 'vit' not in config['model'] or 'img_size' not in config['model']['vit']:
                raise ValueError("ViT model configuration must have 'vit.img_size' field")
            img_size = config['model']['vit']['img_size']
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Expected 'resnet' or 'vit'")
            
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), 
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
            transforms.Resize((img_size, img_size), 
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
        Get the pretrained model for CIFAR-100 (ResNet or ViT based on config)
        
        Returns:
            torch.nn.Module: Pretrained model
        """
        if self.model_type == 'resnet':
            # Load ResNet model using torchvision with CIFAR-specific modifications
            if 'resnet' not in self.config['model']:
                raise ValueError("ResNet model configuration missing 'resnet' section")
            
            resnet_config = self.config['model']['resnet']
            if 'architecture' not in resnet_config:
                raise ValueError("ResNet configuration must have 'architecture' field")
            if 'pretrained_path' not in resnet_config:
                raise ValueError("ResNet configuration must have 'pretrained_path' field")
                
            # Use torchvision ResNet18 for CIFAR
            from torchvision import models
            model = models.resnet18(pretrained=False, num_classes=self._num_classes)
            
            # Modify the first conv layer to match CIFAR architecture (3x3 instead of 7x7)
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = torch.nn.Identity()  # Remove max pooling for CIFAR
            
            # Move model to device first
            model = model.to(self.config['device'])
            
            # Load pretrained weights
            pretrained_path = resnet_config['pretrained_path']
            # config['device'] is already a torch.device or string like 'cuda:1'
            if isinstance(self.config['device'], str):
                device = torch.device(self.config['device'])
            else:
                device = self.config['device']
            state_dict = torch.load(
                os.path.join(self.config['base_dir'], pretrained_path),
                map_location=device,
                weights_only=False
            )
            model.load_state_dict(state_dict)
            logging.info(f"Loaded pretrained ResNet weights from {pretrained_path}")
                
        elif self.model_type == 'vit':
            # Load ViT model
            if timm is None:
                raise ImportError("timm library is required for ViT models but has import issues. Please use ResNet model type instead.")
            
            if 'vit' not in self.config['model']:
                raise ValueError("ViT model configuration missing 'vit' section")
                
            vit_config = self.config['model']['vit']
            required_fields = ['architecture', 'pretrained_path', 'img_size', 'drop_path_rate', 'drop_rate']
            for field in required_fields:
                if field not in vit_config:
                    raise ValueError(f"ViT configuration must have '{field}' field")
            model = timm.create_model(
                vit_config['architecture'],
                pretrained=False,
                num_classes=self._num_classes,
                img_size=vit_config['img_size'],
                drop_path_rate=vit_config['drop_path_rate'],
                drop_rate=vit_config['drop_rate']
            )
            
            # Enable gradient checkpointing for memory efficiency
            model.set_grad_checkpointing(enable=True)
            
            # Move model to device first
            model = model.to(self.config['device'])
            
            # Load pretrained weights
            pretrained_path = vit_config['pretrained_path']
            # config['device'] is already a torch.device or string like 'cuda:1'
            if isinstance(self.config['device'], str):
                device = torch.device(self.config['device'])
            else:
                device = self.config['device']
            state_dict = torch.load(
                os.path.join(self.config['base_dir'], pretrained_path),
                map_location=device
            )
            model.load_state_dict(state_dict)
            logging.info(f"Loaded pretrained ViT weights from {pretrained_path}")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Expected 'resnet' or 'vit'")
        
        model.eval()  # Set to evaluation mode
        return model 