import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
import numpy as np
import os
import logging
from PIL import Image
from typing import List, Tuple
import random
import sys

# Temporarily disable wandb to avoid import issues
os.environ['WANDB_DISABLED'] = 'true'
sys.modules['wandb'] = None

try:
    import timm
except ImportError:
    timm = None

from .base import BaseDataset

class Places365Dataset(TorchDataset):
    """Custom Places365 dataset implementation for loading from directory"""
    
    def __init__(self, img_dir: str, list_file: str, transform=None):
        """
        Initialize Places365 dataset from directory and file list
        
        Args:
            img_dir: Root directory containing images
            list_file: Text file with image paths and labels
            transform: Optional transform to be applied on images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")
            
        # Load image paths and labels from list file
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, label = parts
                    self.samples.append((img_name, int(label)))
        
        logging.info(f"Loaded {len(self.samples)} samples from {list_file}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        # Remove leading slash if present to make it relative
        if img_name.startswith('/'):
            img_name = img_name[1:]
        full_path = os.path.join(self.img_dir, img_name)
            
        image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class Dataset(BaseDataset):
    """
    Places365 dataset implementation
    
    Uses the Places365-Standard dataset:
    - Training set: Uses a subset of training data for efficiency
    - Calibration set: 50% of validation data
    - Test set: 50% of validation data
    
    The validation set is split 50-50 for calibration and testing.
    """
    
    def __init__(self, config):
        """
        Initialize Places365 dataset
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        self._num_classes = 365
        
        # Determine model type and get appropriate image size
        if 'type' not in config['model']:
            raise ValueError("Model configuration must have 'type' field specifying 'resnet' or 'vit'")
        
        self.model_type = config['model']['type'].lower()
        if self.model_type == 'vit':
            if 'vit' not in config['model'] or 'img_size' not in config['model']['vit']:
                raise ValueError("ViT model configuration must have 'vit.img_size' field")
            img_size = config['model']['vit']['img_size']
        else:
            raise ValueError(f"Only 'vit' model type is supported for Places365. Got: {self.model_type}")
            
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(img_size),
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
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['dataset']['mean'],
                std=config['dataset']['std']
            )
        ])
    
    def setup(self):
        """Setup Places365 dataset with transforms and splits"""
        # Get data paths from config
        data_dir = os.path.join(self.config['base_dir'], 'data/places365_small')
            
        # Check for data directories
        train_img_dir = os.path.join(data_dir, 'data_256_standard')
        val_img_dir = os.path.join(data_dir, 'val_256')
        
        # Check for list files
        train_list_file = os.path.join(data_dir, 'places365_train_standard.txt')
        val_list_file = os.path.join(data_dir, 'places365_val.txt')
        
        # Verify paths exist
        if not os.path.exists(train_img_dir):
            raise FileNotFoundError(f"Training images directory not found at {train_img_dir}")
        if not os.path.exists(val_img_dir):
            raise FileNotFoundError(f"Validation images directory not found at {val_img_dir}")
        if not os.path.exists(train_list_file):
            raise FileNotFoundError(f"Training list file not found at {train_list_file}")
        if not os.path.exists(val_list_file):
            raise FileNotFoundError(f"Validation list file not found at {val_list_file}")
        
        logging.info("Loading Places365 dataset...")
        
        # Load training dataset (use subset for efficiency)
        train_dataset = Places365Dataset(
            img_dir=train_img_dir,
            list_file=train_list_file,
            transform=self.train_transform
        )
        
        # Use a subset of training data for efficiency (configurable)
        train_subset_size = self.config.get('dataset', {}).get('train_subset_size', 50000)
        if train_subset_size < len(train_dataset):
            # Random subset for training
            train_indices = np.random.choice(len(train_dataset), train_subset_size, replace=False)
            train_dataset = Subset(train_dataset, train_indices)
            logging.info(f"Using training subset: {train_subset_size} samples")
        
        # Load validation dataset
        val_dataset = Places365Dataset(
            img_dir=val_img_dir,
            list_file=val_list_file,
            transform=self.test_transform
        )
        
        # Split validation set 50-50 for calibration and testing with class balance
        # First, group samples by class - access labels directly without loading images
        class_to_indices = {}
        for idx in range(len(val_dataset.samples)):
            _, label = val_dataset.samples[idx]  # Access label directly from samples list
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        cal_indices = []
        test_indices = []
        
        # For each class, split samples 50-50
        random.seed(42)  # For reproducible splits
        for class_label in sorted(class_to_indices.keys()):
            class_indices = class_to_indices[class_label]
            random.shuffle(class_indices)
            
            # Split this class's samples equally
            split_point = len(class_indices) // 2
            cal_indices.extend(class_indices[:split_point])
            test_indices.extend(class_indices[split_point:])
        
        # Shuffle the final indices to mix classes during training
        random.shuffle(cal_indices)
        random.shuffle(test_indices)
        
        cal_dataset = Subset(val_dataset, cal_indices)
        test_dataset = Subset(val_dataset, test_indices)
        
        # Log class distribution info
        logging.info(f"Class-balanced split: {len(class_to_indices)} classes")
        logging.info(f"Samples per class in calibration: ~{len(cal_indices) // len(class_to_indices)}")
        logging.info(f"Samples per class in test: ~{len(test_indices) // len(class_to_indices)}")
        
        # Verify dataset sizes
        logging.info(f"\nDataset sizes:")
        logging.info(f"Train: {len(train_dataset):,} samples")
        logging.info(f"Calibration: {len(cal_dataset):,} samples")
        logging.info(f"Test: {len(test_dataset):,} samples")
        
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
        Get the pretrained ViT model for Places365
        
        Returns:
            torch.nn.Module: Pretrained ViT model
        """
        if self.model_type != 'vit':
            raise ValueError("Only ViT model is supported for Places365")
            
        if timm is None:
            raise ImportError("timm library is required for ViT models but has import issues.")
        
        if 'vit' not in self.config['model']:
            raise ValueError("ViT model configuration missing 'vit' section")
            
        vit_config = self.config['model']['vit']
        required_fields = ['architecture', 'pretrained_path', 'img_size', 'drop_path_rate', 'drop_rate']
        for field in required_fields:
            if field not in vit_config:
                raise ValueError(f"ViT configuration must have '{field}' field")
                
        # Create ViT model
        model = timm.create_model(
            vit_config['architecture'],
            pretrained=False,
            num_classes=self._num_classes,
            img_size=vit_config['img_size'],
            drop_path_rate=vit_config['drop_path_rate'],
            drop_rate=vit_config['drop_rate']
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(enable=True)
        
        # Move model to device first
        model = model.to(self.config['device'])
        
        # Load pretrained weights
        pretrained_path = vit_config['pretrained_path']
        if not os.path.isabs(pretrained_path):
            pretrained_path = os.path.join(self.config['base_dir'], pretrained_path)
            
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")
            
        # config['device'] is already a torch.device or string like 'cuda:1'
        if isinstance(self.config['device'], str):
            device = torch.device(self.config['device'])
        else:
            device = self.config['device']
            
        state_dict = torch.load(
            pretrained_path,
            map_location=device,
            weights_only=False
        )
        
        # Handle different state dict formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded pretrained ViT weights from {pretrained_path}")
        
        model.eval()  # Set to evaluation mode
        return model