import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset as TorchDataset
import numpy as np
import os
import json
import logging
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from .base import BaseDataset


class PlantNetDataset(TorchDataset):
    """PlantNet-300K PyTorch Dataset"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels (class indices)
            transform: torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class Dataset(BaseDataset):
    """PlantNet-300K dataset implementation"""
    
    def __init__(self, config):
        """
        Initialize PlantNet-300K dataset
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        self._num_classes = 1081
        
        # Setup paths
        self.data_dir = config['dataset']['data_dir']
        self.images_dir = os.path.join(self.data_dir, 'images')
        
        # Load species mapping
        species_map_path = os.path.join(self.data_dir, 'plantnet300K_species_id_2_name.json')
        with open(species_map_path, 'r') as f:
            self.species_map = json.load(f)
        
        # Create class to index mapping
        self.class_ids = sorted(list(self.species_map.keys()))
        self.class_to_idx = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        self.idx_to_class = {idx: class_id for class_id, idx in self.class_to_idx.items()}
        
        # Determine model type and get appropriate image size
        if 'type' not in config['model']:
            raise ValueError("Model configuration must have 'type' field")
        
        self.model_type = config['model']['type'].lower()
        
        if self.model_type == 'vit':
            if 'vit' not in config['model'] or 'img_size' not in config['model']['vit']:
                raise ValueError("ViT model configuration must have 'vit.img_size' field")
            img_size = config['model']['vit']['img_size']
        else:
            raise ValueError(f"PlantNet currently only supports ViT model, got: {self.model_type}")
            
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
    
    def _load_split_data(self, split_name):
        """Load data for a specific split"""
        split_dir = os.path.join(self.images_dir, split_name)
        
        image_paths = []
        labels = []
        
        for class_id in os.listdir(split_dir):
            if class_id not in self.class_to_idx:
                continue
                
            class_dir = os.path.join(split_dir, class_id)
            if not os.path.isdir(class_dir):
                continue
            
            class_idx = self.class_to_idx[class_id]
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_idx)
        
        return image_paths, labels
    
    def setup(self):
        """Setup PlantNet-300K dataset with proper splits"""
        
        # Load training data (use 100% as requested)
        logging.info("Loading PlantNet-300K training data...")
        train_paths, train_labels = self._load_split_data('train')
        
        # Load validation data (will be split into cal and test)
        logging.info("Loading PlantNet-300K validation data...")
        val_paths, val_labels = self._load_split_data('val')
        
        # Create stratified split of validation set into calibration and test
        # Ensure class balance by splitting each class's samples equally
        val_labels_array = np.array(val_labels)
        cal_indices = []
        test_indices = []
        
        logging.info("Creating balanced calibration/test split from validation data...")
        
        # For each class, split its samples equally between calibration and test
        for class_idx in range(self._num_classes):
            class_indices = np.where(val_labels_array == class_idx)[0]
            
            if len(class_indices) > 0:
                # Shuffle with fixed seed for reproducibility
                np.random.seed(self.config.get('seed', 42) + class_idx)
                np.random.shuffle(class_indices)
                
                # Split in half
                split_idx = len(class_indices) // 2
                
                # Ensure at least one sample in each split if possible
                if len(class_indices) >= 2:
                    cal_indices.extend(class_indices[:split_idx] if split_idx > 0 else [class_indices[0]])
                    test_indices.extend(class_indices[split_idx:] if split_idx < len(class_indices) else [class_indices[-1]])
                elif len(class_indices) == 1:
                    # If only one sample, randomly assign to cal or test
                    if np.random.random() < 0.5:
                        cal_indices.append(class_indices[0])
                    else:
                        test_indices.append(class_indices[0])
        
        # Create calibration and test data
        cal_paths = [val_paths[i] for i in cal_indices]
        cal_labels = [val_labels[i] for i in cal_indices]
        test_paths = [val_paths[i] for i in test_indices]
        test_labels = [val_labels[i] for i in test_indices]
        
        # Create datasets
        train_dataset = PlantNetDataset(train_paths, train_labels, transform=self.train_transform)
        cal_dataset = PlantNetDataset(cal_paths, cal_labels, transform=self.test_transform)
        test_dataset = PlantNetDataset(test_paths, test_labels, transform=self.test_transform)
        
        # Verify dataset sizes and class distribution
        logging.info(f"\nDataset sizes:")
        logging.info(f"Train: {len(train_dataset)} (100% of original training set)")
        logging.info(f"Calibration: {len(cal_dataset)} (~50% of validation set)")
        logging.info(f"Test: {len(test_dataset)} (~50% of validation set)")
        
        # Check class distribution
        cal_classes = len(set(cal_labels))
        test_classes = len(set(test_labels))
        logging.info(f"Number of classes - Cal: {cal_classes}, Test: {test_classes}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=True
        )
        
        self.cal_loader = DataLoader(
            cal_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset'].get('num_workers', 4),
            pin_memory=True
        )
    
    def get_model(self):
        """
        Get the pretrained model for PlantNet-300K (ViT)
        
        Returns:
            torch.nn.Module: Pretrained model
        """
        if self.model_type != 'vit':
            raise ValueError(f"PlantNet only supports ViT model, got: {self.model_type}")
        
        if 'vit' not in self.config['model']:
            raise ValueError("ViT model configuration missing 'vit' section")
            
        vit_config = self.config['model']['vit']
        required_fields = ['architecture', 'pretrained_path']
        for field in required_fields:
            if field not in vit_config:
                raise ValueError(f"ViT configuration must have '{field}' field")
        
        # Load the fine-tuned ViT model
        pretrained_path = vit_config['pretrained_path']
        full_path = os.path.join(self.config['base_dir'], pretrained_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Pretrained model not found at {full_path}. "
                                  f"Please run fine-tuning first.")
        
        # Create model with wrapper to extract logits
        model_name = vit_config['architecture']
        vit_model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=self._num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Move model to device first
        if isinstance(self.config['device'], str):
            device = torch.device(self.config['device'])
        else:
            device = self.config['device']
        vit_model = vit_model.to(device)
        
        # Load pretrained weights
        checkpoint = torch.load(full_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            vit_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            vit_model.load_state_dict(checkpoint)
        
        logging.info(f"Loaded pretrained ViT weights from {pretrained_path}")
        
        if 'accuracy' in checkpoint:
            logging.info(f"Model accuracy from checkpoint: {checkpoint['accuracy']:.2f}%")
        
        # Create wrapper to extract logits
        class ViTWrapper(torch.nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model
            
            def forward(self, x):
                outputs = self.vit(x)
                return outputs.logits
        
        model = ViTWrapper(vit_model)
        model.eval()
        return model
    
    @property
    def num_classes(self):
        return self._num_classes