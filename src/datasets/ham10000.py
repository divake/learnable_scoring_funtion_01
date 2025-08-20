import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from typing import Tuple, Dict, Optional
import logging

from .base import BaseDataset


class HAM10000TorchDataset(TorchDataset):
    """HAM10000 Skin Lesion PyTorch Dataset"""
    
    def __init__(self, dataframe, img_dir, processor=None, transform=None, mode='train'):
        """
        Args:
            dataframe: DataFrame with image_id and dx columns
            img_dir: Directory containing images
            processor: HuggingFace ViT processor (if using ViT)
            transform: torchvision transforms (if not using ViT)
            mode: 'train', 'val', 'cal', or 'test'
        """
        self.df = dataframe
        self.img_dir = img_dir
        self.processor = processor
        self.transform = transform
        self.mode = mode
        
        self.class_to_idx = {
            'akiec': 0,  # Actinic keratoses and intraepithelial carcinoma
            'bcc': 1,    # Basal cell carcinoma
            'bkl': 2,    # Benign keratosis-like lesions
            'df': 3,     # Dermatofibroma
            'mel': 4,    # Melanoma
            'nv': 5,     # Melanocytic nevi
            'vasc': 6    # Vascular lesions
        }
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        
        image = Image.open(img_path).convert('RGB')
        
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            image_tensor = inputs['pixel_values'].squeeze(0)
        elif self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        label = self.class_to_idx[row['dx']]
        
        return image_tensor, label


class Dataset(BaseDataset):
    """HAM10000 dataset implementation following the framework pattern"""
    
    def __init__(self, config):
        """
        Initialize HAM10000 dataset
        
        Args:
            config: Configuration object containing dataset parameters
        """
        super().__init__(config)
        self._num_classes = 7
        
        # Determine model type and setup transforms/processor
        self.model_type = config.get('model', {}).get('type', 'vit').lower()
        
        if self.model_type == 'vit':
            # For ViT, we'll use the HuggingFace processor
            model_name = config.get('model', {}).get('name', 'google/vit-base-patch16-224')
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.train_transform = None
            self.test_transform = None
        else:
            # For other models, use standard transforms
            self.processor = None
            self.train_transform = self._get_train_transforms(config)
            self.test_transform = self._get_test_transforms(config)
    
    def setup(self):
        """Setup HAM10000 dataset with proper validation split for conformal prediction"""
        data_dir = self.config.get('dataset', {}).get('data_dir', 'data/ham10000')
        # Construct full path from base_dir if data_dir is relative
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(self.config['base_dir'], data_dir)
        
        img_dir = os.path.join(data_dir, 'images')
        splits_dir = os.path.join(data_dir, 'splits')
        
        # Check if splits exist
        if not os.path.exists(splits_dir):
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
        
        # Load pre-defined splits
        train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(splits_dir, 'validation.csv'))
        
        # Split validation set 50-50 into calibration and test with class balance
        # Group samples by class
        class_to_indices = {}
        for idx, row in val_df.iterrows():
            class_label = row['dx']
            if class_label not in class_to_indices:
                class_to_indices[class_label] = []
            class_to_indices[class_label].append(idx)
        
        cal_indices = []
        test_indices = []
        
        # For each class, split samples 50-50
        np.random.seed(self.config.get('seed', 42))  # For reproducible splits
        for class_label in sorted(class_to_indices.keys()):
            class_indices = class_to_indices[class_label]
            np.random.shuffle(class_indices)
            
            # Split this class's samples equally
            split_point = len(class_indices) // 2
            
            # Ensure at least one sample in each split if possible
            if len(class_indices) >= 2:
                cal_indices.extend(class_indices[:split_point] if split_point > 0 else [class_indices[0]])
                test_indices.extend(class_indices[split_point:] if split_point < len(class_indices) else [class_indices[-1]])
            elif len(class_indices) == 1:
                # If only one sample, assign to calibration (more important for threshold)
                cal_indices.append(class_indices[0])
        
        # Create calibration and test dataframes
        cal_df = val_df.iloc[cal_indices].reset_index(drop=True)
        test_df = val_df.iloc[test_indices].reset_index(drop=True)
        
        # Log class distribution
        logging.info(f"Class-balanced split from validation set:")
        logging.info(f"  Train: {len(train_df)} samples")
        logging.info(f"  Calibration: {len(cal_df)} samples (~50% of validation)")
        logging.info(f"  Test: {len(test_df)} samples (~50% of validation)")
        
        # Check class distribution in splits
        cal_class_dist = cal_df['dx'].value_counts().sort_index()
        test_class_dist = test_df['dx'].value_counts().sort_index()
        logging.info(f"Classes in calibration: {len(cal_class_dist)}, in test: {len(test_class_dist)}")
        
        logging.info(f"Dataset sizes - Train: {len(train_df)}, Cal: {len(cal_df)}, Test: {len(test_df)}")
        
        # Create datasets
        train_dataset = HAM10000TorchDataset(
            train_df, img_dir, self.processor, self.train_transform, mode='train'
        )
        cal_dataset = HAM10000TorchDataset(
            cal_df, img_dir, self.processor, self.test_transform, mode='cal'
        )
        test_dataset = HAM10000TorchDataset(
            test_df, img_dir, self.processor, self.test_transform, mode='test'
        )
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('dataset', {}).get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        self.cal_loader = DataLoader(
            cal_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        logging.info(f"Created dataloaders with batch_size={batch_size}, num_workers={num_workers}")
    
    def get_model(self):
        """
        Get the pretrained model for HAM10000
        
        Returns:
            torch.nn.Module: Pretrained model (ViT or ResNet based on config)
        """
        if self.model_type == 'vit':
            # Load ViT model
            model_name = self.config.get('model', {}).get('name', 'google/vit-base-patch16-224')
            
            # Check if we have a saved fine-tuned model
            saved_model_path = 'models/ham10000_vit_base.pth'
            
            if os.path.exists(saved_model_path):
                logging.info(f"Loading fine-tuned ViT model from {saved_model_path}")
                # Create model architecture
                model = ViTForHAM10000(num_classes=self._num_classes, model_name=model_name)
                # Load weights with map_location to handle device placement
                model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
                return model
            else:
                logging.info(f"Loading pre-trained ViT model from HuggingFace: {model_name}")
                return ViTForHAM10000(num_classes=self._num_classes, model_name=model_name)
        else:
            # For ResNet or other models
            import torchvision.models as models
            logging.info("Loading ResNet18 model")
            model = models.resnet18(pretrained=True)
            # Modify final layer for 7 classes
            model.fc = torch.nn.Linear(model.fc.in_features, self._num_classes)
            return model
    
    def _get_train_transforms(self, config):
        """Get training data augmentation transforms"""
        aug_config = config.get('augmentation', {}).get('train', {})
        transform_list = []
        
        if 'random_resized_crop' in aug_config:
            transform_list.append(transforms.RandomResizedCrop(
                aug_config['random_resized_crop']['size'],
                scale=tuple(aug_config['random_resized_crop'].get('scale', [0.8, 1.0]))
            ))
        else:
            transform_list.append(transforms.Resize(256))
            transform_list.append(transforms.RandomCrop(224))
        
        if aug_config.get('random_horizontal_flip', 0) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(
                p=aug_config['random_horizontal_flip']
            ))
        
        if aug_config.get('random_vertical_flip', 0) > 0:
            transform_list.append(transforms.RandomVerticalFlip(
                p=aug_config['random_vertical_flip']
            ))
        
        if 'random_rotation' in aug_config:
            transform_list.append(transforms.RandomRotation(
                degrees=aug_config['random_rotation']
            ))
        
        if 'color_jitter' in aug_config:
            cj = aug_config['color_jitter']
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get('brightness', 0),
                contrast=cj.get('contrast', 0),
                saturation=cj.get('saturation', 0),
                hue=cj.get('hue', 0)
            ))
        
        # Always add normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.get('dataset', {}).get('mean', [0.485, 0.456, 0.406]),
                std=config.get('dataset', {}).get('std', [0.229, 0.224, 0.225])
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def _get_test_transforms(self, config):
        """Get validation/test data transforms"""
        aug_config = config.get('augmentation', {}).get('val', {})
        transform_list = []
        
        if 'resize' in aug_config:
            transform_list.append(transforms.Resize(aug_config['resize']))
        else:
            transform_list.append(transforms.Resize(256))
        
        if 'center_crop' in aug_config:
            transform_list.append(transforms.CenterCrop(aug_config['center_crop']))
        else:
            transform_list.append(transforms.CenterCrop(224))
        
        # Always add normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.get('dataset', {}).get('mean', [0.485, 0.456, 0.406]),
                std=config.get('dataset', {}).get('std', [0.229, 0.224, 0.225])
            )
        ])
        
        return transforms.Compose(transform_list)


class ViTForHAM10000(torch.nn.Module):
    """ViT model for HAM10000 classification"""
    
    def __init__(self, num_classes=7, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(0.1)
        
    def forward(self, pixel_values):
        # Ensure the model is on the same device as the input
        device = pixel_values.device
        if next(self.parameters()).device != device:
            self = self.to(device)
        
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


if __name__ == "__main__":
    import yaml
    
    # Test the dataset
    with open('src/config/ham10000.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = Dataset(config)
    dataset.setup()
    
    print("\nDataset initialized successfully!")
    print(f"Train batches: {len(dataset.train_loader)}")
    print(f"Cal batches: {len(dataset.cal_loader)}")
    print(f"Test batches: {len(dataset.test_loader)}")
    
    # Test getting a batch
    batch = next(iter(dataset.train_loader))
    images, labels = batch
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test getting the model
    model = dataset.get_model()
    print(f"\nModel loaded: {type(model).__name__}")