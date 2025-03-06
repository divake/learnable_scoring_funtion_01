import os
import numpy as np
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader
import json
import glob
from typing import Dict, Any

class Dataset:
    """
    Dataset class for Kinetics-400 using pre-extracted features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Kinetics-400 dataset.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._num_classes = 400
        
        # Path to extracted features
        self.feature_dir = config['dataset']['feature_dir']
        
        # Load features and labels
        logging.info(f"Loading pre-extracted Kinetics-400 features from {self.feature_dir}")
        self._load_data()
        
        # Log dataset information
        logging.info(f"Kinetics-400 dataset loaded with:")
        logging.info(f"  Train: {self.train_features.shape[0]} samples")
        logging.info(f"  Calibration: {self.cal_features.shape[0]} samples")
        logging.info(f"  Test: {self.test_features.shape[0]} samples")
        logging.info(f"  Feature dimension: {self.train_features.shape[1]}")
    
    def _load_data(self):
        """Load pre-extracted features and labels from the VideoMAE model"""
        # Check if feature directory exists
        if not os.path.exists(self.feature_dir):
            raise FileNotFoundError(f"Feature directory not found: {self.feature_dir}")
        
        # Load metadata if available
        metadata_path = os.path.join(self.feature_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                logging.info(f"Loaded metadata: {self.metadata}")
                # Extract feature dimension from metadata
                self.feature_dim = self.metadata.get('feature_dim', 768)
        else:
            self.feature_dim = 768  # Default feature dimension for VideoMAE
        
        # Check if we're using the new directory structure
        train_dir = os.path.join(self.feature_dir, "train", "features")
        val_dir = os.path.join(self.feature_dir, "val", "features")
        test_dir = os.path.join(self.feature_dir, "test", "features")
        
        if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir):
            logging.info("Using new directory structure with individual feature files")
            self._load_data_from_individual_files(train_dir, val_dir, test_dir)
        else:
            logging.info("Using old directory structure with combined feature files")
            self._load_data_from_combined_files()
    
    def _load_data_from_individual_files(self, train_dir, val_dir, test_dir):
        """Load data from individual .npz files in the new directory structure"""
        # Get all feature files
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
        val_files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
        test_files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
        
        logging.info(f"Found {len(train_files)} train files, {len(val_files)} val files, {len(test_files)} test files")
        
        # Load train features and labels
        train_features = []
        train_labels = []
        train_logits = []
        
        for file_path in train_files:
            data = np.load(file_path)
            train_features.append(data['features'])
            train_labels.append(data['label'])
            if 'logits' in data:
                train_logits.append(data['logits'])
        
        # Load validation features and labels (use as calibration set)
        cal_features = []
        cal_labels = []
        cal_logits = []
        
        for file_path in val_files:
            data = np.load(file_path)
            cal_features.append(data['features'])
            cal_labels.append(data['label'])
            if 'logits' in data:
                cal_logits.append(data['logits'])
        
        # Load test features and labels
        test_features = []
        test_labels = []
        test_logits = []
        
        for file_path in test_files:
            data = np.load(file_path)
            test_features.append(data['features'])
            test_labels.append(data['label'])
            if 'logits' in data:
                test_logits.append(data['logits'])
        
        # Convert lists to numpy arrays
        self.train_features = np.array(train_features)
        self.train_labels = np.array(train_labels)
        self.cal_features = np.array(cal_features)
        self.cal_labels = np.array(cal_labels)
        self.test_features = np.array(test_features)
        self.test_labels = np.array(test_labels)
        
        # Store logits if available
        if train_logits and cal_logits and test_logits:
            self.train_logits = np.array(train_logits)
            self.cal_logits = np.array(cal_logits)
            self.test_logits = np.array(test_logits)
            logging.info(f"Loaded logits with shape: {self.train_logits.shape}")
        else:
            logging.warning("Not all files contain logits, will use features directly")
    
    def _load_data_from_combined_files(self):
        """Load data from combined .npy files in the old directory structure"""
        # Train set
        train_features_path = os.path.join(self.feature_dir, "kinetics400_train_features.npy")
        train_labels_path = os.path.join(self.feature_dir, "kinetics400_train_labels.npy")
        
        if not os.path.exists(train_features_path) or not os.path.exists(train_labels_path):
            raise FileNotFoundError(f"Train features or labels not found in {self.feature_dir}")
        
        self.train_features = np.load(train_features_path)
        self.train_labels = np.load(train_labels_path)
        
        # Calibration set
        cal_features_path = os.path.join(self.feature_dir, "kinetics400_cal_features.npy")
        cal_labels_path = os.path.join(self.feature_dir, "kinetics400_cal_labels.npy")
        
        if not os.path.exists(cal_features_path) or not os.path.exists(cal_labels_path):
            raise FileNotFoundError(f"Calibration features or labels not found in {self.feature_dir}")
        
        self.cal_features = np.load(cal_features_path)
        self.cal_labels = np.load(cal_labels_path)
        
        # Test set
        test_features_path = os.path.join(self.feature_dir, "kinetics400_test_features.npy")
        test_labels_path = os.path.join(self.feature_dir, "kinetics400_test_labels.npy")
        
        if not os.path.exists(test_features_path) or not os.path.exists(test_labels_path):
            raise FileNotFoundError(f"Test features or labels not found in {self.feature_dir}")
        
        self.test_features = np.load(test_features_path)
        self.test_labels = np.load(test_labels_path)
        
        # Load logits if available
        try:
            self.train_logits = np.load(os.path.join(self.feature_dir, "kinetics400_train_logits.npy"))
            self.cal_logits = np.load(os.path.join(self.feature_dir, "kinetics400_cal_logits.npy"))
            self.test_logits = np.load(os.path.join(self.feature_dir, "kinetics400_test_logits.npy"))
            
            # Fix logits shape - remove the extra dimension if present
            if self.train_logits.ndim == 3 and self.train_logits.shape[1] == 1:
                logging.info("Reshaping logits to remove extra dimension")
                self.train_logits = self.train_logits.squeeze(1)
                self.cal_logits = self.cal_logits.squeeze(1)
                self.test_logits = self.test_logits.squeeze(1)
            
            logging.info(f"Loaded logits with shape: {self.train_logits.shape}")
            logging.info("Loaded logits for all splits")
        except FileNotFoundError:
            logging.warning("Logits files not found, will use features directly")
            
        # Fix features shape - remove the extra dimension if present
        if self.train_features.ndim == 3 and self.train_features.shape[1] == 1:
            logging.info("Reshaping features to remove extra dimension")
            self.train_features = self.train_features.squeeze(1)
            self.cal_features = self.cal_features.squeeze(1)
            self.test_features = self.test_features.squeeze(1)
            
        logging.info(f"Features shape: {self.train_features.shape}")
    
    def get_dataloaders(self):
        """
        Create PyTorch DataLoaders for train, calibration, and test sets.
        
        Returns:
            Dictionary of DataLoaders for each split
        """
        batch_size = self.config.get('batch_size', 128)
        num_workers = 4
        
        # Check if we have logits available
        has_logits = hasattr(self, 'train_logits') and hasattr(self, 'cal_logits') and hasattr(self, 'test_logits')
        
        # Create datasets
        if has_logits:
            logging.info("Using pre-extracted logits for dataloaders")
            train_dataset = TensorDataset(
                torch.tensor(self.train_logits, dtype=torch.float32),
                torch.tensor(self.train_labels, dtype=torch.long)
            )
            
            cal_dataset = TensorDataset(
                torch.tensor(self.cal_logits, dtype=torch.float32),
                torch.tensor(self.cal_labels, dtype=torch.long)
            )
            
            test_dataset = TensorDataset(
                torch.tensor(self.test_logits, dtype=torch.float32),
                torch.tensor(self.test_labels, dtype=torch.long)
            )
        else:
            logging.info("Using features for dataloaders")
            train_dataset = TensorDataset(
                torch.tensor(self.train_features, dtype=torch.float32),
                torch.tensor(self.train_labels, dtype=torch.long)
            )
            
            cal_dataset = TensorDataset(
                torch.tensor(self.cal_features, dtype=torch.float32),
                torch.tensor(self.cal_labels, dtype=torch.long)
            )
            
            test_dataset = TensorDataset(
                torch.tensor(self.test_features, dtype=torch.float32),
                torch.tensor(self.test_labels, dtype=torch.long)
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'calibration': cal_loader,
            'test': test_loader
        }
    
    def get_model(self):
        """
        Create a model that returns logits with the correct shape for the metrics computation.
        
        Returns:
            PyTorch model that returns logits with shape [batch_size, num_classes]
        """
        # Check if we have pre-extracted logits
        has_logits = hasattr(self, 'train_logits') and hasattr(self, 'cal_logits') and hasattr(self, 'test_logits')
        
        class KineticsModel(torch.nn.Module):
            def __init__(self, num_classes=400, use_logits=False, feature_dim=768):
                super().__init__()
                self.num_classes = num_classes
                self.use_logits = use_logits
                
                # If not using pre-extracted logits, create a simple linear layer
                # to map features to logits
                if not use_logits:
                    self.classifier = torch.nn.Linear(feature_dim, num_classes)
                    # Initialize with small weights
                    torch.nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
                    torch.nn.init.zeros_(self.classifier.bias)
                
            def forward(self, x):
                batch_size = x.size(0)
                
                # If we're using pre-extracted logits, the input should already be logits
                # with shape [batch_size, num_classes]
                if self.use_logits:
                    # If x is already the right shape, return it directly
                    if x.shape[1] == self.num_classes:
                        return x
                    
                    # If x has an extra dimension, remove it
                    if x.dim() == 3 and x.shape[1] == 1:
                        x = x.squeeze(1)
                        return x
                
                # Otherwise, we need to handle feature inputs
                # The input x has shape [batch_size, feature_dim] or [batch_size, 1, feature_dim]
                
                # First, ensure x has the right shape
                if x.dim() == 3:
                    # If x is [batch_size, 1, feature_dim], reshape to [batch_size, feature_dim]
                    x = x.squeeze(1)
                
                # Use the classifier to map features to logits
                return self.classifier(x)
        
        model = KineticsModel(
            num_classes=self._num_classes, 
            use_logits=has_logits,
            feature_dim=self.feature_dim
        )
        return model.to(self.config['device']) 