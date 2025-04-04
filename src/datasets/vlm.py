"""
VLM dataset module for loading Vision-Language Model logits
from pickle files and preparing them for the scoring function.
"""

import os
import logging
import pickle
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
import random
from src.datasets.base import BaseDataset

class VLMSampleDataset(TorchDataset):
    """
    PyTorch Dataset for VLM logits to be used with DataLoader
    """
    def __init__(self, logits, targets):
        """
        Initialize dataset with logits and targets
        
        Args:
            logits: Tensor of shape [num_samples, 4] containing logits for options A, B, C, D
            targets: Tensor of shape [num_samples] containing target indices (0 to 3)
        """
        self.logits = logits
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # The conformal prediction framework expects softmax probabilities of shape [batch_size, num_classes]
        # We need to return the logits directly for proper processing
        logits = self.logits[idx]  # Shape: [4]
        target = self.targets[idx].item()  # Get scalar value (0, 1, 2, or 3)
        
        return logits, target

class Dataset(BaseDataset):
    """
    VLM dataset class for loading Vision-Language Model logits
    from pickle files and preparing them for the scoring function.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLM dataset
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # Extract VLM-specific configurations
        self.vlm_dir = self.config.get('data_dir', 'data/vlm')
        self.model_name = self.config['dataset'].get('default_model', 'Yi-VL-6B')
        self.dataset_name = self.config['dataset'].get('default_dataset', 'ai2d')
        self.split_ratio = self.config['dataset'].get('split_ratio', [0.7, 0.1, 0.2])
        
        # Store the scoring function input dimension for our model
        self.scoring_dim = 4  # Changed to 4 for multiclass classification
        
        # Load the data
        self.setup()
    
    def setup(self):
        """
        Load and prepare VLM logits data
        """
        logging.info(f"Loading VLM logits for model {self.model_name} and dataset {self.dataset_name}")
        
        # Set up the file path - the pkl files are directly in the model directory, not in subdirectories
        file_path = os.path.join(self.vlm_dir, self.model_name, f"{self.dataset_name}.pkl")
        
        # Check if file exists, if not try the direct path without model subdirectory
        if not os.path.exists(file_path):
            direct_path = os.path.join(self.vlm_dir, f"{self.dataset_name}.pkl")
            if os.path.exists(direct_path):
                file_path = direct_path
                logging.info(f"Using direct path: {file_path}")
            else:
                # Try with absolute path for debugging
                abs_vlm_dir = os.path.abspath(self.vlm_dir)
                abs_file_path = os.path.join(abs_vlm_dir, self.model_name, f"{self.dataset_name}.pkl")
                if os.path.exists(abs_file_path):
                    file_path = abs_file_path
                    logging.info(f"Using absolute path: {file_path}")
                else:
                    # Check for the known working path
                    known_path = f"/ssd_4TB/divake/learnable_scoring_funtion_01/data/vlm/{self.model_name}/{self.dataset_name}.pkl"
                    if os.path.exists(known_path):
                        file_path = known_path
                        logging.info(f"Using known fixed path: {file_path}")
                    else:
                        logging.error(f"Could not find data file. Tried: {file_path}, {direct_path}, {abs_file_path}, {known_path}")
                        raise FileNotFoundError(f"VLM logits file not found: {file_path}")
        
        # Load the pickle file
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"Loaded {len(data)} samples from {file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"VLM logits file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading VLM logits: {str(e)}")
        
        # Process the data for training
        self._process_data(data)
        
        # Create splits (train/cal/test)
        self._create_splits()
        
        # Create dataloaders
        self._create_dataloaders()
        
        logging.info(f"VLM dataset prepared with {len(self.train_logits)} training samples, "
                    f"{len(self.cal_logits)} calibration samples, and "
                    f"{len(self.test_logits)} test samples")
    
    def _process_data(self, data):
        """
        Process the loaded data to prepare logits and labels for training
        
        Args:
            data: List of dictionaries containing logits and answers
        """
        # Extract logits and answers from the data
        self.all_samples = []
        
        # Process the logits and restructure data for the scoring function training
        self.processed_logits = []
        self.processed_labels = []
        
        # Process each sample
        for item in data:
            if 'logits' in item and 'answer' in item:
                logits = item['logits']
                answer = item['answer']
                
                # For multiple-choice data with A, B, C, D options 
                # The first 4 values in the logits array correspond to these options
                if isinstance(logits, np.ndarray) and len(logits) >= 4:
                    # Get the answer index (0 for A, 1 for B, etc.)
                    if len(answer) == 1 and 'A' <= answer <= 'Z':
                        answer_idx = ord(answer) - ord('A')
                    else:
                        # Skip items without clear A/B/C/D answers
                        continue
                    
                    # Extract the first 4 logits corresponding to options A, B, C, D
                    option_logits = logits[:4]
                    
                    # Apply softmax normalization to convert raw logits to probabilities
                    exp_logits = np.exp(option_logits - np.max(option_logits))
                    option_logits = exp_logits / exp_logits.sum()
                    
                    # Add the sample with all 4 options and the correct label
                    self.processed_logits.append(option_logits)
                    self.processed_labels.append(answer_idx)
        
        # Convert to numpy arrays
        self.processed_logits = np.array(self.processed_logits, dtype=np.float32)
        self.processed_labels = np.array(self.processed_labels, dtype=np.int64)
        
        logging.info(f"Processed {len(self.processed_logits)} total samples from {len(data)} questions")
        logging.info(f"Logits shape: {self.processed_logits.shape}")
        logging.info(f"Labels shape: {self.processed_labels.shape}")
        
        # Store original data for reference
        self.raw_data = data
    
    def _create_splits(self):
        """
        Create train/cal/test splits from the processed data
        """
        # Get total number of samples
        total_samples = len(self.processed_logits)
        
        # Calculate split sizes
        train_size = int(self.split_ratio[0] * total_samples)
        cal_size = int(self.split_ratio[1] * total_samples)
        test_size = total_samples - train_size - cal_size
        
        # Create indices for random splitting
        indices = list(range(total_samples))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:train_size]
        cal_indices = indices[train_size:train_size + cal_size]
        test_indices = indices[train_size + cal_size:]
        
        # Split data
        self.train_logits = self.processed_logits[train_indices]
        self.train_labels = self.processed_labels[train_indices]
        
        self.cal_logits = self.processed_logits[cal_indices]
        self.cal_labels = self.processed_labels[cal_indices]
        
        self.test_logits = self.processed_logits[test_indices]
        self.test_labels = self.processed_labels[test_indices]
    
    def _create_dataloaders(self):
        """
        Create PyTorch DataLoaders for train/cal/test splits
        """
        # Convert numpy arrays to torch tensors
        train_logits = torch.tensor(self.train_logits, dtype=torch.float32)
        train_labels = torch.tensor(self.train_labels, dtype=torch.long)
        
        cal_logits = torch.tensor(self.cal_logits, dtype=torch.float32)
        cal_labels = torch.tensor(self.cal_labels, dtype=torch.long)
        
        test_logits = torch.tensor(self.test_logits, dtype=torch.float32)
        test_labels = torch.tensor(self.test_labels, dtype=torch.long)
        
        # Create datasets
        train_dataset = VLMSampleDataset(train_logits, train_labels)
        cal_dataset = VLMSampleDataset(cal_logits, cal_labels)
        test_dataset = VLMSampleDataset(test_logits, test_labels)
        
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
        For VLM datasets, we don't need a model as we're using precomputed logits.
        Return a dummy model that properly reshapes the input logits to be compatible
        with the conformal scoring framework.
        
        Returns:
            torch.nn.Module: Dummy model that passes through input logits
        """
        class DummyVLMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                # The conformal scoring functions expect softmax probabilities
                # So we return x directly since we already normalized in _process_data
                return x
        
        model = DummyVLMModel()
        logging.info(f"Using dummy model for VLM logits with 4-way classification")
        return model
    
    def get_dataloaders(self):
        """
        Get train, calibration, and test data loaders
        
        Returns:
            Dict containing 'train', 'calibration', and 'test' dataloaders
        """
        return {
            'train': self.train_loader,
            'calibration': self.cal_loader,
            'test': self.test_loader
        }

def run_dataset(config, dataset_name):
    """
    Run evaluation for VLM dataset using the learnable scoring function
    
    Args:
        config: Global configuration
        dataset_name: Name of the dataset (e.g., 'vlm')
        
    Returns:
        Results dictionary
    """
    from src.core import ScoringFunction, ScoringFunctionTrainer
    from src.utils import set_seed
    import torch
    
    try:
        logging.info(f"=== Starting evaluation for VLM dataset with learnable scoring function ===")
        
        # Get dataset-specific configuration
        dataset_config = config.copy()
        dataset_config['dataset']['name'] = dataset_name
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create dataset
        dataset = Dataset(dataset_config)
        dataloaders = dataset.get_dataloaders()
        
        # Get the dummy model (for VLM we use precomputed logits, no actual model needed)
        model = dataset.get_model()
        
        # Initialize scoring function with appropriate input dimension for compatibility with core/metrics.py
        scoring_fn = ScoringFunction(
            input_dim=1,  # Changed to 1 to match the expected input in core/metrics.py
            hidden_dims=config['scoring_function']['hidden_dims'],
            output_dim=1,  # Changed to 1 to match the expected output in core/metrics.py
            config=config
        ).to(config['device'])
        
        # Initialize trainer
        trainer = ScoringFunctionTrainer(
            base_model=model,
            scoring_fn=scoring_fn,
            train_loader=dataloaders['train'],
            cal_loader=dataloaders['calibration'],
            test_loader=dataloaders['test'],
            device=config['device'],
            config=config
        )
        
        # Training loop
        results = trainer.train(
            num_epochs=config.get('num_epochs', 50),
            target_coverage=config.get('target_coverage', 0.9),
            tau_config=config.get('tau', {'min': 0.1, 'max': 0.9, 'window_size': 5}),
            set_size_config=config.get('set_size', {'target': 1, 'max': 5.0, 'margin': 5}),
            save_dir=config.get('model_dir', 'models/vlm'),
            plot_dir=config.get('plot_dir', 'plots/vlm')
        )
        
        return results
        
    except Exception as e:
        logging.error(f"Error in VLM dataset evaluation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            'error': str(e),
            'status': 'failed'
        } 