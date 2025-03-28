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
            logits: Tensor of logits
            targets: Tensor of target indices
        """
        self.logits = logits
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # For binary classification with existing core metrics,
        # we need to format our data as [batch_size, 2] tensor
        # where each row is [negative_score, positive_score]
        logit = self.logits[idx].item()  # Get scalar value
        target = self.targets[idx].item()  # Get scalar value
        
        # Create a 2-class tensor to be compatible with core metrics
        # Format: [negative_score, positive_score]
        two_class = torch.zeros(2, dtype=torch.float32)
        
        # If label is 1 (correct), set second position to logit value
        # If label is 0 (incorrect), set first position to logit value
        if target == 1:
            two_class[1] = logit        # Set positive score
            two_class[0] = 1.0 - logit  # Set negative score
        else:
            two_class[0] = logit        # Set negative score
            two_class[1] = 1.0 - logit  # Set positive score
        
        # Return 2-element tensor and label
        # The target is the index of the correct class (0 or 1)
        return two_class, target

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
        self.scoring_dim = self.config['scoring_function'].get('input_dim', 2)
        
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
        
        # Process the logits and restructure data for the MLP training
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
                    
                    # Get logits for options A, B, C, D
                    option_logits = logits[:4]
                    
                    # Add pairs of (logit, is_correct) for each option
                    for i, logit in enumerate(option_logits):
                        # Use a float value instead of tensor to avoid numpy warnings
                        logit_value = float(logit)
                        
                        # Normalize logit to 0-1 range if needed
                        if logit_value > 1.0 or logit_value < 0.0:
                            logit_value = 1.0 / (1.0 + np.exp(-logit_value))  # Sigmoid
                        
                        # Label is 1 if this is the correct answer, 0 otherwise
                        # This is now the target class (0 or 1) rather than a binary label
                        # so that the core metrics can access scores using indexing
                        label = 1 if i == answer_idx else 0
                        
                        self.processed_logits.append(logit_value)  # Single scalar value
                        self.processed_labels.append(label)
        
        # Convert to numpy arrays - make sure to properly handle the array shape
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
        Return a dummy model that expands the inputs to match the scoring function.
        
        Returns:
            torch.nn.Module: Dummy model that reshapes and returns input logits
        """
        # Get the scoring function input dimension
        input_dim = 1  # Fixed to match the scoring function input_dim
        
        class DummyVLMModel(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.input_dim = input_dim
            
            def forward(self, x):
                # Simply pass through the input since formatting is done in the dataset
                return x
        
        model = DummyVLMModel(input_dim)
        logging.info(f"Using dummy model for VLM logits with input_dim={input_dim}")
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

def run_dataset(config, dataset_name, scoring_name="LogMargin"):
    """
    Run evaluation for VLM dataset using the specified scoring function
    
    Args:
        config: Global configuration
        dataset_name: Name of the dataset (e.g., 'vlm')
        scoring_name: Name of the scoring function
        
    Returns:
        Results dictionary
    """
    from src.core import ScoringFunction, ScoringFunctionTrainer
    from src.utils import set_seed
    import torch
    
    try:
        logging.info(f"=== Starting evaluation for VLM dataset using {scoring_name} scoring function ===")
        
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
        
        # Initialize scoring function with appropriate input dimension
        # Logits are usually of shape (num_options,) and we're scoring each option
        scoring_fn = ScoringFunction(
            input_dim=1,  # Single logit value per option
            hidden_dims=config['scoring_function']['hidden_dims'],
            output_dim=1,
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