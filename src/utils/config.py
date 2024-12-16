# src/utils/config.py

import torch
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    base_dir: str = '/ssd1/divake/learnable_scoring_fn'
    data_dir: str = 'data'
    model_dir: str = 'models'
    plot_dir: str = 'plots'
    
    # Model parameters
    num_classes: int = 10
    hidden_dims: list = None  # Will be set to [64, 32] by default
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.001
    lambda1: float = 1.0  # Coverage loss weight
    lambda2: float = 1.0  # Set size loss weight
    target_coverage: float = 0.9
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]
        
        # Create full paths
        self.data_dir = os.path.join(self.base_dir, self.data_dir)
        self.model_dir = os.path.join(self.base_dir, self.model_dir)
        self.plot_dir = os.path.join(self.base_dir, self.plot_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')