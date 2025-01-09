# src/utils/config.py

import torch
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    # Paths
    base_dir: str = '/ssd_4TB/divake/learnable_scoring_fn'
    data_dir: str = 'data'
    model_dir: str = 'models'
    plot_dir: str = 'plots'
    
    # Model parameters
    num_classes: int = 100
    hidden_dims: List[int] = field(default_factory=lambda: [64,32])
    
    # Training parameters
    num_epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 0.0001
    lambda1: float = 0.1  # Coverage loss weight
    lambda2: float = 1.0  # Set size loss weight
    target_coverage: float = 0.9
    size_alpha: float = 0.1
    margin_alpha: float = 0.1
    
    # Model paths (will be set in post_init)
    vit_model_path: Optional[str] = None
    scoring_model_path: Optional[str] = None
    device: Optional[torch.device] = None
    
    def __post_init__(self):
        # Create full paths
        self.data_dir = os.path.abspath(os.path.join(self.base_dir, self.data_dir))
        self.model_dir = os.path.abspath(os.path.join(self.base_dir, self.model_dir))
        self.plot_dir = os.path.abspath(os.path.join(self.base_dir, self.plot_dir))
        
        # Set specific model paths
        self.vit_model_path = os.path.join(self.model_dir, 'vit_phase2_best.pth')
        self.scoring_model_path = os.path.join(self.model_dir, 'scoring_function_best.pth')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')