# src/utils/config.py

import os
import yaml
import torch
from dataclasses import dataclass
from typing import Dict, Any
import importlib

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configuration files"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle inheritance
        if 'inherit' in config:
            base_path = os.path.join(os.path.dirname(self.config_path), config['inherit'])
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)
            # Merge configs (config overrides base_config)
            merged = self._merge_configs(base_config, config)
            return merged
            
        return config
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge two config dictionaries"""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def get_dataset_class(self):
        """Import and return the appropriate dataset class"""
        dataset_name = self.config['dataset']['name']
        try:
            # Try importing from src.datasets first
            module = importlib.import_module(f'src.datasets.{dataset_name}')
            return module.Dataset
        except ImportError:
            # Fallback to direct datasets import
            module = importlib.import_module(f'datasets.{dataset_name}')
            return module.Dataset
    
    def setup_paths(self):
        """Create necessary directories"""
        paths = ['data_dir', 'model_dir', 'plot_dir', 'log_dir']
        for path in paths:
            if path in self.config:
                full_path = os.path.join(self.config['base_dir'], self.config[path])
                os.makedirs(full_path, exist_ok=True)
                # Update config with full path
                self.config[path] = full_path
    
    def setup_device(self):
        """Setup and return torch device"""
        self.config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __getitem__(self, key):
        """Allow dictionary-like access to config"""
        return self.config[key]
    
    def get(self, key, default=None):
        """Safe dictionary-like access with default"""
        return self.config.get(key, default)

@dataclass
class Config:
    # Paths
    base_dir: str = '/ssd_4TB/divake/learnable_scoring_funtion_01'
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