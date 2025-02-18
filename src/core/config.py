"""
Configuration management for the learnable scoring function project.
Handles loading, merging, and accessing configuration files.
"""

import os
import yaml
import torch
import importlib
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and merge configuration files"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please make sure the config file exists and the path is correct.\n"
                f"Config files should be in the src/config/ directory."
            )
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle inheritance
        if 'inherit' in config:
            base_path = os.path.join(os.path.dirname(self.config_path), config['inherit'])
            if not os.path.exists(base_path):
                raise FileNotFoundError(
                    f"Base config file not found: {base_path}\n"
                    f"This file is referenced as 'inherit' in {self.config_path}"
                )
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
        """Setup and return torch device based on config or availability"""
        if 'device' in self.config:
            # Use device specified in config
            if isinstance(self.config['device'], int):
                # If device is specified as an integer (GPU index)
                if torch.cuda.is_available():
                    self.config['device'] = torch.device(f'cuda:{self.config["device"]}')
                else:
                    raise RuntimeError(f"CUDA device {self.config['device']} specified but CUDA is not available")
            else:
                # If device is specified as a string (e.g., 'cuda:0', 'cpu')
                self.config['device'] = torch.device(self.config['device'])
        else:
            # Fallback to default behavior
            self.config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __getitem__(self, key):
        """Allow dictionary-like access to config"""
        return self.config[key]
    
    def get(self, key, default=None):
        """Safe dictionary-like access with default"""
        return self.config.get(key, default) 