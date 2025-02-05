# src/utils/config.py

import torch
import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, config_path: str = "configs/base_config.yml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Base configuration
        self.base_dir = config['base']['base_dir']
        self.data_dir = os.path.join(self.base_dir, config['base']['data_dir'])
        self.model_dir = os.path.join(self.base_dir, config['base']['model_dir'])
        self.plot_dir = os.path.join(self.base_dir, config['base']['plot_dir'])
        self.experiment_dir = os.path.join(self.base_dir, config['base']['experiment_dir'])
        
        # Create directories
        for directory in [self.data_dir, self.model_dir, self.plot_dir, self.experiment_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Model configuration
        self.num_classes = config['model']['num_classes']
        self.hidden_dims = config['model']['scoring_function']['hidden_dims']
        self.target_mean = config['model']['scoring_function']['target_mean']
        self.target_std = config['model']['scoring_function']['target_std']
        self.l2_lambda = config['model']['scoring_function']['l2_lambda']
        self.dropout_rate = config['model']['scoring_function']['dropout_rate']
        
        # Training configuration
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        self.lambda1 = config['training']['lambda1']
        self.lambda2 = config['training']['lambda2']
        self.target_coverage = config['training']['target_coverage']
        self.size_alpha = config['training']['size_alpha']
        self.margin_alpha = config['training']['margin_alpha']
        self.margin = config['training']['margin']
        self.target_size = config['training']['target_size']
        
        # Optimizer configuration
        self.optimizer_config = config['optimizer']
        
        # Paths
        self.vit_model_path = os.path.join(self.model_dir, config['paths']['vit_model'])
        self.scoring_model_path = os.path.join(self.model_dir, config['paths']['scoring_model'])
        self.log_file = os.path.join(self.base_dir, config['paths']['log_file'])
        
        # Visualization
        self.vis_config = config['visualization']
        
        # Set device
        self.device = torch.device(config['base']['device'] if torch.cuda.is_available() else 'cpu')
        
    def save(self, path: str) -> None:
        """Save the current configuration to a YAML file.
        
        Args:
            path: Path where to save the configuration
        """
        config = {
            'base': {
                'base_dir': self.base_dir,
                'data_dir': os.path.relpath(self.data_dir, self.base_dir),
                'model_dir': os.path.relpath(self.model_dir, self.base_dir),
                'plot_dir': os.path.relpath(self.plot_dir, self.base_dir),
                'experiment_dir': os.path.relpath(self.experiment_dir, self.base_dir),
                'device': 'cuda' if self.device.type == 'cuda' else 'cpu'
            },
            'model': {
                'num_classes': self.num_classes,
                'scoring_function': {
                    'hidden_dims': self.hidden_dims,
                    'target_mean': self.target_mean,
                    'target_std': self.target_std,
                    'l2_lambda': self.l2_lambda,
                    'dropout_rate': self.dropout_rate
                }
            },
            'training': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'target_coverage': self.target_coverage,
                'size_alpha': self.size_alpha,
                'margin_alpha': self.margin_alpha,
                'margin': self.margin,
                'target_size': self.target_size
            },
            'optimizer': self.optimizer_config,
            'paths': {
                'vit_model': os.path.relpath(self.vit_model_path, self.model_dir),
                'scoring_model': os.path.relpath(self.scoring_model_path, self.model_dir),
                'log_file': os.path.relpath(self.log_file, self.base_dir)
            },
            'visualization': self.vis_config
        }
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)