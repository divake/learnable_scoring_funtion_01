"""Simple temperature-based scoring function"""

import torch
import torch.nn as nn


class SimpleScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=None, output_dim=None, config=None):
        """
        Initialize simple scoring function that learns temperature scaling of 1-p scores
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        if config is None:
            raise ValueError("config must be provided")
        
        # Get number of classes from config
        if hasattr(config, 'config'):
            config_dict = config.config
        else:
            config_dict = config
            
        if 'dataset' not in config_dict or 'num_classes' not in config_dict['dataset']:
            raise ValueError("num_classes must be specified in config['dataset']")
        
        self.num_classes = config_dict['dataset']['num_classes']
        
        # Single learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Optional: learnable bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        self.l2_lambda = config['scoring_function']['l2_lambda']
        
        # Get training dynamics from config
        dynamics = config.get('training_dynamics', {})
        self.stability_factor = dynamics.get('stability_factor', 0.01)
        
    def forward(self, x):
        """
        Forward pass through the scoring function.
        
        Args:
            x: Input tensor of shape (batch_size, num_classes) containing probability vectors
            
        Returns:
            scores: Output tensor of shape (batch_size, num_classes) containing scores for each class
        """
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, num_classes = x.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Compute temperature-scaled 1-p scores
        # Use softplus to ensure temperature is positive
        temp = torch.nn.functional.softplus(self.temperature)
        
        # Apply temperature scaling to 1-p scores
        scores = (1.0 - x) * temp + self.bias
        
        # Ensure scores are non-negative
        scores = torch.clamp(scores, min=0.0)
        
        # Add L2 regularization
        l2_reg = self.temperature ** 2 + self.bias ** 2
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Stability losses
        self.stability_loss = 0.0
        self.separation_loss = 0.0
        
        return scores
    
    def get_score_stats(self, x):
        """Get statistics about the scores for analysis."""
        scores = self.forward(x)
        return {
            'mean': scores.mean(dim=-1),
            'std': scores.std(dim=-1),
            'min': scores.min(dim=-1)[0],
            'max': scores.max(dim=-1)[0],
            'range': scores.max(dim=-1)[0] - scores.min(dim=-1)[0]
        }