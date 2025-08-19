# src/models/scoring_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[256, 128], output_dim=None, config=None):
        """
        Full Softmax Learnable Scoring Function for Conformal Prediction.
        
        Core Algorithm:
        1. Takes softmax probabilities from base model
        2. For each class, MLP sees ENTIRE probability distribution  
        3. No feature engineering - pure end-to-end learning
        4. Training: true classes → low scores, false classes → high scores
        5. Simple loss: coverage + size (no ranking, no scheduling)
        
        Args:
            input_dim: Number of classes (MLP input dimension)
            hidden_dims: MLP hidden dimensions  
            config: Configuration containing training parameters
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
        
        if input_dim is None:
            input_dim = self.num_classes
            
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Pure Data-Driven MLP Architecture
        # Input: [prob_distribution + class_identity] = 2 * num_classes
        feature_dim = 2 * self.num_classes
        
        # Much simpler architecture to prevent overfitting
        if self.num_classes <= 10:  # CIFAR-10
            hidden_dims = [64, 32]
        elif self.num_classes <= 100:  # CIFAR-100  
            hidden_dims = [128, 64]  # Reduced from [512, 256, 128]
        else:  # ImageNet, complex datasets
            hidden_dims = [256, 128]  # Reduced complexity
        
        layers = []
        prev_dim = feature_dim
        
        # Strong regularization to prevent overfitting
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),  # Standard activation for reliable learning
                nn.Dropout(0.5)  # Increased from 0.2 for stronger regularization
            ])
            prev_dim = hidden_dim
        
        # Output layer - raw scores (no activation)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
        # L2 regularization - no fallback, must be explicitly defined
        if 'scoring_function' not in config or 'l2_lambda' not in config['scoring_function']:
            raise ValueError("config['scoring_function']['l2_lambda'] must be explicitly defined")
        self.l2_lambda = config['scoring_function']['l2_lambda']
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Standard weight initialization for reliable learning"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for stable gradients
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def prepare_full_softmax_input(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Prepare full softmax distribution as input for each class scoring.
        
        Key insight: Let MLP see the entire probability distribution PLUS 
        which class it's scoring. This gives class-specific context.
        
        Args:
            probs: [B, C] probability distributions
            
        Returns:
            full_context: [B, C, C+1] - for each class, provide full softmax + class indicator
        """
        batch_size, num_classes = probs.shape
        
        # For each class, provide the entire softmax distribution as context
        full_context = probs.unsqueeze(1).expand(-1, num_classes, -1)  # [B, C, C]
        
        # Add class identity indicators - this is the key fix!
        # Create one-hot encoding for which class we're scoring
        class_indicators = torch.eye(num_classes, device=probs.device, dtype=probs.dtype)  # [C, C]
        class_indicators = class_indicators.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, C]
        
        # Concatenate probability distribution with class identity
        # Now each class gets: [prob_dist, which_class_am_I]
        full_context = torch.cat([full_context, class_indicators], dim=-1)  # [B, C, C+C] = [B, C, 2C]
        
        return full_context
    
    def forward(self, probs):
        """
        Full softmax learnable scoring function.
        
        Strategy: MLP sees entire probability distribution for each class.
        Pure end-to-end learning without feature engineering.
        """
        # Ensure input has correct shape
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        batch_size, num_classes = probs.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Prepare full softmax context for each class
        full_context = self.prepare_full_softmax_input(probs)  # [B, C, C]
        context_flat = full_context.reshape(batch_size * num_classes, -1)  # [B*C, C]
        
        # MLP processes full probability distribution for each class
        raw_scores = self.scoring_network(context_flat)  # [B*C, 1]
        scores = raw_scores.view(batch_size, num_classes)  # [B, C]
        
        # Pure data-driven scoring: let the MLP learn naturally
        # Raw scores - no activation to allow full range of values
        # scores = scores  # Keep raw MLP outputs
        
        # L2 regularization
        if self.training:
            l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
            self.l2_reg = self.l2_lambda * l2_reg
        else:
            self.l2_reg = 0.0
        
        return scores
    
    def get_score_stats(self, x):
        """
        Get statistics about the scores for analysis.
        
        Returns mean, std, min, max of scores across classes.
        """
        scores = self.forward(x)
        return {
            'mean': scores.mean(dim=-1),
            'std': scores.std(dim=-1),
            'min': scores.min(dim=-1)[0],
            'max': scores.max(dim=-1)[0],
            'range': scores.max(dim=-1)[0] - scores.min(dim=-1)[0]
        }