# src/models/scoring_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[256, 128], output_dim=None, config=None):
        """
        Learnable Scoring Function for Conformal Prediction.
        
        Core Algorithm:
        1. Takes softmax probabilities from base model
        2. Learns to score each class optimally for conformal prediction
        3. Pushes true classes toward low scores (close to 0)
        4. Pushes false classes toward high scores (1, 2, 5, etc.)
        5. Maintains 90% coverage while minimizing set size
        
        Args:
            input_dim: Number of classes
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
        
        # Simple but effective feature set: only essential features
        # Core features: prob, rank, relative_to_max, log_prob, entropy
        feature_dim = 5
        
        # Ultra-stable MLP architecture - much smaller and simpler
        hidden_dims = [64, 32]  # Much smaller network
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)  # Higher dropout for stability
            ])
            prev_dim = hidden_dim
        
        # Output layer - single score per class
        layers.append(nn.Linear(prev_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
        # L2 regularization
        self.l2_lambda = config['scoring_function'].get('l2_lambda', 0.01)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable learning"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_features(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Extract essential features for scoring function.
        
        Key insight: Keep it simple - let the MLP learn the complex patterns.
        Only provide the most fundamental features that help distinguish
        true vs false classes.
        
        Args:
            probs: [B, C] probability distributions
            
        Returns:
            features: [B, C, 5] tensor of features per class
        """
        batch_size, num_classes = probs.shape
        device = probs.device
        
        # 1. Raw probability (most important signal)
        prob_feature = probs.unsqueeze(-1)  # [B, C, 1]
        
        # 2. Rank within distribution (normalized)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)
        ranks = torch.zeros_like(probs)
        for b in range(batch_size):
            ranks[b, sorted_indices[b]] = torch.arange(num_classes, device=device).float()
        ranks = ranks / (num_classes - 1)  # Normalize to [0, 1]
        rank_feature = ranks.unsqueeze(-1)  # [B, C, 1]
        
        # 3. Relative to max probability
        p_max = probs.max(dim=1, keepdim=True)[0]
        relative_feature = (probs / (p_max + 1e-8)).unsqueeze(-1)  # [B, C, 1]
        
        # 4. Log probability (for gradient stability)
        log_prob_feature = torch.log(probs + 1e-8).unsqueeze(-1)  # [B, C, 1]
        
        # 5. Distribution entropy (global uncertainty)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)  # [B, 1]
        entropy_feature = entropy.unsqueeze(-1).expand(-1, num_classes, -1)  # [B, C, 1]
        
        # Concatenate all features
        features = torch.cat([
            prob_feature,
            rank_feature, 
            relative_feature,
            log_prob_feature,
            entropy_feature
        ], dim=-1)  # [B, C, 5]
        
        return features
    
    def forward(self, probs):
        """
        Pure learnable scoring function - MLP learns optimal scoring from scratch.
        
        Strategy: Let the MLP discover the best scoring function for the specific
        data distribution. No constraints or baselines - pure learning.
        """
        # Ensure input has correct shape
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        batch_size, num_classes = probs.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Extract features for MLP to learn from
        features = self.compute_features(probs)  # [B, C, 5]
        features_flat = features.view(batch_size * num_classes, -1)  # [B*C, 5]
        
        # Let MLP learn the scoring function directly
        raw_scores = self.scoring_network(features_flat)  # [B*C, 1]
        scores = raw_scores.view(batch_size, num_classes)  # [B, C]
        
        # Apply activation to ensure positive scores for conformal prediction
        # Use ReLU + small offset to ensure scores > 0
        scores = F.relu(scores) + 0.01
        
        # Optional: Add upper bound to prevent extreme scores
        scores = torch.clamp(scores, 0.01, 10.0)
        
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