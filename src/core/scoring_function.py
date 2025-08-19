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
        
        # NEW: Full Distribution + Global Features Architecture
        # Input: [all_probabilities + global_features] = C + 5
        # Global features: entropy, max_prob, top2_gap, top3_mass, gini
        feature_dim = self.num_classes + 5  # Much more compact than 2*C
        
        # Simpler, more focused architecture for distribution-level learning
        if self.num_classes <= 10:  # CIFAR-10
            hidden_dims = [128, 64]
        elif self.num_classes <= 100:  # CIFAR-100  
            hidden_dims = [256, 128]  # Can afford larger due to compact input
        else:  # ImageNet, complex datasets
            hidden_dims = [512, 256]  # Rich features enable larger networks
        
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
        
        # Output layer - C scores (one per class, no activation)
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
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
    
    def compute_global_features(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute global features from the full softmax distribution.
        
        These features capture uncertainty patterns that help the MLP learn
        meaningful discrimination beyond simple probability ranking.
        
        Args:
            probs: [B, C] probability distributions
            
        Returns:
            features: [B, num_features] global distribution features
        """
        batch_size, num_classes = probs.shape
        
        # 1. Entropy: -Σ(pᵢ × log(pᵢ))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B]
        
        # 2. Max probability
        max_prob, _ = torch.max(probs, dim=1)  # [B]
        
        # 3. Top-2 gap: p_max - p_second_max
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        top2_gap = sorted_probs[:, 0] - sorted_probs[:, 1]  # [B]
        
        # 4. Top-3 mass: sum of top-3 probabilities
        top3_mass = torch.sum(sorted_probs[:, :3], dim=1)  # [B]
        
        # 5. Gini coefficient (distribution spread measure)
        # Sort probabilities for Gini calculation
        sorted_probs_gini, _ = torch.sort(probs, dim=1)
        n = num_classes
        indices = torch.arange(1, n + 1, device=probs.device, dtype=probs.dtype)
        gini = (2 * torch.sum(indices.unsqueeze(0) * sorted_probs_gini, dim=1) / 
                (n * torch.sum(sorted_probs_gini, dim=1)) - (n + 1) / n)  # [B]
        
        # Stack all features: [B, 5]
        features = torch.stack([entropy, max_prob, top2_gap, top3_mass, gini], dim=1)
        
        return features
    
    def prepare_distribution_input(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Prepare input using full distribution + global features approach.
        
        New design: Instead of class-specific one-hot encoding, we use
        the complete distribution context with rich uncertainty features.
        
        Args:
            probs: [B, C] probability distributions
            
        Returns:
            input_features: [B, C + num_features] combined input
        """
        # Compute global uncertainty features
        global_features = self.compute_global_features(probs)  # [B, 5]
        
        # Concatenate probability distribution with global features
        # Input: [all_probs, entropy, max_prob, top2_gap, top3_mass, gini]
        input_features = torch.cat([probs, global_features], dim=1)  # [B, C + 5]
        
        return input_features
    
    def forward(self, probs):
        """
        NEW: Full distribution + global features scoring function.
        
        Strategy: MLP sees complete probability distribution plus rich
        uncertainty features, then outputs one score per class.
        """
        # Ensure input has correct shape
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        batch_size, num_classes = probs.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # NEW: Prepare distribution input with global features
        input_features = self.prepare_distribution_input(probs)  # [B, C + 5]
        
        # MLP processes complete distribution context once to get all scores
        scores = self.scoring_network(input_features)  # [B, C]
        
        # Pure data-driven scoring: let the MLP learn naturally
        # Raw scores - no activation to allow full range of values
        # The MLP can now learn rich patterns across the entire distribution
        
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