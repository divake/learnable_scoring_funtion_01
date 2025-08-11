# src/models/scoring_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[512, 256], output_dim=None, config=None):
        """
        Initialize distribution-aware learnable scoring function.
        
        This function learns to score classes based on their relative relationships 
        within the distribution, using rank-based features and context-aware adjustments.
        
        Args:
            input_dim: Number of classes (dimension of probability vector)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (same as input_dim for scores per class)
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
        
        if input_dim is None:
            input_dim = self.num_classes
        if output_dim is None:
            output_dim = self.num_classes
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dims[0] if hidden_dims else 512
        
        # Feature dimensions for distribution-aware scoring
        # Per-class: 12 features, Full context: 6 * num_classes
        feature_dim = 12 + 6 * self.num_classes
        
        # Main scoring network
        self.scoring_network = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)  # Output: single score per class
        )
        
        # Context-aware adjustment network
        self.context_network = nn.Sequential(
            nn.Linear(6, 64),  # 6 distribution features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: [scale, shift, temperature]
        )
        
        self.l2_lambda = config['scoring_function'].get('l2_lambda', 0.005)
        
        # Get training dynamics from config
        dynamics = config.get('training_dynamics', {})
        self.stability_factor = dynamics.get('stability_factor', 0.05)
        self.separation_factor = dynamics.get('separation_factor', 0.5)
        self.perturbation_noise = dynamics.get('perturbation_noise', 0.01)
        self.xavier_init_gain = dynamics.get('xavier_init_gain', 0.5)
        
        # Initialize weights for better learning
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to encourage diverse scores"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for output layers
                if module.out_features == 1 or module.out_features == 3:
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_features(self, probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all relative and distributional features
        
        Args:
            probs: [B, C] probability distributions
            
        Returns:
            Dictionary of features
        """
        batch_size, num_classes = probs.shape
        device = probs.device
        
        # 1. Rank-based features (1 is highest prob, num_classes is lowest)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)
        ranks = torch.zeros_like(probs)
        for b in range(batch_size):
            ranks[b, sorted_indices[b]] = torch.arange(num_classes, device=device).float() + 1
        ranks = ranks / num_classes  # Normalize to [0, 1]
        
        # 2. Relative probability features
        p_max = probs.max(dim=1, keepdim=True)[0]
        p_mean = probs.mean(dim=1, keepdim=True)
        p_std = probs.std(dim=1, keepdim=True)
        
        relative_probs = probs / (p_max + 1e-8)  # Relative to max
        deviations = (probs - p_mean) / (p_std + 1e-8)  # Standardized
        
        # 3. Log probabilities (for better gradient flow with small probs)
        log_probs = torch.log(probs + 1e-8)
        
        # 4. Top-k indicators (is this class in top-k?)
        top_k = 5
        _, top_indices = torch.topk(probs, min(top_k, num_classes), dim=1)
        is_top_k = torch.zeros_like(probs)
        is_top_k.scatter_(1, top_indices, 1)
        
        # 5. Distribution-level features (same for all classes in a sample)
        entropy = -(probs * log_probs).sum(dim=1)  # Shannon entropy
        max_prob = p_max.squeeze(1)
        top5_mass = probs.topk(min(5, num_classes), dim=1)[0].sum(dim=1)
        top10_mass = probs.topk(min(10, num_classes), dim=1)[0].sum(dim=1)
        effective_classes = 1.0 / (probs ** 2).sum(dim=1)  # Inverse Simpson index
        prob_range = probs.max(dim=1)[0] - probs.min(dim=1)[0]
        
        distribution_features = torch.stack([
            entropy,
            max_prob,
            top5_mass,
            top10_mass,
            effective_classes / num_classes,  # Normalize
            prob_range
        ], dim=1)  # [B, 6]
        
        return {
            'probs': probs,
            'ranks': ranks,
            'relative_probs': relative_probs,
            'deviations': deviations,
            'log_probs': log_probs,
            'is_top_k': is_top_k,
            'distribution_features': distribution_features
        }
    
    def forward(self, probs):
        """
        Compute relative scores for given probability distributions
        
        Args:
            probs: Input tensor of shape (batch_size, num_classes) containing probability vectors
            
        Returns:
            scores: Tensor of shape (batch_size, num_classes) with scores for each class
        """
        # Ensure input has correct shape
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        batch_size, num_classes = probs.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Extract all features
        features = self.compute_features(probs)
        
        # Expand distribution features to match per-class shape
        dist_features_expanded = features['distribution_features'].unsqueeze(1).expand(
            batch_size, num_classes, -1
        )  # [B, C, 6]
        
        # Concatenate all features for each class
        per_class_features = torch.cat([
            features['probs'].unsqueeze(-1),
            features['ranks'].unsqueeze(-1),
            features['relative_probs'].unsqueeze(-1),
            features['deviations'].unsqueeze(-1),
            features['log_probs'].unsqueeze(-1),
            features['is_top_k'].unsqueeze(-1),
            dist_features_expanded
        ], dim=-1)  # [B, C, 12]
        
        # Flatten batch of features
        per_class_features = per_class_features.view(batch_size * num_classes, -1)
        
        # Replicate each class's features with full distribution context
        probs_repeated = probs.unsqueeze(1).expand(-1, num_classes, -1)  # [B, C, C]
        probs_context = probs_repeated.reshape(batch_size * num_classes, num_classes)
        
        # Combine per-class features with full context
        full_features = torch.cat([
            per_class_features,
            probs_context,
            features['ranks'].unsqueeze(1).expand(-1, num_classes, -1).reshape(batch_size * num_classes, num_classes),
            features['relative_probs'].unsqueeze(1).expand(-1, num_classes, -1).reshape(batch_size * num_classes, num_classes),
            features['deviations'].unsqueeze(1).expand(-1, num_classes, -1).reshape(batch_size * num_classes, num_classes),
            features['log_probs'].unsqueeze(1).expand(-1, num_classes, -1).reshape(batch_size * num_classes, num_classes),
            features['is_top_k'].unsqueeze(1).expand(-1, num_classes, -1).reshape(batch_size * num_classes, num_classes)
        ], dim=-1)
        
        # Score each class with full context
        scores = self.scoring_network(full_features)  # [B*C, 1]
        scores = scores.view(batch_size, num_classes)  # [B, C]
        
        # Apply context-aware adjustments
        context_params = self.context_network(features['distribution_features'])
        scale, shift, temperature = context_params[:, 0:1], context_params[:, 1:2], context_params[:, 2:3]
        
        # Apply adjustments
        scale = torch.sigmoid(scale) * 2  # Scale between 0 and 2
        temperature = torch.sigmoid(temperature) * 2 + 0.1  # Temperature between 0.1 and 2.1
        
        scores = (scores * scale + shift) / temperature
        
        # Ensure scores are positive and well-behaved
        scores = F.softplus(scores) + 0.001
        
        # Add strong bias based on rank to ensure discrimination
        rank_penalty = features['ranks'] * 5.0  # Strong penalty for low-probability classes
        scores = scores + rank_penalty
        
        # Add L2 regularization
        if self.training:
            l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
            self.l2_reg = self.l2_lambda * l2_reg
            self.stability_loss = 0.0
        else:
            self.l2_reg = 0.0
            self.stability_loss = 0.0
        
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