# src/models/scoring_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[256, 128], output_dim=None, config=None):
        """
        Class-Specific Learnable Scoring Function for Conformal Prediction.
        
        Core Algorithm:
        1. Takes softmax probabilities from base model
        2. For each class, MLP sees CLASS-SPECIFIC features (not entire distribution)
        3. Class-aware feature engineering for discrimination
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
        
        # NEW: Class-Specific Architecture
        # Input per class: [class_prob, rank, gap_to_max, is_top1, is_top3, is_top5, entropy, max_prob]
        # This is MUCH smaller and more focused than previous C+5 features
        feature_dim = 8  # 8 class-specific features
        
        # Smaller, more efficient architecture since we have focused features
        if self.num_classes <= 10:  # CIFAR-10
            hidden_dims = [32, 16]
        elif self.num_classes <= 100:  # CIFAR-100  
            hidden_dims = [64, 32]  # Much smaller than before
        else:  # ImageNet, complex datasets
            hidden_dims = [128, 64]  # Still smaller due to focused features
        
        layers = []
        prev_dim = feature_dim
        
        # Strong regularization to prevent overfitting
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),  # Standard activation for reliable learning
                nn.Dropout(0.3)  # Reduced from 0.5 since we have smaller network
            ])
            prev_dim = hidden_dim
        
        # Output layer - single score (no activation)
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
    
    def prepare_class_specific_features(self, probs: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Prepare class-specific features for a single class.
        
        Each class gets unique features based on its position in the distribution,
        allowing the MLP to learn class-aware scoring patterns.
        
        Args:
            probs: [B, C] probability distributions
            class_idx: Index of the class to extract features for
            
        Returns:
            class_features: [B, 8] features specific to this class
        """
        batch_size, num_classes = probs.shape
        
        # 1. Class probability
        class_prob = probs[:, class_idx:class_idx+1]  # [B, 1]
        
        # 2. Rank of this class (1 = highest probability)
        # Efficient rank computation using argsort twice
        sorted_indices = torch.argsort(probs, dim=1, descending=True)
        ranks = torch.zeros_like(probs)
        # Use scatter to assign ranks efficiently
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(sorted_indices)
        rank_values = torch.arange(1, num_classes + 1).unsqueeze(0).expand_as(sorted_indices).to(probs.device)
        ranks[batch_indices, sorted_indices] = rank_values.float()
        class_rank = ranks[:, class_idx:class_idx+1] / num_classes  # Normalize by num_classes
        
        # 3. Gap to maximum probability
        max_prob, _ = torch.max(probs, dim=1, keepdim=True)
        gap_to_max = max_prob - class_prob  # [B, 1]
        
        # 4-6. Binary indicators for top-k membership
        is_top1 = (class_rank <= 1.0/num_classes).float()  # [B, 1]
        is_top3 = (class_rank <= 3.0/num_classes).float()  # [B, 1]
        is_top5 = (class_rank <= 5.0/num_classes).float()  # [B, 1]
        
        # 7-8. Global context features (same for all classes but provides context)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)  # [B, 1]
        max_prob_global = max_prob  # [B, 1]
        
        # Concatenate all features: [B, 8]
        class_features = torch.cat([
            class_prob,      # How confident is this class?
            class_rank,      # Where does it rank?
            gap_to_max,      # How far from the best?
            is_top1,         # Binary indicators
            is_top3,
            is_top5,
            entropy,         # Global uncertainty
            max_prob_global  # Global confidence
        ], dim=1)
        
        return class_features
    
    def forward(self, probs):
        """
        CLASS-SPECIFIC scoring function.
        
        Strategy: Process each class separately with its unique features,
        allowing the MLP to learn discriminative scoring patterns.
        """
        # Ensure input has correct shape
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        batch_size, num_classes = probs.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Process each class separately to get class-specific scores
        all_scores = []
        
        for class_idx in range(num_classes):
            # Extract class-specific features for this class
            class_features = self.prepare_class_specific_features(probs, class_idx)  # [B, 8]
            
            # Get score for this specific class
            class_score = self.scoring_network(class_features)  # [B, 1]
            
            all_scores.append(class_score)
        
        # Concatenate all class scores: [B, C]
        scores = torch.cat(all_scores, dim=1)
        
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