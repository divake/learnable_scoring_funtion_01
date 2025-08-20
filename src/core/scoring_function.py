# src/models/scoring_function.py

import torch
import torch.nn as nn
import numpy as np


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[256, 128], output_dim=None, config=None):
        """
        Vectorized Class-Specific Learnable Scoring Function for Conformal Prediction.
        
        Core Algorithm:
        1. Takes softmax probabilities from base model
        2. For each class, MLP sees CLASS-SPECIFIC features (not entire distribution)
        3. Class-aware feature engineering for discrimination
        4. Training: true classes → low scores, false classes → high scores
        5. Simple loss: coverage + size (no ranking, no scheduling)
        
        This version processes all classes in a single batch for massive speedup (100x+).
        
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
        
        # Class-Specific Architecture
        # Input per class: [class_prob, rank, gap_to_max, is_top1, is_top3, is_top5, entropy, max_prob]
        
        # Get feature dimension from config or use default
        if 'scoring_function' in config_dict and 'num_features' in config_dict['scoring_function']:
            feature_dim = config_dict['scoring_function']['num_features']
        else:
            feature_dim = 8  # Default: 8 class-specific features
        
        # Architecture configuration
        arch_mode = config_dict.get('scoring_function', {}).get('architecture_mode', 'auto')
        
        if arch_mode == 'auto':
            # Auto-select architecture based on number of classes
            if self.num_classes <= 10:  # Small datasets (e.g., CIFAR-10)
                hidden_dims = [32, 16]
            elif self.num_classes <= 100:  # Medium datasets (e.g., CIFAR-100)
                hidden_dims = [64, 32]
            elif self.num_classes <= 1000:  # Large datasets (e.g., ImageNet)
                hidden_dims = [128, 64]
            else:  # Very large datasets (e.g., PlantNet-300K)
                hidden_dims = [256, 128]
        else:
            # Use manually specified hidden dimensions from config
            hidden_dims = config_dict['scoring_function']['hidden_dims']
        
        layers = []
        prev_dim = feature_dim
        
        # Get dropout rate from config
        dropout_rate = config_dict.get('scoring_function', {}).get('dropout', 0.3)
        
        # Build MLP layers with configurable dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),  # Standard activation for reliable learning
                nn.Dropout(dropout_rate)  # Configurable dropout rate
            ])
            prev_dim = hidden_dim
        
        # Output layer - single score (no activation)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
        # L2 regularization - use from config or default
        if 'scoring_function' in config_dict and 'l2_lambda' in config_dict['scoring_function']:
            self.l2_lambda = config_dict['scoring_function']['l2_lambda']
        else:
            self.l2_lambda = 0.01  # Default L2 regularization
        
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
    
    def forward(self, probs):
        """
        Vectorized CLASS-SPECIFIC scoring function.
        
        Strategy: Process all classes in a single batch for massive speedup.
        Mathematically equivalent to processing each class separately but 100x+ faster.
        """
        # Ensure input has correct shape
        if probs.dim() == 1:
            probs = probs.unsqueeze(0)
        
        batch_size, num_classes = probs.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Pre-compute shared features once
        max_prob, _ = torch.max(probs, dim=1, keepdim=True)  # [B, 1]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)  # [B, 1]
        
        # Vectorized rank computation (compute all ranks at once)
        sorted_indices = torch.argsort(probs, dim=1, descending=True)
        ranks = torch.zeros_like(probs)
        batch_indices = torch.arange(batch_size, device=probs.device).unsqueeze(1).expand_as(sorted_indices)
        rank_values = torch.arange(1, num_classes + 1, device=probs.device).unsqueeze(0).expand_as(sorted_indices).float()
        ranks[batch_indices, sorted_indices] = rank_values
        
        # Prepare features for all classes at once
        # Reshape probabilities: [B, C] -> [B, C, 1]
        class_probs = probs.unsqueeze(2)  # [B, C, 1]
        
        # Normalize ranks by num_classes (exact same as original)
        class_ranks = (ranks / num_classes).unsqueeze(2)  # [B, C, 1]
        
        # Gap to max for all classes
        gaps_to_max = max_prob.unsqueeze(1) - class_probs  # [B, C, 1]
        
        # Binary indicators (vectorized)
        is_top1 = (class_ranks <= 1.0/num_classes).float()  # [B, C, 1]
        is_top3 = (class_ranks <= 3.0/num_classes).float()  # [B, C, 1]
        is_top5 = (class_ranks <= 5.0/num_classes).float()  # [B, C, 1]
        
        # Broadcast global features to all classes
        entropy_broadcast = entropy.unsqueeze(1).expand(batch_size, num_classes, 1)  # [B, C, 1]
        max_prob_broadcast = max_prob.unsqueeze(1).expand(batch_size, num_classes, 1)  # [B, C, 1]
        
        # Stack all features: [B, C, 8]
        all_features = torch.cat([
            class_probs,        # Feature 1: class probability
            class_ranks,        # Feature 2: normalized rank
            gaps_to_max,        # Feature 3: gap to max
            is_top1,           # Feature 4: is top 1
            is_top3,           # Feature 5: is top 3
            is_top5,           # Feature 6: is top 5
            entropy_broadcast,  # Feature 7: entropy
            max_prob_broadcast  # Feature 8: max prob
        ], dim=2)
        
        # Reshape for batch processing: [B*C, 8]
        all_features_flat = all_features.reshape(batch_size * num_classes, 8)
        
        # Single MLP forward pass for ALL classes at once!
        all_scores_flat = self.scoring_network(all_features_flat)  # [B*C, 1]
        
        # Reshape back to [B, C]
        scores = all_scores_flat.reshape(batch_size, num_classes)
        
        # L2 regularization
        if self.training:
            l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
            self.l2_reg = self.l2_lambda * l2_reg
        else:
            self.l2_reg = 0.0
        
        return scores