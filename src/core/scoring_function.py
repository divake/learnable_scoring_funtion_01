# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[64, 32], output_dim=None, config=None):
        """
        Initialize learnable scoring function with feature extraction
        
        This function extracts features from probability distributions and learns
        to produce optimal scores that minimize set size at target coverage.
        
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
        
        # Extract features from probability distribution
        dropout_rate = config['scoring_function']['dropout']
        
        # Feature extraction: compute statistics from probability vector
        # Input will be concatenated features (5 features per class)
        feature_dim = 5  # prob, 1-prob, rank, entropy contribution, relative prob
        
        # Network to process features and output scores
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self.l2_lambda = config['scoring_function']['l2_lambda']
        
        # Get training dynamics from config
        dynamics = config.get('training_dynamics', {})
        self.stability_factor = dynamics.get('stability_factor', 0.05)
        self.separation_factor = dynamics.get('separation_factor', 0.5)
        self.perturbation_noise = dynamics.get('perturbation_noise', 0.01)
        self.xavier_init_gain = dynamics.get('xavier_init_gain', 0.5)
        
        # Initialize weights to encourage symmetric behavior (after setting xavier_init_gain)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to approximate 1-p baseline initially"""
        for i, module in enumerate(self.feature_net.modules()):
            if isinstance(module, nn.Linear):
                if i == 0:  # First layer
                    # Initialize to emphasize 1-p feature (index 1)
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.weight.shape[1] >= 2:
                        module.weight.data[:, 1] = 0.5  # Emphasize 1-p feature
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    # Other layers: small random init
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the scoring function.
        
        This extracts features from probability distributions and learns
        optimal scores to minimize set size at target coverage.
        
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
        
        # Extract features for each probability
        features_list = []
        
        # Feature 1: Raw probability
        features_list.append(x)
        
        # Feature 2: 1-p (baseline score)
        features_list.append(1.0 - x)
        
        # Feature 3: Rank-based feature (normalized position when sorted)
        sorted_probs, indices = torch.sort(x, dim=1, descending=True)
        ranks = torch.zeros_like(x)
        for i in range(batch_size):
            ranks[i, indices[i]] = torch.arange(num_classes, dtype=x.dtype, device=x.device) / (num_classes - 1)
        features_list.append(ranks)
        
        # Feature 4: Entropy contribution
        entropy_contrib = -x * torch.log(x + 1e-8)
        features_list.append(entropy_contrib)
        
        # Feature 5: Relative probability (prob / max_prob)
        max_probs = x.max(dim=1, keepdim=True)[0]
        relative_probs = x / (max_probs + 1e-8)
        features_list.append(relative_probs)
        
        # Stack features
        features = torch.stack(features_list, dim=2)  # (batch_size, num_classes, 5)
        
        # Process each class independently
        features_flat = features.reshape(-1, 5)  # (batch_size * num_classes, 5)
        scores_flat = self.feature_net(features_flat)  # (batch_size * num_classes, 1)
        
        # Reshape back
        scores = scores_flat.reshape(batch_size, num_classes)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Add stability term during training
        if self.training:
            # For stability, add small L2 penalty on score variance
            score_variance = torch.var(scores, dim=1).mean()
            self.stability_loss = self.stability_factor * score_variance
            
            # No separation loss needed
            self.separation_loss = 0.0
        else:
            self.stability_loss = 0.0
            self.separation_loss = 0.0
        
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