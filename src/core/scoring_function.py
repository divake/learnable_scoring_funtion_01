# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[64, 32], output_dim=None, config=None):
        """
        Initialize enhanced scoring function with distribution features and threshold prediction
        
        This function processes probability distributions along with extracted features
        to learn optimal scores and threshold for each distribution.
        
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
        
        # Build the main network for scoring
        layers = []
        prev_dim = input_dim
        
        # Get activation configuration
        activation_config = config['scoring_function']['activation']
        if activation_config['name'] == 'LeakyReLU':
            activation = nn.LeakyReLU(**activation_config['params'])
        else:
            raise ValueError(f"Unsupported activation: {activation_config['name']}")
        
        # Build hidden layers
        dropout_rate = config['scoring_function']['dropout']
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final layer - outputs scores for each class
        final_layer = nn.Linear(prev_dim, output_dim)
        layers.append(final_layer)
        
        # Get final activation configuration
        final_activation_config = config['scoring_function']['final_activation']
        if final_activation_config['name'] == 'Softplus':
            final_activation = nn.Softplus(**final_activation_config['params'])
        elif final_activation_config['name'] == 'Sigmoid':
            # Sigmoid ensures output is in [0, 1] range
            final_activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported final activation: {final_activation_config['name']}")
        layers.append(final_activation)
        
        self.network = nn.Sequential(*layers)
        
        # Remove threshold prediction network - we'll use calibration set for tau
        
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
        """Initialize weights to encourage learning from 1-p baseline"""
        modules_list = list(self.modules())
        for i, module in enumerate(modules_list):
            if isinstance(module, nn.Linear):
                # Initialize with very small weights to start near identity-like function
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    # Initialize bias to small negative values to start with lower scores
                    nn.init.constant_(module.bias, -0.1)
    
    def _extract_distribution_features(self, probs):
        """
        Extract informative features from probability distribution
        
        Args:
            probs: Probability vectors of shape (batch_size, num_classes)
            
        Returns:
            features: Tensor of shape (batch_size, num_dist_features)
        """
        batch_size = probs.shape[0]
        features = []
        
        # 1. Entropy - measures uncertainty
        # Clamp probabilities to avoid log(0)
        probs_clamped = torch.clamp(probs, min=1e-8, max=1-1e-8)
        entropy = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)
        features.append(entropy)
        
        # 2. Max probability - confidence level
        max_prob, _ = torch.max(probs, dim=1)
        features.append(max_prob)
        
        # 3. Top-5 probability sum - concentration in top classes
        top5_probs, _ = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        top5_sum = torch.sum(top5_probs, dim=1)
        features.append(top5_sum)
        
        # 4. Gini coefficient - inequality measure
        sorted_probs, _ = torch.sort(probs, dim=1)
        n = probs.shape[1]
        index = torch.arange(1, n + 1, dtype=probs.dtype, device=probs.device).unsqueeze(0)
        sum_probs = torch.sum(sorted_probs, dim=1, keepdim=True)
        # Avoid division by zero
        sum_probs = torch.clamp(sum_probs, min=1e-8)
        gini = (2 * torch.sum(index * sorted_probs, dim=1)) / (n * sum_probs.squeeze()) - (n + 1) / n
        features.append(gini)
        
        # 5. Variance - spread of probabilities
        mean_prob = torch.mean(probs, dim=1, keepdim=True)
        variance = torch.mean((probs - mean_prob) ** 2, dim=1)
        features.append(variance)
        
        # 6. Distance from uniform distribution
        uniform_prob = 1.0 / probs.shape[1]
        dist_from_uniform = torch.mean(torch.abs(probs - uniform_prob), dim=1)
        features.append(dist_from_uniform)
        
        # 7. Number of classes above threshold (0.01) - sparsity
        num_above_threshold = torch.sum(probs > 0.01, dim=1).float()
        features.append(num_above_threshold)
        
        # 8. Ratio of max to second max - margin
        top2_probs, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
        if probs.shape[1] > 1:
            # Clamp ratio to prevent extreme values
            ratio = torch.clamp(top2_probs[:, 0] / (top2_probs[:, 1] + 1e-8), max=1000.0)
        else:
            ratio = torch.ones(batch_size, device=probs.device)
        features.append(ratio)
        
        # 9. Effective number of classes (perplexity)
        # Clamp entropy to prevent overflow in exp
        perplexity = torch.exp(torch.clamp(entropy, max=10.0))
        features.append(perplexity)
        
        # Stack all features
        features = torch.stack(features, dim=1)
        
        # Check for NaN/inf and replace with zeros
        features = torch.nan_to_num(features, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        return features
    
    def forward(self, x):
        """
        Forward pass through the enhanced scoring function.
        
        This processes probability vectors along with extracted distribution features
        to produce scores for each class.
        
        Args:
            x: Input tensor of shape (batch_size, num_classes) containing probability vectors
            
        Returns:
            scores: Tensor of shape (batch_size, num_classes) with scores for each class
        """
        # Ensure input has correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size, num_classes = x.shape
        
        if num_classes != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} classes, got {num_classes}")
        
        # Pass probabilities directly through the network
        # No feature extraction - keep it simple
        scores = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # No need to clamp if using sigmoid activation
        # For other activations, ensure scores are non-negative
        if not isinstance(self.network[-1], nn.Sigmoid):
            scores = torch.clamp(scores, min=0.0)
        
        # Add stability term during training
        if self.training:
            # Small perturbation to test robustness
            perturbed_x = x + torch.randn_like(x) * self.perturbation_noise
            perturbed_x = perturbed_x / perturbed_x.sum(dim=-1, keepdim=True)
            
            perturbed_scores = self.network(perturbed_x)
            
            if not isinstance(self.network[-1], nn.Sigmoid):
                perturbed_scores = torch.clamp(perturbed_scores, min=0.0)
            
            # Stability loss encourages consistent outputs
            self.stability_loss = self.stability_factor * torch.mean((scores - perturbed_scores)**2)
            
            # Add class-agnostic regularization
            # Encourage the model to produce similar score distributions for similar probability patterns
            # This is achieved through permutation augmentation in the trainer
            # The separation loss will be computed in the trainer using self.separation_factor
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