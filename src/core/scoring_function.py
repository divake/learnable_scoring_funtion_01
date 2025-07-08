# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[64, 32], output_dim=None, config=None):
        """
        Initialize class-agnostic scoring function that processes full probability vectors
        
        This function learns a transformation that takes the full probability distribution
        and outputs scores for each class. It's designed to be class-agnostic through
        training with permutation augmentation and symmetric loss functions.
        
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
        
        # Build the network
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
        else:
            raise ValueError(f"Unsupported final activation: {final_activation_config['name']}")
        layers.append(final_activation)
        
        self.network = nn.Sequential(*layers)
        
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
        """Initialize weights to encourage symmetric behavior across classes"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small random initialization to break symmetry but keep it minimal
                nn.init.xavier_uniform_(module.weight, gain=self.xavier_init_gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the scoring function.
        
        This processes the full probability vector to understand the distribution
        and outputs scores for each class. The function is made class-agnostic
        through permutation augmentation during training.
        
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
        
        # Pass the full probability vector through the network
        # This allows the MLP to see the entire distribution and learn
        # relationships between probabilities
        scores = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Ensure scores are non-negative
        scores = torch.clamp(scores, min=0.0)
        
        # Add stability term during training
        if self.training:
            # Small perturbation to test robustness
            perturbed_x = x + torch.randn_like(x) * 0.01
            perturbed_x = perturbed_x / perturbed_x.sum(dim=-1, keepdim=True)
            
            perturbed_scores = self.network(perturbed_x)
            perturbed_scores = torch.clamp(perturbed_scores, min=0.0)
            
            # Stability loss encourages consistent outputs
            self.stability_loss = self.stability_factor * torch.mean((scores - perturbed_scores)**2)
            
            # Add class-agnostic regularization
            # Encourage the model to produce similar score distributions for similar probability patterns
            # This is achieved through permutation augmentation in the trainer
            self.separation_loss = 0.0  # Will be computed in trainer if needed
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