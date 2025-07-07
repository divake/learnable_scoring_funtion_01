# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=None, hidden_dims=[64, 32], output_dim=None, config=None):
        """
        Initialize scoring function
        
        Args:
            input_dim: Input dimension (number of classes). If None, will be taken from config
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (number of classes). If None, will be same as input_dim
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        if config is None:
            raise ValueError("config must be provided")
        
        # Get input dimension from config if not provided
        if input_dim is None:
            # Handle both dict-like config and ConfigManager objects
            if hasattr(config, 'config'):
                # It's a ConfigManager object
                config_dict = config.config
            else:
                # It's already a dictionary
                config_dict = config
                
            if 'dataset' not in config_dict or 'num_classes' not in config_dict['dataset']:
                raise ValueError("num_classes must be specified in config['dataset'] when input_dim is not provided")
            input_dim = config_dict['dataset']['num_classes']
        
        self.input_dim = input_dim  # Store for later use
        
        # Output dimension should be same as input dimension (scores for each class)
        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim
            
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        layers = []
        prev_dim = input_dim
        
        # First layer - careful initialization
        first_layer = nn.Linear(prev_dim, hidden_dims[0])
        # Initialize to process probability vectors effectively
        nn.init.xavier_normal_(first_layer.weight)
        nn.init.constant_(first_layer.bias, 0.0)
        
        # Get activation configuration
        activation_config = config['scoring_function']['activation']
        if activation_config['name'] == 'LeakyReLU':
            activation = nn.LeakyReLU(**activation_config['params'])
        else:
            raise ValueError(f"Unsupported activation: {activation_config['name']}")
        
        layers.append(first_layer)
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(activation)
        
        # Hidden layers with dropout
        dropout_rate = config['scoring_function']['dropout']
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                activation,
                nn.Dropout(dropout_rate)
            ])
        
        # Final layer with bounded initialization
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.uniform_(final_layer.weight, -0.01, 0.01)
        nn.init.constant_(final_layer.bias, 0.5)
        layers.append(final_layer)
        
        # Get final activation configuration
        final_activation_config = config['scoring_function']['final_activation']
        if final_activation_config['name'] == 'Softplus':
            final_activation = nn.Softplus(**final_activation_config['params'])
        else:
            raise ValueError(f"Unsupported final activation: {final_activation_config['name']}")
            
        layers.append(final_activation)
        
        self.network = nn.Sequential(*layers)
        
        # Apply initialization to all layers except the first
        for module in self.network:
            if isinstance(module, nn.Linear) and module != first_layer:
                init_weights(module)
        
        self.l2_lambda = config['scoring_function']['l2_lambda']
        
        # Add a stability factor to encourage consistent coverage
        self.stability_factor = 0.05  # Reduced from 0.1 to allow more separation
        
        # Add a separation factor to encourage separation between true and false classes
        self.separation_factor = 0.5
        
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
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {x.size(-1)}")
        
        # Get scores for each class
        scores = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Force output to be reasonable (scores should be non-negative)
        scores = torch.clamp(scores, min=0.0)
        
        # Add stability term to encourage consistent behavior
        if self.training:
            # Add a small perturbation to input during training to test robustness
            perturbed_x = x + torch.randn_like(x) * 0.01
            # Ensure perturbed probabilities still sum to 1 (renormalize)
            perturbed_x = perturbed_x / perturbed_x.sum(dim=-1, keepdim=True)
            perturbed_scores = self.network(perturbed_x)
            perturbed_scores = torch.clamp(perturbed_scores, min=0.0)
            
            # Stability loss encourages similar outputs for similar inputs
            self.stability_loss = self.stability_factor * torch.mean((scores - perturbed_scores)**2)
            
            # Initialize separation loss (will be set in the trainer if needed)
            self.separation_loss = 0.0
        else:
            self.stability_loss = 0.0
            self.separation_loss = 0.0
        
        return scores