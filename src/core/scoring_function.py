# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1, config=None):
        """
        Initialize scoring function
        
        Args:
            input_dim: Input dimension (default: 1)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (default: 1)
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        if config is None:
            raise ValueError("config must be provided")
        
        # Flag to determine whether to process entire probability vector
        self.vector_input = config['scoring_function'].get('vector_input', False)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # More conservative initialization
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.5)
        
        layers = []
        prev_dim = input_dim
        
        if self.vector_input:
            # For vector input approach, use a more standard initialization
            first_layer = nn.Linear(prev_dim, hidden_dims[0])
            nn.init.xavier_uniform_(first_layer.weight)
            nn.init.zeros_(first_layer.bias)
        else:
            # Original approach: layer specifically initialized to approximate 1-x
            first_layer = nn.Linear(prev_dim, hidden_dims[0])
            nn.init.constant_(first_layer.weight, -1.0)
            nn.init.constant_(first_layer.bias, 1.0)
        
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
        
        # Final layer
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        if self.vector_input:
            nn.init.xavier_uniform_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        else:
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
        
        # Skip initialization of first layer which we set manually
        for m in self.network[3:]:  
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                init_weights(m)
        
        self.l2_lambda = config['scoring_function']['l2_lambda']
        
        # Add a stability factor to encourage consistent coverage
        self.stability_factor = 0.05  # Reduced from 0.1 to allow more separation
        
        # Add a separation factor to encourage separation between true and false classes
        self.separation_factor = 0.5
        
    def forward(self, x):
        scores = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Force output to be reasonable but allow more separation
        # Lower minimum to allow true classes to get closer to 0
        scores = torch.clamp(scores, min=0.01, max=1.0)
        
        # Add stability term to encourage consistent behavior
        if self.training:
            # Add a small perturbation to input during training to test robustness
            perturbed_x = x + torch.randn_like(x) * 0.01
            perturbed_scores = self.network(perturbed_x)
            perturbed_scores = torch.clamp(perturbed_scores, min=0.01, max=1.0)
            
            # Stability loss encourages similar outputs for similar inputs
            self.stability_loss = self.stability_factor * torch.mean((scores - perturbed_scores)**2)
            
            # Initialize separation loss (will be set in the trainer if needed)
            self.separation_loss = 0.0
        else:
            self.stability_loss = 0.0
            self.separation_loss = 0.0
        
        return scores