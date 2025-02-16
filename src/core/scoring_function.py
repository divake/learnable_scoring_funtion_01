# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # More conservative initialization
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.5)
        
        layers = []
        prev_dim = input_dim
        
        # First layer specifically initialized to approximate 1-x
        first_layer = nn.Linear(prev_dim, hidden_dims[0])
        nn.init.constant_(first_layer.weight, -1.0)  # Initialize to approximate 1-x
        nn.init.constant_(first_layer.bias, 1.0)
        
        layers.append(first_layer)
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.1))  # Use LeakyReLU for more stable gradients
        
        # Hidden layers with dropout
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)  # Add dropout for regularization
            ])
        
        # Final layer with bounded initialization
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.uniform_(final_layer.weight, -0.01, 0.01)
        nn.init.constant_(final_layer.bias, 0.5)
        layers.append(final_layer)
        
        # Use Softplus with higher beta for sharper transitions
        layers.append(nn.Softplus(beta=10))
        
        self.network = nn.Sequential(*layers)
        
        # Skip initialization of first layer which we set manually
        for m in self.network[3:]:  
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                init_weights(m)
        
        self.l2_lambda = 0.1  # Stronger L2 regularization
        
    def forward(self, x):
        scores = self.network(x)
        
        # Add L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Force output to be reasonable
        scores = torch.clamp(scores, min=0.0, max=1.0)
        
        return scores