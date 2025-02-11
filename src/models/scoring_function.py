# src/models/scoring_function.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[256, 128, 64], output_dim=1):
        super().__init__()
        
        # Initialize target score parameters
        self.register_buffer('target_mean', torch.tensor(0.5))
        self.register_buffer('target_std', torch.tensor(0.1))
        
        # Temperature scaling parameter for final output
        self.temperature = nn.Parameter(torch.ones(1) * 5.0)
        
        # Build efficient network with fused operations
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim, bias=False),  # Remove bias as it's handled by LayerNorm
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        # Use ModuleList for better performance
        self.layers = nn.ModuleList(layers)
        
        # Final layer with careful initialization
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.zeros_(self.final_layer.bias)
        
        self.l2_lambda = 0.0001
        
        # Enable torch.compile for faster execution
        if hasattr(torch, 'compile'):
            self.forward = torch.compile(self.forward)
    
    def _process_batch(self, x):
        """Process a batch of inputs efficiently."""
        # Handle input dimensions
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x.view(-1, 1) if x.dim() != 2 else x
        
        # Ensure input requires grad
        if not x.requires_grad and self.training:
            x = x.detach().requires_grad_(True)
        
        return x
    
    def forward(self, x):
        x = self._process_batch(x)
        
        # Forward pass with residual connections
        residual = x
        for i in range(0, len(self.layers), 4):
            # Process layer group (Linear + LayerNorm + GELU + Dropout)
            layer_group = self.layers[i:i+4]
            
            # Compute layer output
            for layer in layer_group:
                x = layer(x)
            
            # Add residual if dimensions match
            if x.size(-1) == residual.size(-1):
                x = x + residual
            residual = x
        
        # Final layer with temperature scaling
        scores = self.final_layer(x)
        scores = 1.0 - torch.sigmoid(scores / self.temperature)
        
        # L2 regularization during training
        if self.training:
            self.l2_reg = self.l2_lambda * sum(p.pow(2).sum() for p in self.parameters())
        
        return scores