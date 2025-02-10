# src/models/scoring_function.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[256, 128, 64], output_dim=1):
        super().__init__()
        
        # Initialize target score parameters
        self.register_buffer('target_mean', torch.tensor(0.5))
        self.register_buffer('target_std', torch.tensor(0.1))
        
        # Temperature scaling parameter
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)
        
        # Build network more efficiently with ModuleList
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for dim in hidden_dims:
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ))
            prev_dim = dim
        
        # Final layer with careful initialization
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.zeros_(self.final_layer.bias)
        
        self.l2_lambda = 0.0001
    
    def forward(self, x):
        # Handle input dimensions efficiently
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = x.view(-1, 1) if x.dim() != 2 else x
        
        # Forward pass with skip connections - more efficient implementation
        residual = x
        for layer in self.layers:
            out = layer(x)
            if out.shape == residual.shape:
                x = out + residual
            else:
                x = out
            residual = x
        
        # Final layer and normalization
        scores = self.final_layer(x)
        
        # Temperature scaling and normalization - done in one go
        scores = self.target_mean + (torch.sigmoid(scores / self.temperature) - 0.5) * self.target_std
        
        # L2 regularization - compute only during training
        if self.training:
            self.l2_reg = self.l2_lambda * sum(p.pow(2).sum() for p in self.parameters())
        
        return scores

class ConformalPredictor:
    def __init__(self, base_model, scoring_fn):
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        
    def get_prediction_sets(self, inputs, tau):
        with torch.no_grad():
            # Get softmax probabilities from base model
            logits = self.base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            batch_size = probs.size(0)
            
            # Reshape and compute scores efficiently
            flat_probs = probs.reshape(-1, 1)
            flat_scores = self.scoring_fn(flat_probs)
            scores = flat_scores.reshape(batch_size, -1)
            
            # Generate prediction sets
            prediction_sets = []
            for i in range(batch_size):
                pred_set = torch.where(scores[i] <= tau)[0]
                prediction_sets.append(pred_set)
                
        return prediction_sets, probs