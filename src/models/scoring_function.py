# src/models/scoring_function.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[128, 64, 32], output_dim=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Better normalization strategy
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Initialize target score parameters
        self.register_buffer('target_mean', torch.tensor(0.5))
        self.register_buffer('target_std', torch.tensor(0.1))
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Final layer with careful initialization
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.xavier_uniform_(self.final_layer.weight, gain=0.01)
        nn.init.zeros_(self.final_layer.bias)
        
        self.network = nn.Sequential(*layers)
        self.l2_lambda = 0.0001
    
    def forward(self, x):
        x = self.input_norm(x)
        features = self.network(x)
        scores = self.final_layer(features)
        
        # Normalize scores to target range
        scores = torch.sigmoid(scores)  # Bound between 0 and 1
        scores = self.target_mean + (scores - 0.5) * self.target_std
        
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