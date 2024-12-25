# src/models/scoring_function.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Normalization layer
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Wider architecture
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        # Final layer
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
        nn.init.constant_(final_layer.bias, 0.0)
        layers.append(final_layer)
        
        # Use softplus instead of sigmoid for smoother gradients
        layers.append(nn.Softplus(beta=5))
        
        self.network = nn.Sequential(*layers)
        self.l2_lambda = 0.001  # Reduced L2 regularization
    
    def forward(self, x):
        x = self.input_norm(x)
        scores = self.network(x)
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