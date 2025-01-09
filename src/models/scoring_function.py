# src/models/scoring_function.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[128, 64, 32], output_dim=1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Use BatchNorm1d for better stability with probabilities
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Wider and deeper architecture for CIFAR-100
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)  # Increased dropout for better generalization
            ])
            prev_dim = dim
        
        # Final layer with careful initialization
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        # Smaller initialization for better initial scores
        nn.init.xavier_uniform_(final_layer.weight, gain=0.01)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)
        
        # Modified softplus for better score range
        layers.append(nn.Softplus(beta=10))  # Increased beta for sharper transitions
        
        self.network = nn.Sequential(*layers)
        self.l2_lambda = 0.0005  # Further reduced L2 regularization
        
        # Initialize running statistics for score normalization
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))
        self.momentum = 0.1
    
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Forward pass through network
        scores = self.network(x)
        
        if self.training:
            # Update running statistics during training
            with torch.no_grad():
                batch_mean = scores.mean()
                batch_std = scores.std()
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std
        
        # Normalize scores
        scores = (scores - self.running_mean) / (self.running_std + 1e-6)
        
        # L2 regularization
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