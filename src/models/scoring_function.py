# src/models/scoring_function.py

# src/models/scoring_function.py

import torch
import torch.nn as nn

class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        """
        Class agnostic scoring function that takes a single probability and outputs a score
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Softplus()  # Ensure positive scores with smooth gradients
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: single probability value [batch_size, 1]
        Returns:
            score: [batch_size, 1]
        """
        return self.network(x)

class ConformalPredictor:
    def __init__(self, base_model, scoring_fn, num_classes=10):
        """
        Wrapper for base model and scoring function for conformal prediction
        
        Args:
            base_model: Frozen pretrained model
            scoring_fn: Trained scoring function
            num_classes: Number of classes
        """
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.num_classes = num_classes
        
    def get_prediction_sets(self, inputs, tau):
        """
        Generate prediction sets based on non-conformity scores
        
        Args:
            inputs: Input images
            tau: Threshold for prediction sets
        Returns:
            prediction_sets: List of prediction sets for each input
            softmax_probs: Softmax probabilities
        """
        with torch.no_grad():
            # Get softmax probabilities from base model
            logits = self.base_model(inputs)
            softmax_probs = torch.softmax(logits, dim=1)
            
            # Get non-conformity scores for all classes
            scores = torch.zeros_like(softmax_probs)
            for i in range(self.num_classes):
                scores[:, i] = self.scoring_fn(softmax_probs[:, i:i+1]).squeeze()
            
            # Generate prediction sets
            prediction_sets = []
            for i in range(len(inputs)):
                pred_set = torch.where(scores[i] <= tau)[0]
                prediction_sets.append(pred_set)
                
        return prediction_sets, softmax_probs