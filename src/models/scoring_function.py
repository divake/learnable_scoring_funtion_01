# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=[64, 32]):  # Removed output_dim parameter
        super().__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            
            # Second layer
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            
            # Output layer - outputs scores for all classes
            nn.Linear(hidden_dims[1], input_dim)  # Changed to input_dim to match number of classes
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.l2_lambda = 0.01
    
    def process_probs_for_class(self, probs):
        """
        Process the full probability distribution for scoring
        Args:
            probs: Tensor of shape [batch_size, num_classes] containing softmax probabilities
        Returns:
            Processed probabilities
        """
        return probs  # Return full distribution
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Tensor of shape [batch_size, num_classes] containing softmax probabilities
        Returns:
            Scores of shape [batch_size, num_classes]
        """
        # Ensure input has correct shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) > 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
            
        # Scale input to [-1, 1] range
        x = 2 * x - 1
        
        # Network forward pass
        x = self.network(x)
        
        # L2 regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        self.l2_reg = self.l2_lambda * l2_reg
        
        # Bounded output in [0, 1]
        return torch.sigmoid(x)  # Will return [batch_size, num_classes]

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