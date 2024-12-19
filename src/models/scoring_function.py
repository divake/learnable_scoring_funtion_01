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