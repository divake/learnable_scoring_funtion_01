# src/models/scoring_function.py

# src/models/scoring_function.py

import torch
import torch.nn as nn


class ScoringFunction(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        super().__init__()
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Initialize to approximate 1-x function initially
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.1)  # Small positive bias
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        layers = []
        prev_dim = input_dim
        
        # First layer to approximate 1-x
        first_layer = nn.Linear(prev_dim, hidden_dims[0])
        nn.init.constant_(first_layer.weight, -1.0)  # Initialize to approximate 1-x
        nn.init.constant_(first_layer.bias, 1.0)
        
        layers.append(first_layer)
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # Remaining layers
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU()
            ])
        
        # Final layer with careful initialization
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        nn.init.constant_(final_layer.weight, 0.1)  # Small positive weights
        nn.init.constant_(final_layer.bias, 0.1)    # Small positive bias
        layers.append(final_layer)
        
        # Use Softplus instead of ReLU for final activation
        layers.append(nn.Softplus(beta=5))  # Higher beta for sharper transition
        
        self.network = nn.Sequential(*layers)
        
        # Apply initialization to remaining layers
        for m in self.network[3:]:  # Skip first layer which we initialized specially
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                init_weights(m)

    def forward(self, x):
        """
        Forward pass to compute non-conformity scores
        
        Args:
            x: Input tensor of shape (batch_size, 1) containing softmax probabilities
        Returns:
            Non-conformity scores of shape (batch_size, 1)
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