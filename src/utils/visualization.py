# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

class BasePlot:
    """Base class for all plots to reduce code duplication"""
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
    
    def setup(self):
        """Setup the plot"""
        plt.figure(figsize=self.figsize)
    
    def save(self, save_dir, filename):
        """Save the plot"""
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()

def plot_training_curves(epochs, train_losses, train_coverages, train_sizes,
                        val_coverages, val_sizes, tau_values, save_dir):
    """Plot training metrics including tau values."""
    plotter = BasePlot(figsize=(20, 5))
    plotter.setup()
    
    # Plot loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot coverage
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_coverages, label='Train Coverage')
    plt.plot(epochs, val_coverages, label='Val Coverage')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Epoch')
    plt.legend()
    
    # Plot set size
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_sizes, label='Train Set Size')
    plt.plot(epochs, val_sizes, label='Val Set Size')
    plt.xlabel('Epoch')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Epoch')
    plt.legend()
    
    # Plot tau values
    plt.subplot(1, 4, 4)
    plt.plot(epochs, tau_values, label='Tau')
    plt.xlabel('Epoch')
    plt.ylabel('Tau Value')
    plt.title('Tau vs Epoch')
    plt.legend()
    
    plotter.save(save_dir, 'training_curves.png')

def plot_score_distributions(true_scores, false_scores, tau, save_dir):
    """Plot distribution of conformity scores."""
    plotter = BasePlot()
    plotter.setup()
    
    # Check variance of scores to avoid KDE warnings
    true_var = np.var(true_scores) if len(true_scores) > 0 else 0
    false_var = np.var(false_scores) if len(false_scores) > 0 else 0
    
    # Use KDE for scores with sufficient variance, otherwise use histograms
    if true_var > 1e-10:
        sns.kdeplot(true_scores, label='True Class Scores')
    else:
        plt.hist(true_scores, bins=10, alpha=0.5, label='True Class Scores')
        
    if false_var > 1e-10:
        sns.kdeplot(false_scores, label='False Class Scores')
    else:
        plt.hist(false_scores, bins=10, alpha=0.5, label='False Class Scores')
    
    plt.axvline(x=tau, color='r', linestyle='--', label='Tau Threshold')
    
    plt.xlabel('Non-Conformity Score')
    plt.ylabel('Density/Frequency')
    plt.title('Distribution of Non-Conformity Scores')
    plt.legend()
    
    plotter.save(save_dir, 'score_distributions.png')

def plot_set_size_distribution(set_sizes, save_dir):
    """Plot distribution of prediction set sizes."""
    plotter = BasePlot()
    plotter.setup()
    
    plt.hist(set_sizes, bins=range(11), align='left', rwidth=0.8)
    plt.xlabel('Prediction Set Size')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Set Sizes')
    plt.xticks(range(10))
    
    plotter.save(save_dir, 'set_size_distribution.png')

def plot_scoring_function_behavior(scoring_fn, device, plot_dir):
    """Plot the learned non-conformity scoring function behavior."""
    plotter = BasePlot(figsize=(12, 10))
    plotter.setup()
    
    # For a class-agnostic scoring function, we need to understand how it maps
    # a probability value to a non-conformity score
    
    # Create a range of probability values
    n_points = 1000
    prob_values = torch.linspace(0.001, 0.999, n_points, device=device)
    
    # Get the expected number of classes from the scoring function
    num_classes = scoring_fn.num_classes if hasattr(scoring_fn, 'num_classes') else scoring_fn.input_dim
    
    # Create probability vectors with varying confidence for one class
    # while distributing the remaining probability uniformly
    scores_list = []
    
    with torch.no_grad():
        for p in prob_values:
            # Create a probability vector where one class has probability p
            # and the rest share the remaining probability equally
            prob_vec = torch.ones(1, num_classes, device=device) * (1 - p) / (num_classes - 1)
            prob_vec[0, 0] = p  # Set first class to have probability p
            
            # Get scores from the scoring function
            scores = scoring_fn(prob_vec)
            # Extract the score for the high-probability class
            scores_list.append(scores[0, 0].cpu().item())
    
    scores_array = np.array(scores_list)
    prob_array = prob_values.cpu().numpy()
    
    # Plot 1: Main scoring function curve
    plt.subplot(2, 2, 1)
    plt.plot(prob_array, scores_array, 'b-', linewidth=2, label='Learned Scoring Function')
    plt.plot(prob_array, 1 - prob_array, 'r--', linewidth=2, alpha=0.7, label='1-p (APS baseline)')
    plt.xlabel('Probability')
    plt.ylabel('Non-conformity Score')
    plt.title('Learned Non-conformity Scoring Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, max(1, np.max(scores_array) * 1.1))
    
    # Plot 2: Score difference from 1-p baseline
    plt.subplot(2, 2, 2)
    difference = scores_array - (1 - prob_array)
    plt.plot(prob_array, difference, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Probability')
    plt.ylabel('Score Difference from 1-p')
    plt.title('Learned Function vs 1-p Baseline')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Plot 3: Gradient/derivative of the scoring function
    plt.subplot(2, 2, 3)
    # Compute numerical gradient
    gradient = np.gradient(scores_array, prob_array)
    plt.plot(prob_array, gradient, 'purple', linewidth=2, label='Learned Function Gradient')
    plt.axhline(y=-1, color='r', linestyle='--', alpha=0.7, label='1-p Gradient (-1)')
    plt.xlabel('Probability')
    plt.ylabel('Gradient (d(score)/d(prob))')
    plt.title('Scoring Function Gradient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Plot 4: Class-agnostic verification
    # Test if the function gives similar scores for the same probability
    # regardless of which class has that probability
    plt.subplot(2, 2, 4)
    
    # Test with a few specific probability values
    test_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    class_indices = [0, num_classes//2, num_classes-1] if num_classes > 2 else [0, 1]
    
    for class_idx in class_indices:
        scores_for_class = []
        
        with torch.no_grad():
            for p in test_probs:
                # Create probability vector with class_idx having probability p
                prob_vec = torch.ones(1, num_classes, device=device) * (1 - p) / (num_classes - 1)
                prob_vec[0, class_idx] = p
                
                scores = scoring_fn(prob_vec)
                scores_for_class.append(scores[0, class_idx].cpu().item())
        
        plt.plot(test_probs, scores_for_class, 'o-', 
                label=f'Class {class_idx}', markersize=8, linewidth=2)
    
    plt.xlabel('Probability')
    plt.ylabel('Non-conformity Score')
    plt.title('Class-Agnostic Behavior Verification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Add overall title
    plt.suptitle('Learned Non-conformity Scoring Function Analysis', fontsize=16)
    
    plotter.save(plot_dir, 'scoring_function.png')