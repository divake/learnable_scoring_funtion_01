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
    plt.title('Distribution of Conformity Scores')
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
    """Plot the learned scoring function behavior."""
    plotter = BasePlot(figsize=(12, 8))
    plotter.setup()
    
    # Since the scoring function now takes the entire probability vector,
    # we'll visualize how it behaves with different probability distributions
    
    # Get the expected number of classes from the scoring function
    num_classes = scoring_fn.input_dim
    
    # Create probability distributions for visualization
    n_points = 100
    
    if num_classes == 2:
        # Simple 2-class visualization
        probs_class0 = torch.linspace(0, 1, n_points, device=device)
        probs_class1 = 1 - probs_class0
        prob_vectors = torch.stack([probs_class0, probs_class1], dim=1)
    else:
        # For multi-class, create scenarios where one class dominates
        prob_vectors = []
        probs_class0 = torch.linspace(0, 1, n_points, device=device)
        
        for i in range(n_points):
            # Create a probability vector where class 0 has probability probs_class0[i]
            # and the remaining probability is distributed among other classes
            prob_vec = torch.zeros(num_classes, device=device)
            prob_vec[0] = probs_class0[i]
            # Distribute remaining probability equally among other classes
            remaining_prob = 1 - probs_class0[i]
            if num_classes > 1:
                prob_vec[1:] = remaining_prob / (num_classes - 1)
            prob_vectors.append(prob_vec)
        
        prob_vectors = torch.stack(prob_vectors, dim=0)  # Shape: (n_points, num_classes)
    
    # Get scores for these probability distributions
    with torch.no_grad():
        all_scores = scoring_fn(prob_vectors).cpu().numpy()  # Shape: (n_points, num_classes)
    
    # Extract scores for specific classes
    probs_class0_np = probs_class0.cpu().numpy()
    scores_class0 = all_scores[:, 0]  # Scores for class 0
    scores_last = all_scores[:, -1]   # Scores for last class
    
    # Plot 1: Scores vs probability of class 0
    plt.subplot(2, 2, 1)
    plt.plot(probs_class0_np, scores_class0, label='Score for Class 0')
    plt.plot(probs_class0_np, scores_last, label=f'Score for Class {num_classes-1}')
    plt.xlabel('Probability of Class 0')
    plt.ylabel('Non-conformity Score')
    plt.title('Learned Scores vs Class Probability')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Score differences
    plt.subplot(2, 2, 2)
    score_diff = scores_class0 - scores_last
    plt.plot(probs_class0_np, score_diff)
    plt.xlabel('Probability of Class 0')
    plt.ylabel('Score Difference (Class 0 - Last Class)')
    plt.title('Score Difference Between Classes')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Compare with static baselines
    plt.subplot(2, 2, 3)
    baseline_1p = 1 - probs_class0_np  # 1-p baseline for class 0
    baseline_aps = 1 - probs_class0_np  # APS baseline (same as 1-p for this case)
    plt.plot(probs_class0_np, scores_class0, label='Learned (Class 0)', linewidth=2)
    plt.plot(probs_class0_np, baseline_1p, '--', label='1-p baseline', alpha=0.7)
    plt.xlabel('Probability of Class 0')
    plt.ylabel('Non-conformity Score')
    plt.title('Learned vs Static Baselines')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Heatmap for different probability scenarios
    plt.subplot(2, 2, 4)
    if num_classes >= 3:
        # For multi-class, show how score varies with different distributions
        n_grid = 20
        scores_grid = np.zeros((n_grid, n_grid))
        
        for i in range(n_grid):
            for j in range(n_grid):
                p1 = i / (n_grid - 1)
                p2 = j / (n_grid - 1) * (1 - p1)
                p3 = 1 - p1 - p2
                if p3 >= 0:  # Valid probability distribution
                    # Create prob vector for current num_classes
                    prob_vec = torch.zeros(1, num_classes, device=device, dtype=torch.float32)
                    prob_vec[0, 0] = p1
                    prob_vec[0, 1] = p2
                    if num_classes > 3:
                        # Distribute remaining prob among other classes
                        prob_vec[0, 2:] = p3 / (num_classes - 2)
                    else:
                        prob_vec[0, 2] = p3
                    
                    with torch.no_grad():
                        scores = scoring_fn(prob_vec).cpu().numpy()
                        # Score for class 0
                        scores_grid[i, j] = scores[0, 0]
    else:
        # For 2-class, create a simple heatmap
        n_grid = 20
        scores_grid = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            p0 = i / (n_grid - 1)
            prob_vec = torch.tensor([[p0, 1-p0]], device=device, dtype=torch.float32)
            with torch.no_grad():
                scores = scoring_fn(prob_vec).cpu().numpy()
                scores_grid[i, :] = scores[0, 0]
    
    plt.imshow(scores_grid, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Score for Class 0')
    if num_classes >= 3:
        plt.xlabel('P(Class 1) / (1 - P(Class 0))')
        plt.ylabel('P(Class 0)')
        plt.title(f'Score Heatmap ({num_classes}-class scenario)')
    else:
        plt.xlabel('Sample Index')
        plt.ylabel('P(Class 0)')
        plt.title('Score Heatmap (2-class scenario)')
    
    plotter.save(plot_dir, 'scoring_function.png')