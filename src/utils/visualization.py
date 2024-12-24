# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch

def plot_training_curves(epochs, train_losses, train_coverages, train_sizes,
                        val_coverages, val_sizes, tau_values, save_dir):
    """Plot training metrics including tau values."""
    plt.figure(figsize=(20, 5))
    
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_scoring_function_behavior(scoring_fn, device, save_dir):
    """Plot the learned scoring function behavior for each class."""
    num_classes = 10
    num_points = 1000
    
    plt.figure(figsize=(20, 15))
    
    # Create a base probability distribution
    base_probs = torch.zeros((num_points, num_classes), device=device)
    x = torch.linspace(0, 1, num_points, device=device)
    
    # Plot for each class
    for class_idx in range(num_classes):
        plt.subplot(3, 4, class_idx + 1)
        
        # Create probability distributions with varying probability for the current class
        probs = base_probs.clone()
        probs[:, class_idx] = x
        
        # Distribute remaining probability equally among other classes
        remaining_prob = (1 - x) / (num_classes - 1)
        for other_idx in range(num_classes):
            if other_idx != class_idx:
                probs[:, other_idx] = remaining_prob
        
        # Get scores
        with torch.no_grad():
            scores = scoring_fn(probs)
        
        # Plot scores for the current class
        plt.plot(x.cpu().numpy(), scores[:, class_idx].cpu().numpy(), 
                label=f'Learned Score', color='blue')
        plt.plot(x.cpu().numpy(), 1-x.cpu().numpy(), '--', 
                label='1-p (reference)', color='orange')
        
        plt.xlabel(f'Class {class_idx} Probability')
        plt.ylabel('Non-conformity Score')
        plt.title(f'Scoring Function for Class {class_idx}')
        plt.grid(True)
        if class_idx == 0:  # Only show legend for first plot
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scoring_function_by_class.png'))
    plt.close()

def plot_score_distributions(true_scores, false_scores, tau, save_dir):
    """Plot distribution of non-conformity scores."""
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(true_scores, label='True Class Scores')
    sns.kdeplot(false_scores, label='False Class Scores')
    plt.axvline(x=tau, color='r', linestyle='--', label='Tau Threshold')
    
    plt.xlabel('Non-Conformity Score')
    plt.ylabel('Density')
    plt.title('Distribution of Non-Conformity Scores')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_distributions.png'))
    plt.close()

def plot_set_size_distribution(set_sizes, save_dir):
    """Plot distribution of prediction set sizes."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(set_sizes, bins=range(11), align='left', rwidth=0.8)
    plt.xlabel('Prediction Set Size')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Set Sizes')
    plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'set_size_distribution.png'))
    plt.close()