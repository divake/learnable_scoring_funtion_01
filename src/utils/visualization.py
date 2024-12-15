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
    """Plot the learned scoring function behavior."""
    # Create input range from 0 to 1
    softmax_scores = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    
    # Get non-conformity scores
    with torch.no_grad():
        nonconf_scores = scoring_fn(softmax_scores).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(softmax_scores.cpu().numpy(), nonconf_scores)
    plt.xlabel('Softmax Score')
    plt.ylabel('Non-conformity Score')
    plt.title('Learned Scoring Function Behavior')
    plt.grid(True)
    
    # Add reference line y=1-x for comparison
    ref_line = 1 - softmax_scores.cpu().numpy()
    plt.plot(softmax_scores.cpu().numpy(), ref_line, '--', label='1-p (reference)')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scoring_function.png'))
    plt.close()

def plot_score_distributions(true_scores, false_scores, tau, save_dir):
    """Plot distribution of conformity scores."""
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(true_scores, label='True Class Scores')
    sns.kdeplot(false_scores, label='False Class Scores')
    plt.axvline(x=tau, color='r', linestyle='--', label='Tau Threshold')
    
    plt.xlabel('Conformity Score')
    plt.ylabel('Density')
    plt.title('Distribution of Conformity Scores')
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