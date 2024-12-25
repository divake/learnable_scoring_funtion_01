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
    
    # Plot loss with better scaling
    plt.subplot(1, 4, 1)
    log_losses = np.array([np.log10(loss) if loss > 0 else 0 for loss in train_losses])
    plt.plot(epochs, log_losses, label='Training Loss (log10)')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot coverage with target band
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_coverages, label='Train Coverage', linewidth=2)
    plt.plot(epochs, val_coverages, label='Val Coverage', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target')
    plt.fill_between(epochs, [0.89] * len(epochs), [0.91] * len(epochs), 
                    color='r', alpha=0.1, label='Target Range')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.85, 1.0)
    
    # Plot set size with constraints
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_sizes, label='Train Set Size', linewidth=2)
    plt.plot(epochs, val_sizes, label='Val Set Size', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Minimum Size')
    plt.axhline(y=2.0, color='orange', linestyle='--', label='Maximum Size')
    plt.fill_between(epochs, [1.0] * len(epochs), [2.0] * len(epochs),
                    color='g', alpha=0.1, label='Target Range')
    plt.xlabel('Epoch')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(max(train_sizes), max(val_sizes)) + 1)
    
    # Plot tau values with bounds
    plt.subplot(1, 4, 4)
    plt.plot(epochs, tau_values, label='Tau', linewidth=2)
    plt.axhline(y=0.3, color='r', linestyle='--', label='Min Tau')
    plt.axhline(y=0.9, color='orange', linestyle='--', label='Max Tau')
    plt.fill_between(epochs, [0.3] * len(epochs), [0.9] * len(epochs),
                    color='b', alpha=0.1, label='Valid Range')
    plt.xlabel('Epoch')
    plt.ylabel('Tau Value')
    plt.title('Tau vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_scoring_function_behavior(scoring_fn, device, save_dir):
    """Plot the learned scoring function behavior with detailed analysis."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Basic scoring function behavior
    plt.subplot(1, 3, 1)
    softmax_scores = torch.linspace(0, 1, 1000, device=device).reshape(-1, 1)
    
    with torch.no_grad():
        nonconf_scores = scoring_fn(softmax_scores).cpu().numpy()
    
    plt.plot(softmax_scores.cpu().numpy(), nonconf_scores, 
             label='Learned Function', linewidth=2)
    ref_line = 1 - softmax_scores.cpu().numpy()
    plt.plot(softmax_scores.cpu().numpy(), ref_line, '--', 
             label='1-p (reference)', linewidth=2)
    
    plt.xlabel('Softmax Score')
    plt.ylabel('Non-conformity Score')
    plt.title('Scoring Function Behavior')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Derivative analysis
    plt.subplot(1, 3, 2)
    derivative = np.diff(nonconf_scores.flatten()) / np.diff(softmax_scores.cpu().numpy().flatten())
    plt.plot(softmax_scores.cpu().numpy()[1:], derivative, 
             label='Derivative', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Line')
    plt.xlabel('Softmax Score')
    plt.ylabel('Derivative')
    plt.title('Scoring Function Derivative')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Score difference from reference
    plt.subplot(1, 3, 3)
    diff = nonconf_scores.flatten() - ref_line.flatten()
    plt.plot(softmax_scores.cpu().numpy(), diff, 
             label='Difference', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', label='Reference Line')
    plt.xlabel('Softmax Score')
    plt.ylabel('Score Difference')
    plt.title('Difference from Reference (1-p)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scoring_function.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_score_distributions(true_scores, false_scores, tau, save_dir):
    """Plot distribution of conformity scores with detailed analysis."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Basic distribution
    plt.subplot(1, 3, 1)
    sns.kdeplot(true_scores, label='True Class', linewidth=2)
    sns.kdeplot(false_scores, label='False Class', linewidth=2)
    plt.axvline(x=tau, color='r', linestyle='--', label='Tau')
    plt.xlabel('Conformity Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Cumulative distribution
    plt.subplot(1, 3, 2)
    sns.ecdfplot(true_scores, label='True Class', linewidth=2)
    sns.ecdfplot(false_scores, label='False Class', linewidth=2)
    plt.axvline(x=tau, color='r', linestyle='--', label='Tau')
    plt.xlabel('Conformity Score')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distributions')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Score differences
    plt.subplot(1, 3, 3)
    score_diff = np.mean(false_scores) - np.mean(true_scores)
    plt.hist(false_scores - np.random.choice(true_scores, size=len(false_scores)),
            bins=50, alpha=0.5, label='Score Differences')
    plt.axvline(x=score_diff, color='r', linestyle='--', 
                label=f'Mean Diff: {score_diff:.3f}')
    plt.xlabel('Score Difference (False - True)')
    plt.ylabel('Count')
    plt.title('Score Separation Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_set_size_distribution(set_sizes, save_dir):
    """Plot distribution of prediction set sizes with statistics."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Histogram
    plt.subplot(1, 3, 1)
    counts, bins, _ = plt.hist(set_sizes, bins=range(int(max(set_sizes))+2), 
                              align='left', rwidth=0.8)
    plt.xlabel('Set Size')
    plt.ylabel('Count')
    plt.title('Set Size Distribution')
    plt.grid(True)
    
    # Add statistics
    mean_size = np.mean(set_sizes)
    median_size = np.median(set_sizes)
    plt.axvline(x=mean_size, color='r', linestyle='--', 
                label=f'Mean: {mean_size:.2f}')
    plt.axvline(x=median_size, color='g', linestyle='--', 
                label=f'Median: {median_size:.2f}')
    plt.legend()
    
    # Plot 2: Cumulative distribution
    plt.subplot(1, 3, 2)
    plt.hist(set_sizes, bins=range(int(max(set_sizes))+2), 
            align='left', rwidth=0.8, cumulative=True, density=True)
    plt.xlabel('Set Size')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Set Size Distribution')
    plt.grid(True)
    
    # Plot 3: Box plot with statistics
    plt.subplot(1, 3, 3)
    plt.boxplot(set_sizes, vert=False)
    plt.xlabel('Set Size')
    plt.title(f'Set Size Statistics\nMean: {mean_size:.2f}, Std: {np.std(set_sizes):.2f}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'set_size_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()