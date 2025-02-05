# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from typing import List

def plot_training_curves(
    epochs: List[int],
    train_losses: List[float],
    train_coverages: List[float],
    train_sizes: List[float],
    val_coverages: List[float],
    val_sizes: List[float],
    tau_values: List[float],
    save_dir: str
) -> None:
    """Plot training curves.
    
    Args:
        epochs: List of epoch numbers
        train_losses: List of training losses
        train_coverages: List of training coverage values
        train_sizes: List of training set sizes
        val_coverages: List of validation coverage values
        val_sizes: List of validation set sizes
        tau_values: List of tau values
        save_dir: Directory to save plots
    """
    if not epochs or len(epochs) < 2:
        return  # Need at least 2 points to plot
        
    plt.style.use('seaborn')
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Convert lists to numpy arrays for safety
    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    train_coverages = np.array(train_coverages)
    train_sizes = np.array(train_sizes)
    val_coverages = np.array(val_coverages)
    val_sizes = np.array(val_sizes)
    tau_values = np.array(tau_values)
    
    # Plot 1: Training Loss
    if len(train_losses) > 0:
        log_losses = np.log10(train_losses)
        axes[0].plot(epochs, log_losses, label='Training Loss (log10)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Log Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot 2: Coverage
    if len(train_coverages) > 0 and len(val_coverages) > 0:
        axes[1].plot(epochs, train_coverages, label='Train Coverage')
        axes[1].plot(epochs, val_coverages, label='Val Coverage')
        axes[1].axhline(y=0.9, color='r', linestyle='--', label='Target')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Coverage')
        axes[1].set_title('Coverage vs Epoch')
        axes[1].legend()
        axes[1].grid(True)
    
    # Plot 3: Set Size
    if len(train_sizes) > 0 and len(val_sizes) > 0:
        axes[2].plot(epochs, train_sizes, label='Train Size')
        axes[2].plot(epochs, val_sizes, label='Val Size')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Average Set Size')
        axes[2].set_title('Set Size vs Epoch')
        axes[2].legend()
        axes[2].grid(True)
    
    # Plot 4: Tau Values
    if len(tau_values) > 0:
        axes[3].plot(epochs, tau_values, label='Tau')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Tau')
        axes[3].set_title('Tau vs Epoch')
        axes[3].legend()
        axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

def plot_scoring_function_behavior(scoring_fn, device, save_dir):
    """Plot the learned scoring function behavior with detailed analysis."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Basic scoring function behavior
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