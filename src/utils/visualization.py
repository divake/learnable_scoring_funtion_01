# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from typing import List
import pandas as pd
from scipy.stats import gaussian_kde

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
        axes[0].plot(epochs, train_losses, label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
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

def plot_nonconformity_scores(true_scores, false_scores, tau, save_dir, epoch=None, num_epochs=None):
    """Plot distribution of nonconformity scores for true and false classes."""
    plt.figure(figsize=(12, 8))
    
    # Define colors
    true_color = '#2ecc71'    # Emerald green
    false_color = '#e74c3c'   # Pomegranate red
    tau_color = '#3498db'     # Peter river blue
    
    # Calculate KDE for both distributions
    true_kde = gaussian_kde(true_scores.ravel())
    false_kde = gaussian_kde(false_scores.ravel())
    
    # Create evaluation points for the KDE
    x_grid = np.linspace(
        min(true_scores.min(), false_scores.min()),
        max(true_scores.max(), false_scores.max()),
        200
    )
    
    # Evaluate KDEs
    true_density = true_kde(x_grid)
    false_density = false_kde(x_grid)
    
    # Plot true class distribution
    plt.fill_between(x_grid, true_density, alpha=0.3, color=true_color)
    plt.plot(x_grid, true_density, color=true_color, linewidth=2, label='True Class')
    
    # Plot false class distribution
    plt.fill_between(x_grid, false_density, alpha=0.3, color=false_color)
    plt.plot(x_grid, false_density, color=false_color, linewidth=2, label='False Class')
    
    # Plot tau line
    plt.axvline(x=tau, color=tau_color, linestyle='--', linewidth=2, label='Tau')
    plt.text(tau + 0.00001, plt.gca().get_ylim()[1], f'τ = {tau:.6f}', 
             rotation=90, va='top', color=tau_color)
    
    # Calculate statistics
    true_mean = float(np.mean(true_scores))
    false_mean = float(np.mean(false_scores))
    separation = abs(false_mean - true_mean)
    
    # Add statistics box
    stats_text = (f'Distribution Stats:\n'
                 f'True Mean: {true_mean:.6f}\n'
                 f'False Mean: {false_mean:.6f}\n'
                 f'Separation: {separation:.6f}')
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set title based on epoch information
    if epoch is not None and num_epochs is not None:
        plt.title(f'Score Distributions (Epoch {epoch}/{num_epochs})')
    else:
        plt.title('Score Distributions')
    
    plt.xlabel('Conformity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nonconformity_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()