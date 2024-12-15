# src/utils/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_training_curves(epochs, train_losses, train_coverages, train_sizes,
                        val_coverages, val_sizes, save_dir):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot coverage
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_coverages, label='Train Coverage')
    plt.plot(epochs, val_coverages, label='Val Coverage')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage')
    plt.title('Coverage vs Epoch')
    plt.legend()
    
    # Plot set size
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_sizes, label='Train Set Size')
    plt.plot(epochs, val_sizes, label='Val Set Size')
    plt.xlabel('Epoch')
    plt.ylabel('Average Set Size')
    plt.title('Set Size vs Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
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