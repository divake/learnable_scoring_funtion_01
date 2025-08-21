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
                        val_coverages, val_sizes, tau_values, save_dir, val_non_empty_sizes=None):
    """Plot training metrics including tau values."""
    plotter = BasePlot(figsize=(20, 5))
    plotter.setup()
    
    # Plot loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('Training Loss')
    plt.legend()
    
    # Plot coverage
    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_coverages, label='Train Coverage')
    plt.plot(epochs, val_coverages, label='Val Coverage')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Target')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Coverage', fontweight='bold')
    plt.title('Coverage vs Epoch')
    plt.legend()
    
    # Plot set size
    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_sizes, label='Train Set Size')
    plt.plot(epochs, val_sizes, label='Val Set Size (All)')
    if val_non_empty_sizes is not None and len(val_non_empty_sizes) > 0:
        plt.plot(epochs[:len(val_non_empty_sizes)], val_non_empty_sizes, 
                label='Val Non-Empty Set Size', linestyle='--', color='green')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Average Set Size', fontweight='bold')
    plt.title('Set Size vs Epoch')
    plt.legend()
    
    # Plot tau values
    plt.subplot(1, 4, 4)
    plt.plot(epochs, tau_values, label='Tau')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Tau Value', fontweight='bold')
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
    
    plt.xlabel('Non-Conformity Score', fontweight='bold')
    plt.ylabel('Density/Frequency', fontweight='bold')
    plt.title('Distribution of Non-Conformity Scores')
    plt.legend()
    
    plotter.save(save_dir, 'score_distributions.png')

def plot_set_size_distribution(set_sizes, save_dir):
    """Plot distribution of prediction set sizes."""
    plotter = BasePlot()
    plotter.setup()
    
    plt.hist(set_sizes, bins=range(11), align='left', rwidth=0.8)
    plt.xlabel('Prediction Set Size', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
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
    plt.xlabel('Probability', fontweight='bold')
    plt.ylabel('Non-conformity Score', fontweight='bold')
    plt.title('Learned Non-conformity Scoring Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    # Dynamic y-axis to show full range including negative values
    y_min = min(np.min(scores_array) * 1.1, -0.5)  # Show at least down to -0.5
    y_max = max(np.max(scores_array) * 1.1, 1.5)    # Show at least up to 1.5
    plt.ylim(y_min, y_max)
    
    # Plot 2: Score difference from 1-p baseline
    plt.subplot(2, 2, 2)
    difference = scores_array - (1 - prob_array)
    plt.plot(prob_array, difference, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Probability', fontweight='bold')
    plt.ylabel('Score Difference from 1-p', fontweight='bold')
    plt.title('Learned Function vs 1-p Baseline')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    
    # Plot 3: Gradient/derivative of the scoring function
    plt.subplot(2, 2, 3)
    # Compute numerical gradient
    gradient = np.gradient(scores_array, prob_array)
    plt.plot(prob_array, gradient, 'purple', linewidth=2, label='Learned Function Gradient')
    plt.axhline(y=-1, color='r', linestyle='--', alpha=0.7, label='1-p Gradient (-1)')
    plt.xlabel('Probability', fontweight='bold')
    plt.ylabel('Gradient (d(score)/d(prob))', fontweight='bold')
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
    
    plt.xlabel('Probability', fontweight='bold')
    plt.ylabel('Non-conformity Score', fontweight='bold')
    plt.title('Class-Agnostic Behavior Verification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    # Dynamic y-axis for this plot too
    all_scores = []
    for class_idx in class_indices:
        with torch.no_grad():
            for p in test_probs:
                prob_vec = torch.ones(1, num_classes, device=device) * (1 - p) / (num_classes - 1)
                prob_vec[0, class_idx] = p
                scores = scoring_fn(prob_vec)
                all_scores.append(scores[0, class_idx].cpu().item())
    if all_scores:
        y_min = min(min(all_scores) * 1.1, -0.5)
        y_max = max(max(all_scores) * 1.1, 1.5)
        plt.ylim(y_min, y_max)
    
    # Add overall title
    plt.suptitle('Learned Non-conformity Scoring Function Analysis', fontsize=16)
    
    plotter.save(plot_dir, 'scoring_function.png')


def generate_best_model_visualizations(scoring_fn, test_loader, config, plot_dir, tau):
    """
    Generate high-quality visualizations for the best model.
    Called only when a new best model is saved.
    
    Args:
        scoring_fn: The trained scoring function
        test_loader: Test data loader
        config: Configuration dict
        plot_dir: Directory to save plots (should be plots/{dataset}/best_model_perf/)
        tau: The calibrated tau value from training
    """
    import os
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # Create directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    device = scoring_fn.device if hasattr(scoring_fn, 'device') else next(scoring_fn.parameters()).device
    scoring_fn.eval()
    
    # Set high-quality plot settings
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 1.2
    })
    
    # Use a professional color palette
    colors = sns.color_palette("husl", 8)
    sns.set_style("whitegrid")
    
    # 1. Generate scoring function curve
    generate_scoring_curve(scoring_fn, device, plot_dir, config)
    
    # 2. Generate true/false separation analysis
    generate_separation_analysis(scoring_fn, test_loader, device, plot_dir)
    
    # 3. Generate performance metrics (with tau for proper set size calculation)
    generate_performance_metrics(scoring_fn, test_loader, device, plot_dir, tau)
    
    # 4. Generate unified scoring function
    generate_unified_scoring(scoring_fn, device, plot_dir, config)
    
    # 5. Generate set size distribution (with tau from training)
    generate_set_size_distribution_best(scoring_fn, test_loader, device, plot_dir, tau)


def generate_scoring_curve(scoring_fn, device, save_dir, config):
    """Generate the main scoring function curve visualization"""
    import torch
    
    num_classes = config['dataset']['num_classes']
    n_samples = 100
    top_class_probs = np.linspace(0.01, 0.99, n_samples)
    
    scores_for_top = []
    scores_for_second = []
    scores_for_low = []
    
    for top_prob in top_class_probs:
        probs = torch.zeros(1, num_classes, device=device)
        probs[0, 0] = top_prob
        remaining = 1.0 - top_prob
        
        if num_classes > 1:
            second_prob = min(remaining * 0.3, remaining)
            probs[0, 1] = second_prob
            rest = remaining - second_prob
            if num_classes > 2:
                probs[0, 2:] = rest / (num_classes - 2)
        
        with torch.no_grad():
            scores = scoring_fn(probs)
        
        scores_for_top.append(scores[0, 0].cpu().item())
        scores_for_second.append(scores[0, 1].cpu().item())
        scores_for_low.append(scores[0, 50 if num_classes > 50 else num_classes//2].cpu().item())
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scoring function curve
    ax1.plot(top_class_probs, scores_for_top, 'b-', linewidth=2.5, label='Top Class', alpha=0.8)
    ax1.plot(top_class_probs, scores_for_second, 'r--', linewidth=2, label='Second Class', alpha=0.8)
    ax1.plot(top_class_probs[::5], scores_for_low[::5], 'g:', linewidth=2, label='Low Prob Class', alpha=0.8, marker='o', markersize=3)
    
    ax1.set_xlabel('Softmax Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Non-conformity Score', fontsize=12, fontweight='bold')
    ax1.set_title('Learned Scoring Function', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Dynamic y-axis
    all_scores = scores_for_top + scores_for_second + scores_for_low
    y_min = min(min(all_scores), -0.2)
    y_max = max(max(all_scores), 1.5)
    ax1.set_ylim(y_min, y_max)
    
    # Plot 2: Score distribution histogram
    ax2.hist(scores_for_top, bins=20, alpha=0.5, label='Top Class', color='blue', edgecolor='black')
    ax2.hist(scores_for_second, bins=20, alpha=0.5, label='Second Class', color='red', edgecolor='black')
    ax2.hist(scores_for_low[::5], bins=20, alpha=0.5, label='Low Prob Class', color='green', edgecolor='black')
    
    ax2.set_xlabel('Non-conformity Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scoring_function_curve.png'), bbox_inches='tight', dpi=150)
    plt.close()


def generate_separation_analysis(scoring_fn, test_loader, device, save_dir):
    """Generate true/false class separation analysis"""
    import torch
    
    scoring_fn.eval()
    all_probs = []
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            probs = inputs.to(device)
            scores = scoring_fn(probs)
            
            all_probs.append(probs.cpu())
            all_scores.append(scores.cpu())
            all_labels.append(targets.cpu())
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    n_samples = len(all_labels)
    n_classes = all_probs.shape[1]
    
    true_scores = []
    false_scores = []
    true_probs = []
    false_probs = []
    
    for i in range(n_samples):
        true_label = all_labels[i]
        true_scores.append(all_scores[i, true_label])
        true_probs.append(all_probs[i, true_label])
        
        for j in range(min(10, n_classes)):  # Sample some false classes
            if j != true_label:
                false_scores.append(all_scores[i, j])
                false_probs.append(all_probs[i, j])
    
    # Calculate tau for visualization purposes (just for the threshold line)
    # This is OK since it's only for showing where the threshold would be
    tau = np.percentile(true_scores, 90)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Score distributions
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(true_scores, bins=30, alpha=0.6, label='True Class', color='blue', density=True)
    ax1.hist(false_scores[:len(true_scores)*5], bins=30, alpha=0.6, label='False Classes', color='red', density=True)
    ax1.axvline(x=tau, color='green', linestyle='--', linewidth=2, label=f'τ={tau:.3f}')
    ax1.set_xlabel('Non-conformity Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Score Distribution')
    ax1.legend()
    
    # 2. Score vs Probability
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(true_probs, true_scores, alpha=0.5, s=20, c='blue', label='True Classes')
    ax2.scatter(false_probs[:len(true_probs)*2], false_scores[:len(true_probs)*2], 
                alpha=0.3, s=10, c='red', label='False Classes')
    ax2.axhline(y=tau, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Softmax Probability', fontweight='bold')
    ax2.set_ylabel('Non-conformity Score', fontweight='bold')
    ax2.set_title('Score vs Probability')
    ax2.legend()
    
    # 3. Box plot
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot([true_scores, false_scores[:len(true_scores)*5]], 
                      labels=['True', 'False'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.axhline(y=tau, color='green', linestyle='--', linewidth=2)
    ax3.set_ylabel('Non-conformity Score', fontweight='bold')
    ax3.set_title('Score Comparison')
    
    # 4. Cumulative distribution
    ax4 = plt.subplot(2, 3, 4)
    true_sorted = np.sort(true_scores)
    false_sorted = np.sort(false_scores[:len(true_scores)*5])
    ax4.plot(true_sorted, np.arange(len(true_sorted))/len(true_sorted), 'b-', linewidth=2, label='True')
    ax4.plot(false_sorted, np.arange(len(false_sorted))/len(false_sorted), 'r-', linewidth=2, label='False')
    ax4.axvline(x=tau, color='green', linestyle='--', linewidth=2)
    ax4.set_xlabel('Non-conformity Score', fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontweight='bold')
    ax4.set_title('CDF')
    ax4.legend()
    
    # 5. Violin plot
    ax5 = plt.subplot(2, 3, 5)
    parts = ax5.violinplot([true_scores, false_scores[:len(true_scores)*5]], 
                           positions=[1, 2], widths=0.7, showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], ['blue', 'red']):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    ax5.axhline(y=tau, color='green', linestyle='--', linewidth=2)
    ax5.set_xticks([1, 2])
    ax5.set_xticklabels(['True', 'False'])
    ax5.set_ylabel('Non-conformity Score', fontweight='bold')
    ax5.set_title('Distribution')
    
    # 6. Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""Statistics:
    
True Class:
  Mean: {np.mean(true_scores):.4f}
  Std: {np.std(true_scores):.4f}
  Min: {np.min(true_scores):.4f}
  Max: {np.max(true_scores):.4f}
  
False Class:
  Mean: {np.mean(false_scores[:len(true_scores)*5]):.4f}
  Std: {np.std(false_scores[:len(true_scores)*5]):.4f}
  
Separation:
  τ (90%): {tau:.4f}
  Coverage: {np.mean([s <= tau for s in true_scores]):.2%}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.suptitle('True vs False Class Separation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'true_false_separation.png'), bbox_inches='tight', dpi=150)
    plt.close()


def generate_performance_metrics(scoring_fn, test_loader, device, save_dir, tau=None):
    """Generate performance metrics visualization"""
    import torch
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    scoring_fn.eval()
    all_scores = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            probs = inputs.to(device)
            scores = scoring_fn(probs)
            
            all_probs.append(probs.cpu())
            all_scores.append(scores.cpu())
            all_labels.append(targets.cpu())
    
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    n_samples, n_classes = all_scores.shape
    
    # Prepare binary labels for AUROC
    y_true = []
    y_scores = []
    
    for i in range(n_samples):
        for j in range(n_classes):
            y_true.append(1 if j == all_labels[i] else 0)
            y_scores.append(-all_scores[i, j])  # Negative because lower scores = more confident
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. ROC Curve
    ax1 = axes[0, 0]
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax1.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    ax2 = axes[0, 1]
    ax2.plot(recall, precision, color='darkgreen', lw=2.5, label=f'PR (AUC = {pr_auc:.4f})')
    ax2.fill_between(recall, precision, alpha=0.3, color='darkgreen')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    # 3. Coverage vs Set Size
    ax3 = axes[1, 0]
    coverages = np.linspace(0.5, 0.99, 20)
    set_sizes = []
    
    for coverage in coverages:
        tau = np.percentile([all_scores[i, all_labels[i]] for i in range(n_samples)], coverage * 100)
        sizes = []
        for i in range(n_samples):
            size = np.sum(all_scores[i] <= tau)
            sizes.append(size)
        set_sizes.append(np.mean(sizes))
    
    ax3.plot(coverages, set_sizes, 'b-', linewidth=2.5, marker='o', markersize=4)
    ax3.axvline(x=0.9, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (90%)')
    ax3.set_xlabel('Coverage', fontweight='bold')
    ax3.set_ylabel('Average Set Size', fontweight='bold')
    ax3.set_title('Coverage vs Set Size Trade-off')
    ax3.legend()
    
    # 4. Score-Confidence Correlation
    ax4 = axes[1, 1]
    max_probs = np.max(all_probs, axis=1)
    min_scores = np.min(all_scores, axis=1)
    
    hexbin = ax4.hexbin(max_probs, min_scores, gridsize=25, cmap='YlOrRd', mincnt=1)
    ax4.set_xlabel('Max Softmax Probability', fontweight='bold')
    ax4.set_ylabel('Min Non-conformity Score', fontweight='bold')
    ax4.set_title('Confidence-Score Correlation')
    plt.colorbar(hexbin, ax=ax4, label='Count')
    
    corr = np.corrcoef(max_probs, min_scores)[0, 1]
    ax4.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Performance Metrics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_metrics.png'), bbox_inches='tight', dpi=150)
    plt.close()


def generate_set_size_distribution_best(scoring_fn, test_loader, device, save_dir, tau):
    """Generate set size distribution for the best model with detailed statistics"""
    import torch
    import numpy as np
    
    scoring_fn.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            probs = inputs.to(device)
            scores = scoring_fn(probs)
            
            all_scores.append(scores.cpu())
            all_labels.append(targets.cpu())
    
    all_scores = torch.cat(all_scores, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    n_samples = len(all_labels)
    n_classes = all_scores.shape[1]
    
    # Use the tau from training (already calibrated on calibration set)
    # This ensures consistency with the reported metrics during training
    tau = float(tau)  # Ensure it's a scalar
    
    # Calculate set sizes for each sample
    set_sizes = []
    empty_sets = 0
    singleton_sets = 0
    
    for i in range(n_samples):
        size = np.sum(all_scores[i] <= tau)
        set_sizes.append(size)
        if size == 0:
            empty_sets += 1
        elif size == 1:
            singleton_sets += 1
    
    set_sizes = np.array(set_sizes)
    
    # Calculate coverage (samples where true label is included)
    coverage = np.mean([all_scores[i, all_labels[i]] <= tau for i in range(n_samples)])
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Main histogram of set sizes
    ax1 = plt.subplot(2, 3, 1)
    max_size = min(max(set_sizes), 20)  # Cap at 20 for better visualization
    bins = np.arange(0, max_size + 2) - 0.5
    counts, _, patches = ax1.hist(set_sizes, bins=bins, edgecolor='black', linewidth=1.2)
    
    # Color code the bars
    for i, patch in enumerate(patches):
        if i == 0:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
        elif i == 1:
            patch.set_facecolor('green')
            patch.set_alpha(0.7)
        elif i <= 3:
            patch.set_facecolor('blue')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('gray')
            patch.set_alpha(0.5)
    
    ax1.set_xlabel('Prediction Set Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Set Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, min(max_size + 1, 21)))
    
    # Add vertical line for mean
    mean_size = np.mean(set_sizes)
    ax1.axvline(x=mean_size, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_size:.2f}')
    ax1.legend()
    
    # 2. Cumulative distribution
    ax2 = plt.subplot(2, 3, 2)
    sorted_sizes = np.sort(set_sizes)
    cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes)
    ax2.plot(sorted_sizes, cumulative, 'b-', linewidth=2.5)
    ax2.fill_between(sorted_sizes, 0, cumulative, alpha=0.3)
    ax2.set_xlabel('Set Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(set_sizes, p)
        ax2.axvline(x=val, color='gray', linestyle=':', alpha=0.7)
        ax2.text(val, 0.05, f'{p}%', rotation=90, fontsize=9)
    
    # 3. Box plot with violin plot overlay
    ax3 = plt.subplot(2, 3, 3)
    parts = ax3.violinplot([set_sizes], positions=[1], widths=0.7, 
                           showmeans=True, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Add box plot on top
    bp = ax3.boxplot([set_sizes], positions=[1], widths=0.3, 
                     patch_artist=True, zorder=3)
    bp['boxes'][0].set_facecolor('white')
    bp['boxes'][0].set_alpha(0.8)
    
    ax3.set_ylabel('Set Size', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution Summary', fontsize=14, fontweight='bold')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['Set Sizes'])
    
    # 4. Set size by percentile
    ax4 = plt.subplot(2, 3, 4)
    percentile_range = np.linspace(0, 100, 101)
    percentile_values = [np.percentile(set_sizes, p) for p in percentile_range]
    ax4.plot(percentile_range, percentile_values, 'b-', linewidth=2.5)
    ax4.fill_between(percentile_range, 0, percentile_values, alpha=0.3)
    ax4.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Set Size', fontsize=12, fontweight='bold')
    ax4.set_title('Set Size by Percentile', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Highlight key percentiles
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(set_sizes, p)
        ax4.plot(p, val, 'ro', markersize=8)
        ax4.text(p, val + 0.5, f'{val:.1f}', ha='center', fontsize=9)
    
    # 5. Pie chart of set size categories
    ax5 = plt.subplot(2, 3, 5)
    categories = ['Empty (0)', 'Singleton (1)', 'Small (2-3)', 'Medium (4-10)', 'Large (>10)']
    counts = [
        empty_sets,
        singleton_sets,
        np.sum((set_sizes >= 2) & (set_sizes <= 3)),
        np.sum((set_sizes >= 4) & (set_sizes <= 10)),
        np.sum(set_sizes > 10)
    ]
    
    # Filter out zero counts
    non_zero = [(cat, cnt) for cat, cnt in zip(categories, counts) if cnt > 0]
    if non_zero:
        cats, cnts = zip(*non_zero)
        colors_pie = ['red', 'green', 'blue', 'orange', 'purple'][:len(cats)]
        wedges, texts, autotexts = ax5.pie(cnts, labels=cats, colors=colors_pie, 
                                            autopct='%1.1f%%', startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    ax5.set_title('Size Categories', fontsize=14, fontweight='bold')
    
    # 6. Statistics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""Key Statistics:
    
Coverage: {coverage:.1%}
Tau threshold: {tau:.4f}

Set Size Statistics:
  Mean: {np.mean(set_sizes):.2f}
  Median: {np.median(set_sizes):.1f}
  Std Dev: {np.std(set_sizes):.2f}
  Min: {np.min(set_sizes)}
  Max: {np.max(set_sizes)}
  
Distribution:
  Empty sets: {empty_sets} ({empty_sets/n_samples*100:.1f}%)
  Singletons: {singleton_sets} ({singleton_sets/n_samples*100:.1f}%)
  Size ≤ 3: {np.sum(set_sizes <= 3)} ({np.sum(set_sizes <= 3)/n_samples*100:.1f}%)
  Size > 10: {np.sum(set_sizes > 10)} ({np.sum(set_sizes > 10)/n_samples*100:.1f}%)
  
Percentiles:
  10th: {np.percentile(set_sizes, 10):.1f}
  25th: {np.percentile(set_sizes, 25):.1f}
  50th: {np.percentile(set_sizes, 50):.1f}
  75th: {np.percentile(set_sizes, 75):.1f}
  90th: {np.percentile(set_sizes, 90):.1f}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes)
    
    plt.suptitle(f'Set Size Distribution Analysis (τ={tau:.4f}, Coverage={coverage:.1%})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'set_size_distribution.png'), bbox_inches='tight', dpi=150)
    plt.close()


def generate_unified_scoring(scoring_fn, device, save_dir, config):
    """Generate unified scoring function comparison"""
    import torch
    
    num_classes = config['dataset']['num_classes']
    n_points = 100
    probabilities = np.linspace(0.01, 0.99, n_points)
    
    # Storage for different scenarios
    scores_rank1 = []
    scores_rank2 = []
    baseline_1_minus_p = 1.0 - probabilities
    
    for p in probabilities:
        # Rank 1: Class is top prediction
        probs_rank1 = torch.zeros(1, num_classes, device=device)
        probs_rank1[0, 0] = p
        probs_rank1[0, 1:] = (1.0 - p) / (num_classes - 1)
        
        with torch.no_grad():
            scores = scoring_fn(probs_rank1)
            scores_rank1.append(scores[0, 0].cpu().item())
        
        # Rank 2: Class is second prediction (only valid when p < 0.5)
        if p < 0.5:
            probs_rank2 = torch.zeros(1, num_classes, device=device)
            probs_rank2[0, 0] = min(0.99, p + 0.1)  # Top class
            probs_rank2[0, 1] = p  # Second class
            remaining = 1.0 - probs_rank2[0, 0] - p
            if num_classes > 2 and remaining > 0:
                probs_rank2[0, 2:] = remaining / (num_classes - 2)
            
            with torch.no_grad():
                scores = scoring_fn(probs_rank2)
                scores_rank2.append(scores[0, 1].cpu().item())
        else:
            scores_rank2.append(np.nan)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Primary scoring functions
    ax1.plot(probabilities, scores_rank1, 'b-', linewidth=3, label='Learned: Rank 1', alpha=0.9)
    ax1.plot(probabilities[:50], scores_rank2[:50], 'r--', linewidth=2.5, label='Learned: Rank 2', alpha=0.8)
    ax1.plot(probabilities, baseline_1_minus_p, 'k:', linewidth=2, label='Baseline: 1-p', alpha=0.6)
    
    ax1.set_xlabel('Class Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Non-conformity Score', fontsize=12, fontweight='bold')
    ax1.set_title('Unified Scoring Function', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlim([0, 1])
    
    # Dynamic y-axis
    all_scores = scores_rank1 + [s for s in scores_rank2 if not np.isnan(s)]
    y_min = min(min(all_scores), -0.2)
    y_max = max(max(all_scores), 1.5)
    ax1.set_ylim(y_min, y_max)
    
    # Plot 2: Difference from baseline
    ax2.plot(probabilities, np.array(scores_rank1) - baseline_1_minus_p, 'b-', linewidth=3, 
             label='Rank 1 vs 1-p', alpha=0.9)
    valid_rank2 = ~np.isnan(scores_rank2)
    ax2.plot(probabilities[valid_rank2], 
             np.array(scores_rank2)[valid_rank2] - baseline_1_minus_p[valid_rank2], 
             'r--', linewidth=2.5, label='Rank 2 vs 1-p', alpha=0.8)
    
    ax2.set_xlabel('Class Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score Difference from 1-p', fontsize=12, fontweight='bold')
    ax2.set_title('Deviation from Static Baseline', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    ax2.fill_between(probabilities, -0.05, 0.05, alpha=0.2, color='gray')
    ax2.set_xlim([0, 1])
    
    plt.suptitle('Unified Scoring Function Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'unified_scoring_function.png'), bbox_inches='tight', dpi=150)
    plt.close()