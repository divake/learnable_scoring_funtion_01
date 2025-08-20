"""
High-quality visualization module for static scoring functions.
Reuses core visualization logic from visualization.py for consistency.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import json

# We don't need to import dataset classes here since we're using pre-computed results

# Configure matplotlib for high-quality output
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def generate_static_scoring_curve(scorer_name: str, dataset_name: str, save_dir: str, num_classes: int):
    """Generate scoring function curve for static scoring functions."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create probability range
    probs = np.linspace(0, 1, 1000)
    
    # Calculate scores based on the static scoring function
    if scorer_name == 'OneMinus_P' or scorer_name == '1-p':
        scores = 1 - probs
        formula = r'$s(p) = 1 - p$'
    elif scorer_name == 'APS':
        # APS: sum of sorted probabilities
        scores = []
        for p in probs:
            # Simulate a probability distribution with max prob = p
            dist = np.zeros(min(num_classes, 100))
            dist[0] = p
            remaining = 1 - p
            if len(dist) > 1:
                dist[1:] = remaining / (len(dist) - 1)
            sorted_dist = np.sort(dist)[::-1]
            cumsum = np.cumsum(sorted_dist)
            score = cumsum[-1] if len(cumsum) > 0 else p
            scores.append(score)
        scores = np.array(scores)
        formula = r'$s(p) = \sum_{i=1}^{k} p_i$ (cumulative sorted probs)'
    elif scorer_name == 'LogMargin':
        # Avoid log(0)
        eps = 1e-10
        scores = -np.log(np.maximum(probs, eps))
        formula = r'$s(p) = -\log(p)$'
    elif scorer_name == 'Sparsemax':
        # Sparsemax approximation
        scores = np.maximum(0, 1 - probs)
        formula = r'$s(p) = \max(0, 1 - p)$'
    else:
        scores = 1 - probs  # Default to 1-p
        formula = r'$s(p) = 1 - p$'
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scoring function curve
    ax1.plot(probs, scores, 'b-', linewidth=2.5, label=formula)
    ax1.fill_between(probs, scores, alpha=0.3)
    ax1.set_xlabel('Probability (p)', fontweight='bold')
    ax1.set_ylabel('Non-conformity Score', fontweight='bold')
    ax1.set_title(f'{scorer_name} Scoring Function', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add annotations for key points
    key_probs = [0.1, 0.5, 0.9]
    for kp in key_probs:
        idx = np.argmin(np.abs(probs - kp))
        ax1.plot(kp, scores[idx], 'ro', markersize=6)
        ax1.annotate(f'p={kp:.1f}\ns={scores[idx]:.2f}', 
                    xy=(kp, scores[idx]), 
                    xytext=(kp+0.05, scores[idx]+0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                    fontsize=8)
    
    # Plot 2: Gradient/Derivative (Rate of change)
    gradients = np.gradient(scores, probs)
    ax2.plot(probs, gradients, 'g-', linewidth=2.5, label='Rate of change')
    ax2.fill_between(probs, gradients, alpha=0.3, color='green')
    ax2.set_xlabel('Probability (p)', fontweight='bold')
    ax2.set_ylabel('Gradient (ds/dp)', fontweight='bold')
    ax2.set_title(f'Sensitivity Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'{scorer_name} Static Scoring Function Analysis - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{scorer_name}_{dataset_name}_scoring_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Scoring curve saved to {save_path}")

def generate_static_separation_analysis(scorer_name: str, dataset_name: str, results_dict: dict, save_dir: str):
    """Analyze score separation between true and false classes for static scorers."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract scores from results if available
    true_scores = results_dict.get('true_class_scores', [])
    false_scores = results_dict.get('false_class_scores', [])
    
    if not true_scores or not false_scores:
        logging.warning(f"No score separation data available for {scorer_name}")
        return
    
    true_scores = np.array(true_scores)
    false_scores = np.array(false_scores)
    
    # Remove NaN and Inf values
    true_scores = true_scores[np.isfinite(true_scores)]
    false_scores = false_scores[np.isfinite(false_scores)]
    
    if len(true_scores) == 0 or len(false_scores) == 0:
        logging.warning(f"All scores are NaN or Inf for {scorer_name}")
        return
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Score distributions
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.linspace(min(true_scores.min(), false_scores.min()), 
                      max(true_scores.max(), false_scores.max()), 50)
    ax1.hist(true_scores, bins=bins, alpha=0.5, label='True Class', color='green', density=True)
    ax1.hist(false_scores, bins=bins, alpha=0.5, label='False Classes', color='red', density=True)
    ax1.set_xlabel('Non-conformity Score', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Score Distribution Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. KDE plots
    ax2 = fig.add_subplot(gs[0, 2])
    from scipy.stats import gaussian_kde
    kde_true = gaussian_kde(true_scores)
    kde_false = gaussian_kde(false_scores)
    x_range = np.linspace(min(true_scores.min(), false_scores.min()),
                          max(true_scores.max(), false_scores.max()), 200)
    ax2.plot(x_range, kde_true(x_range), 'g-', linewidth=2, label='True Class')
    ax2.plot(x_range, kde_false(x_range), 'r-', linewidth=2, label='False Classes')
    ax2.fill_between(x_range, kde_true(x_range), alpha=0.3, color='green')
    ax2.fill_between(x_range, kde_false(x_range), alpha=0.3, color='red')
    ax2.set_xlabel('Score', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('KDE Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plots
    ax3 = fig.add_subplot(gs[1, 0])
    box_data = [true_scores, false_scores]
    bp = ax3.boxplot(box_data, labels=['True', 'False'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax3.set_ylabel('Non-conformity Score', fontweight='bold')
    ax3.set_title('Score Distribution Boxplot', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Violin plots
    ax4 = fig.add_subplot(gs[1, 1])
    parts = ax4.violinplot([true_scores, false_scores], positions=[0, 1], 
                           widths=0.7, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['green', 'red'][i])
        pc.set_alpha(0.7)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['True', 'False'])
    ax4.set_ylabel('Non-conformity Score', fontweight='bold')
    ax4.set_title('Score Distribution Violin Plot', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Q-Q plot
    ax5 = fig.add_subplot(gs[1, 2])
    from scipy import stats
    stats.probplot(true_scores, dist="norm", plot=ax5)
    ax5.get_lines()[0].set_markerfacecolor('green')
    ax5.get_lines()[0].set_markeredgecolor('green')
    ax5.get_lines()[0].set_markersize(4)
    ax5.set_title('Q-Q Plot (True Class)', fontweight='bold')
    ax5.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax5.set_ylabel('Sample Quantiles', fontweight='bold')
    
    # 6. Separation metrics
    ax6 = fig.add_subplot(gs[2, :])
    
    # Calculate separation metrics
    mean_diff = np.mean(false_scores) - np.mean(true_scores)
    median_diff = np.median(false_scores) - np.median(true_scores)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(true_scores) + np.var(false_scores)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Overlap coefficient
    min_max = max(true_scores.min(), false_scores.min())
    max_min = min(true_scores.max(), false_scores.max())
    overlap = max(0, max_min - min_max) / (max(true_scores.max(), false_scores.max()) - 
                                           min(true_scores.min(), false_scores.min()))
    
    # Statistical test
    ks_stat, ks_pval = stats.ks_2samp(true_scores, false_scores)
    
    metrics_text = f"""
    Separation Metrics for {scorer_name}:
    
    Mean Difference: {mean_diff:.4f}
    Median Difference: {median_diff:.4f}
    Cohen's d (Effect Size): {cohens_d:.4f}
    Distribution Overlap: {overlap:.2%}
    
    KS Test Statistic: {ks_stat:.4f}
    KS Test p-value: {ks_pval:.2e}
    
    True Class - Mean: {np.mean(true_scores):.4f}, Std: {np.std(true_scores):.4f}
    False Class - Mean: {np.mean(false_scores):.4f}, Std: {np.std(false_scores):.4f}
    """
    
    ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    ax6.axis('off')
    
    plt.suptitle(f'{scorer_name} Score Separation Analysis - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, f'{scorer_name}_{dataset_name}_separation_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Separation analysis saved to {save_path}")

def generate_static_performance_metrics(scorer_name: str, dataset_name: str, results_dict: dict, save_dir: str):
    """Generate comprehensive performance metrics visualization for static scorers."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics from results
    coverage = results_dict.get('empirical_coverage', 0.9)
    avg_set_size = results_dict.get('average_set_size', 1.0)
    auroc = results_dict.get('auroc', 0.5)
    efficiency = results_dict.get('efficiency', 0.0)
    
    # Additional metrics if available
    coverage_gap = abs(coverage - 0.9)  # Gap from target
    normalized_size = avg_set_size / results_dict.get('num_classes', 10)
    
    # Create comprehensive metrics visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main metrics bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Coverage', 'AUROC', 'Efficiency', '1/Avg Set Size']
    values = [coverage, auroc, efficiency, 1.0/avg_set_size if avg_set_size > 0 else 0]
    colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in values]
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title(f'Performance Metrics - {scorer_name}', fontweight='bold')
    ax1.axhline(y=0.9, color='blue', linestyle='--', alpha=0.5, label='Target (0.9)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Coverage vs Set Size trade-off
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(avg_set_size, coverage, s=200, c='blue', marker='o', edgecolor='black', linewidth=2)
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target Coverage')
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Ideal Set Size')
    ax2.set_xlabel('Average Set Size', fontweight='bold')
    ax2.set_ylabel('Empirical Coverage', fontweight='bold')
    ax2.set_title('Coverage-Size Trade-off', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add annotation
    ax2.annotate(f'({avg_set_size:.2f}, {coverage:.3f})', 
                xy=(avg_set_size, coverage), xytext=(avg_set_size+0.5, coverage-0.05),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.5))
    
    # 3. Radar chart for multi-metric comparison
    ax3 = fig.add_subplot(gs[1, 0], projection='polar')
    
    categories = ['Coverage\nMatch', 'AUROC', 'Efficiency', 'Size\nOptimality', 'Stability']
    values_radar = [
        1 - coverage_gap,  # How close to target coverage
        auroc,
        efficiency,
        1.0/avg_set_size if avg_set_size > 0 else 0,  # Inverse for better visualization
        1 - normalized_size  # Smaller is better
    ]
    values_radar += values_radar[:1]  # Complete the circle
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax3.plot(angles, values_radar, 'o-', linewidth=2, color='blue')
    ax3.fill(angles, values_radar, alpha=0.25, color='blue')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title('Multi-Metric Performance', fontweight='bold', pad=20)
    ax3.grid(True)
    
    # 4. Comparison with ideal metrics
    ax4 = fig.add_subplot(gs[1, 1])
    ideal_metrics = {'Coverage': 0.9, 'AUROC': 1.0, 'Efficiency': 1.0, 'Avg Set Size': 1.0}
    actual_metrics = {'Coverage': coverage, 'AUROC': auroc, 'Efficiency': efficiency, 'Avg Set Size': avg_set_size}
    
    x = np.arange(len(ideal_metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, list(ideal_metrics.values()), width, label='Ideal', alpha=0.7, color='green')
    bars2 = ax4.bar(x + width/2, list(actual_metrics.values()), width, label='Actual', alpha=0.7, color='blue')
    
    ax4.set_xlabel('Metrics', fontweight='bold')
    ax4.set_ylabel('Value', fontweight='bold')
    ax4.set_title('Actual vs Ideal Performance', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(list(ideal_metrics.keys()), rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[1, 2])
    
    summary_text = f"""
    {scorer_name} Performance Summary
    {'='*30}
    
    Dataset: {dataset_name.upper()}
    Target Coverage: 0.900
    
    Achieved Metrics:
    • Coverage: {coverage:.3f} ({coverage_gap*100:.1f}% gap)
    • AUROC: {auroc:.3f}
    • Avg Set Size: {avg_set_size:.2f}
    • Efficiency: {efficiency:.3f}
    
    Performance Rating:
    • Coverage: {'✓' if coverage_gap < 0.02 else '✗'}
    • Discrimination: {'Excellent' if auroc > 0.9 else 'Good' if auroc > 0.7 else 'Fair'}
    • Efficiency: {'High' if efficiency > 0.8 else 'Medium' if efficiency > 0.5 else 'Low'}
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, 
             fontsize=9, verticalalignment='center', fontfamily='monospace')
    ax5.axis('off')
    
    plt.suptitle(f'{scorer_name} Performance Metrics - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, f'{scorer_name}_{dataset_name}_performance_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Performance metrics saved to {save_path}")

def generate_static_set_size_distribution(scorer_name: str, dataset_name: str, results_dict: dict, save_dir: str):
    """Generate comprehensive set size distribution analysis for static scorers."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract set sizes from results
    set_sizes = results_dict.get('set_sizes', [])
    if not set_sizes:
        logging.warning(f"No set size data available for {scorer_name}")
        return
    
    set_sizes = np.array(set_sizes)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Set size histogram with KDE
    ax1 = fig.add_subplot(gs[0, :2])
    counts, bins, patches = ax1.hist(set_sizes, bins=range(1, int(set_sizes.max())+2), 
                                     alpha=0.7, color='blue', edgecolor='black', density=True)
    
    # Overlay KDE if we have enough data
    if len(set_sizes) > 10:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(set_sizes)
        x_smooth = np.linspace(set_sizes.min(), set_sizes.max(), 200)
        ax1.plot(x_smooth, kde(x_smooth), 'r-', linewidth=2, label='KDE')
    
    ax1.axvline(x=set_sizes.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {set_sizes.mean():.2f}')
    ax1.axvline(x=np.median(set_sizes), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(set_sizes):.0f}')
    ax1.set_xlabel('Set Size', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title(f'Set Size Distribution - {scorer_name}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CDF of set sizes
    ax2 = fig.add_subplot(gs[0, 2])
    sorted_sizes = np.sort(set_sizes)
    cdf = np.arange(1, len(sorted_sizes)+1) / len(sorted_sizes)
    ax2.plot(sorted_sizes, cdf, 'b-', linewidth=2)
    ax2.fill_between(sorted_sizes, 0, cdf, alpha=0.3)
    ax2.set_xlabel('Set Size', fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontweight='bold')
    ax2.set_title('Cumulative Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(set_sizes, p)
        ax2.axhline(y=p/100, color='red', linestyle=':', alpha=0.5)
        ax2.axvline(x=val, color='red', linestyle=':', alpha=0.5)
        ax2.text(val, p/100, f'{p}%:{val:.0f}', fontsize=8)
    
    # 3. Box plot with outliers
    ax3 = fig.add_subplot(gs[1, 0])
    bp = ax3.boxplot(set_sizes, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    ax3.set_ylabel('Set Size', fontweight='bold')
    ax3.set_title('Set Size Box Plot', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Frequency table
    ax4 = fig.add_subplot(gs[1, 1])
    unique_sizes, size_counts = np.unique(set_sizes, return_counts=True)
    size_probs = size_counts / len(set_sizes)
    
    # Show top 10 most common sizes
    top_indices = np.argsort(size_counts)[-10:][::-1]
    top_sizes = unique_sizes[top_indices]
    top_counts = size_counts[top_indices]
    top_probs = size_probs[top_indices]
    
    bars = ax4.bar(range(len(top_sizes)), top_probs, color='skyblue', edgecolor='black')
    ax4.set_xticks(range(len(top_sizes)))
    ax4.set_xticklabels([f'{int(s)}' for s in top_sizes], rotation=45)
    ax4.set_xlabel('Set Size', fontweight='bold')
    ax4.set_ylabel('Probability', fontweight='bold')
    ax4.set_title('Top 10 Most Common Set Sizes', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, prob, count in zip(bars, top_probs, top_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}\n({count})', ha='center', va='bottom', fontsize=8)
    
    # 5. Violin plot with quartiles
    ax5 = fig.add_subplot(gs[1, 2])
    parts = ax5.violinplot([set_sizes], positions=[1], widths=0.7, 
                           showmeans=True, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)
    
    # Add quartile lines
    quartiles = np.percentile(set_sizes, [25, 50, 75])
    ax5.hlines(quartiles, 0.7, 1.3, colors=['red', 'blue', 'red'], 
              linestyles=['--', '-', '--'], linewidths=2)
    
    ax5.set_xticks([1])
    ax5.set_xticklabels([scorer_name])
    ax5.set_ylabel('Set Size', fontweight='bold')
    ax5.set_title('Set Size Distribution (Violin)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical summary
    ax6 = fig.add_subplot(gs[2, :])
    
    # Calculate additional statistics
    mode_size = unique_sizes[np.argmax(size_counts)]
    skewness = stats.skew(set_sizes)
    kurtosis = stats.kurtosis(set_sizes)
    
    # Efficiency metrics
    singleton_rate = np.mean(set_sizes == 1)
    small_set_rate = np.mean(set_sizes <= 3)
    large_set_rate = np.mean(set_sizes >= 10)
    
    summary_text = f"""
    Set Size Distribution Statistics - {scorer_name}
    {'='*50}
    
    Basic Statistics:
    • Count: {len(set_sizes):,}
    • Mean: {set_sizes.mean():.2f}
    • Median: {np.median(set_sizes):.0f}
    • Mode: {mode_size:.0f} (occurs {size_counts[unique_sizes == mode_size][0]:,} times)
    • Std Dev: {set_sizes.std():.2f}
    • Min: {set_sizes.min():.0f}
    • Max: {set_sizes.max():.0f}
    
    Distribution Shape:
    • Skewness: {skewness:.3f} ({'right-skewed' if skewness > 0 else 'left-skewed' if skewness < 0 else 'symmetric'})
    • Kurtosis: {kurtosis:.3f} ({'heavy-tailed' if kurtosis > 0 else 'light-tailed' if kurtosis < 0 else 'normal'})
    
    Percentiles:
    • 25th: {np.percentile(set_sizes, 25):.0f}
    • 50th: {np.percentile(set_sizes, 50):.0f}
    • 75th: {np.percentile(set_sizes, 75):.0f}
    • 90th: {np.percentile(set_sizes, 90):.0f}
    • 95th: {np.percentile(set_sizes, 95):.0f}
    
    Efficiency Metrics:
    • Singleton rate: {singleton_rate:.1%} (set size = 1)
    • Small set rate: {small_set_rate:.1%} (set size ≤ 3)
    • Large set rate: {large_set_rate:.1%} (set size ≥ 10)
    """
    
    ax6.text(0.05, 0.5, summary_text, transform=ax6.transAxes, 
             fontsize=9, verticalalignment='center', fontfamily='monospace')
    ax6.axis('off')
    
    plt.suptitle(f'{scorer_name} Set Size Distribution Analysis - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, f'{scorer_name}_{dataset_name}_set_size_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Set size distribution saved to {save_path}")

def generate_all_static_visualizations(dataset_name: str, output_base_dir: str = None):
    """Generate all high-quality visualizations for static scoring functions."""
    
    if output_base_dir is None:
        output_base_dir = f'/ssd_4TB/divake/learnable_scoring_funtion_01/results_static_conformal/{dataset_name}'
    
    # Use the base directory directly - no nested folder
    viz_dir = output_base_dir
    
    # List of static scorers
    scorers = ['OneMinus_P', 'APS', 'LogMargin', 'Sparsemax']
    
    # Load the summary results
    summary_file = os.path.join(output_base_dir, f'{dataset_name}_static_conformal_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
    else:
        summary_data = {}
    
    # Get number of classes for the dataset
    num_classes_map = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
        'places365': 365,
        'ham10000': 7,
        'plantnet': 300
    }
    num_classes = num_classes_map.get(dataset_name, 10)
    
    for scorer in scorers:
        # Load scorer-specific results
        results_file = os.path.join(output_base_dir, f'{scorer}_{dataset_name}_results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            # Try alternative naming
            alt_name = '1-p' if scorer == 'OneMinus_P' else scorer
            results_file = os.path.join(output_base_dir, f'{alt_name}_{dataset_name}_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
            else:
                logging.warning(f"Results file not found for {scorer}")
                results = {}
        
        # Generate visualizations - save directly in the base directory
        try:
            # 1. Scoring function curve
            generate_static_scoring_curve(scorer, dataset_name, viz_dir, num_classes)
            
            # 2. Separation analysis (if data available)
            if 'true_class_scores' in results or 'false_class_scores' in results:
                generate_static_separation_analysis(scorer, dataset_name, results, viz_dir)
            
            # 3. Performance metrics
            generate_static_performance_metrics(scorer, dataset_name, results, viz_dir)
            
            # 4. Set size distribution (if data available)
            if 'set_sizes' in results:
                generate_static_set_size_distribution(scorer, dataset_name, results, viz_dir)
            
            logging.info(f"Generated all visualizations for {scorer} on {dataset_name}")
            
        except Exception as e:
            logging.error(f"Error generating visualizations for {scorer}: {str(e)}")
            continue
    
    # Generate comparative analysis across all scorers
    generate_comparative_analysis(dataset_name, scorers, summary_data, viz_dir)
    
    logging.info(f"All static scorer visualizations generated in {viz_dir}")

def generate_comparative_analysis(dataset_name: str, scorers: List[str], summary_data: dict, save_dir: str):
    """Generate comparative analysis across all static scorers."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Prepare data for comparison
    metrics_data = {
        'Coverage': [],
        'AUROC': [],
        'Avg Set Size': [],
        'Efficiency': []
    }
    scorer_labels = []
    
    # Get the scoring functions data
    scoring_funcs_data = summary_data.get('scoring_functions', summary_data)
    
    for scorer in scorers:
        # Try different naming conventions
        data = None
        if scorer in scoring_funcs_data:
            data = scoring_funcs_data[scorer]
        else:
            # Try alternative naming
            alt_name = '1-p' if scorer == 'OneMinus_P' else scorer
            if alt_name in scoring_funcs_data:
                data = scoring_funcs_data[alt_name]
        
        if data is None:
            continue
        
        # Extract metrics from data - handle ERROR values
        coverage = data.get('empirical_coverage', 0)
        if coverage == 'ERROR':
            coverage = 0
        
        auroc = data.get('auroc_scoring_function', data.get('auroc', 0))
        if auroc == 'ERROR':
            auroc = 0
            
        avg_size = data.get('average_set_size_excluding_empty', data.get('average_set_size', 0))
        if avg_size == 'ERROR':
            avg_size = 1  # Default to 1 to avoid division by zero
        
        # Calculate efficiency if not present
        efficiency = data.get('efficiency', coverage / avg_size if avg_size > 0 else 0)
        
        metrics_data['Coverage'].append(coverage)
        metrics_data['AUROC'].append(auroc)
        metrics_data['Avg Set Size'].append(avg_size)
        metrics_data['Efficiency'].append(efficiency)
        scorer_labels.append(scorer)
    
    if not scorer_labels:
        logging.warning("No data available for comparative analysis")
        return
    
    # 1. Bar chart comparison
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(scorer_labels))
    width = 0.2
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, values, width, label=metric, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Scoring Function', fontweight='bold')
    ax1.set_ylabel('Value', fontweight='bold')
    ax1.set_title(f'Static Scorer Comparison - {dataset_name.upper()}', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scorer_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coverage vs Set Size scatter
    ax2 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.tab10(range(len(scorer_labels)))
    for i, scorer in enumerate(scorer_labels):
        ax2.scatter(metrics_data['Avg Set Size'][i], metrics_data['Coverage'][i], 
                   s=200, c=[colors[i]], label=scorer, edgecolor='black', linewidth=2)
    
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target Coverage')
    ax2.set_xlabel('Average Set Size', fontweight='bold')
    ax2.set_ylabel('Empirical Coverage', fontweight='bold')
    ax2.set_title('Coverage-Size Trade-off', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Radar chart for all scorers
    ax3 = fig.add_subplot(gs[1, 1], projection='polar')
    
    categories = ['Coverage', 'AUROC', 'Efficiency', '1/Size']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, scorer in enumerate(scorer_labels):
        values = [
            metrics_data['Coverage'][i],
            metrics_data['AUROC'][i],
            metrics_data['Efficiency'][i],
            1.0/metrics_data['Avg Set Size'][i] if metrics_data['Avg Set Size'][i] > 0 else 0
        ]
        values += values[:1]
        ax3.plot(angles, values, 'o-', linewidth=2, label=scorer, color=colors[i])
        ax3.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title('Multi-Metric Comparison', fontweight='bold', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax3.grid(True)
    
    # 4. Heatmap of metrics
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Normalize metrics for heatmap
    metrics_matrix = np.array([metrics_data[m] for m in ['Coverage', 'AUROC', 'Efficiency']])
    
    im = ax4.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(scorer_labels)))
    ax4.set_xticklabels(scorer_labels, rotation=45, ha='right')
    ax4.set_yticks(range(3))
    ax4.set_yticklabels(['Coverage', 'AUROC', 'Efficiency'])
    ax4.set_title('Performance Heatmap', fontweight='bold')
    
    # Add text annotations
    for i in range(3):
        for j in range(len(scorer_labels)):
            text = ax4.text(j, i, f'{metrics_matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax4)
    
    # 5. Ranking table
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate ranks for each metric
    rankings = {}
    for metric in ['Coverage Gap', 'AUROC', 'Set Size', 'Efficiency']:
        if metric == 'Coverage Gap':
            values = [abs(c - 0.9) for c in metrics_data['Coverage']]
            ranks = np.argsort(values) + 1  # Lower gap is better
        elif metric == 'Set Size':
            ranks = np.argsort(metrics_data['Avg Set Size']) + 1  # Lower is better
        else:
            ranks = np.argsort(metrics_data[metric])[::-1] + 1  # Higher is better
        
        rankings[metric] = ranks
    
    # Create ranking text
    ranking_text = "Performance Rankings\n" + "="*50 + "\n\n"
    ranking_text += f"{'Scorer':<15} {'Coverage Gap':<15} {'AUROC':<10} {'Set Size':<10} {'Efficiency':<12} {'Overall':<10}\n"
    ranking_text += "-"*80 + "\n"
    
    for i, scorer in enumerate(scorer_labels):
        overall_rank = np.mean([rankings[m][i] for m in rankings])
        ranking_text += f"{scorer:<15} "
        ranking_text += f"{rankings['Coverage Gap'][i]:<15} "
        ranking_text += f"{rankings['AUROC'][i]:<10} "
        ranking_text += f"{rankings['Set Size'][i]:<10} "
        ranking_text += f"{rankings['Efficiency'][i]:<12} "
        ranking_text += f"{overall_rank:.1f}\n"
    
    ranking_text += "\n(Lower rank number = better performance)"
    
    ax5.text(0.1, 0.5, ranking_text, transform=ax5.transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    ax5.axis('off')
    
    plt.suptitle(f'Static Scorer Comparative Analysis - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold')
    
    save_path = os.path.join(save_dir, f'comparative_analysis_{dataset_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Comparative analysis saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate high-quality visualizations for static scorers')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Generate visualizations
    generate_all_static_visualizations(args.dataset, args.output_dir)