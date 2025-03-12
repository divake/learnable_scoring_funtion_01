"""
Advanced metrics for evaluating conformal prediction models.

This module provides functions to calculate and visualize:
1. AUROC (Area Under the Receiver Operating Characteristic curve)
2. AUARC (Area Under the Adaptive Risk Control curve)
3. ECE (Expected Calibration Error)

These metrics are particularly useful for evaluating conformal prediction models
and understanding the trade-offs between error rates, prediction set sizes, and calibration.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from typing import List, Dict, Tuple, Union, Optional, Any
import os
from sklearn.preprocessing import label_binarize


def calculate_auroc(y_true: np.ndarray, y_scores: np.ndarray, multi_class: str = 'ovr') -> float:
    """
    Calculate the Area Under the Receiver Operating Characteristic curve (AUROC).
    
    Args:
        y_true: Ground truth binary labels (0 for negative, 1 for positive)
        y_scores: Predicted scores or probabilities
        multi_class: Strategy for multi-class classification:
                    'ovr' - One-vs-Rest
                    'ovo' - One-vs-One
    
    Returns:
        AUROC score
    
    Note:
        For binary classification, y_scores should be the probability of the positive class.
        For multi-class, y_scores should be a 2D array of shape (n_samples, n_classes).
    """
    from sklearn.metrics import roc_auc_score
    
    # Handle different input types
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Check if binary or multi-class
    if len(y_scores.shape) == 1 or y_scores.shape[1] == 1:
        # Binary classification
        return roc_auc_score(y_true, y_scores)
    else:
        # Multi-class classification
        # For multi-class, we need to ensure the scores are proper probabilities
        # that sum to 1.0 across classes
        
        # Get the number of classes from the scores shape
        n_classes = y_scores.shape[1]
        
        # Check if we need to normalize the scores
        row_sums = np.sum(y_scores, axis=1)
        if not np.allclose(row_sums, 1.0):
            # Normalize to ensure scores sum to 1.0 for each sample
            y_scores = y_scores / row_sums[:, np.newaxis]
        
        # For one-vs-rest approach, we can use label_binarize to convert y_true to one-hot encoding
        if multi_class == 'ovr':
            # Get unique classes
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            
            # If only 2 classes, roc_auc_score expects 1D array of scores for the positive class
            if len(classes) == 2:
                return roc_auc_score(y_true_bin, y_scores[:, 1])
            else:
                return roc_auc_score(y_true_bin, y_scores, multi_class=multi_class, average='macro')
        else:
            # For one-vs-one, we can use the multi_class parameter directly
            return roc_auc_score(y_true, y_scores, multi_class=multi_class, average='macro')


def calculate_auarc(prediction_sets: List[set], true_labels: np.ndarray, 
                   set_sizes: np.ndarray) -> float:
    """
    Calculate the Area Under the Adaptive Risk Control curve (AUARC).
    
    In conformal prediction, this is the area under the curve of
    error rate (1 - coverage) vs. average set size.
    
    Args:
        prediction_sets: List of prediction sets (each set contains class indices)
        true_labels: Ground truth labels
        set_sizes: Size of each prediction set
    
    Returns:
        AUARC score (lower is better)
    """
    # Handle different input types
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(set_sizes, torch.Tensor):
        set_sizes = set_sizes.cpu().numpy()
    
    # Calculate error rates for different set sizes
    unique_sizes = np.sort(np.unique(set_sizes))
    error_rates = []
    avg_set_sizes = []
    
    for size in unique_sizes:
        # Find samples with set size <= current size
        mask = set_sizes <= size
        if not np.any(mask):
            continue
            
        # Calculate error rate for these samples (true label not in prediction set)
        errors = 0
        for i, label in enumerate(true_labels):
            if mask[i] and label not in prediction_sets[i]:
                errors += 1
        
        error_rate = errors / np.sum(mask)
        avg_size = np.mean(set_sizes[mask])
        
        error_rates.append(error_rate)
        avg_set_sizes.append(avg_size)
    
    # Ensure we have at least two points for AUC calculation
    if len(error_rates) < 2:
        return 0.0
    
    # Calculate AUC using the trapezoidal rule
    # Note: We're calculating the area under the error rate vs. set size curve
    auarc = np.trapz(error_rates, avg_set_sizes)
    
    # Normalize by the maximum possible set size to get a value between 0 and 1
    # In conformal prediction, the maximum set size is the number of classes
    max_possible_size = np.max(set_sizes)
    if max_possible_size > 0:
        auarc /= max_possible_size
    
    return auarc


def calculate_auarc_from_scores(y_true: np.ndarray, y_scores: np.ndarray, 
                               tau_values: np.ndarray) -> float:
    """
    Calculate AUARC from raw scores and tau thresholds.
    
    This is an alternative implementation that works directly with scores
    rather than prediction sets.
    
    Args:
        y_true: Ground truth labels
        y_scores: Score matrix of shape (n_samples, n_classes)
                 In conformal prediction, lower scores indicate higher confidence
                 (included in prediction set if score <= tau)
        tau_values: Array of tau thresholds to evaluate
    
    Returns:
        AUARC score (lower is better)
    """
    # Handle different input types
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    if isinstance(tau_values, torch.Tensor):
        tau_values = tau_values.cpu().numpy()
    
    error_rates = []
    avg_set_sizes = []
    
    for tau in tau_values:
        # Create prediction sets for this tau
        pred_sets = y_scores <= tau
        set_sizes = np.sum(pred_sets, axis=1)
        
        # Calculate error rate (true label not in prediction set)
        correct = 0
        for i, label in enumerate(y_true):
            if pred_sets[i, label]:
                correct += 1
        
        accuracy = correct / len(y_true)
        error_rate = 1 - accuracy
        avg_size = np.mean(set_sizes)
        
        error_rates.append(error_rate)
        avg_set_sizes.append(avg_size)
    
    # Sort by set size for proper AUC calculation
    sorted_indices = np.argsort(avg_set_sizes)
    error_rates = np.array(error_rates)[sorted_indices]
    avg_set_sizes = np.array(avg_set_sizes)[sorted_indices]
    
    # Calculate AUC
    auarc = np.trapz(error_rates, avg_set_sizes)
    
    # Normalize by the maximum possible set size (number of classes)
    max_possible_size = y_scores.shape[1]
    if max_possible_size > 0:
        auarc /= max_possible_size
    
    return auarc


def calculate_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures how well a model's predicted probabilities align with actual outcomes.
    It quantifies the calibration of your model by dividing predictions into confidence bins
    and calculating the weighted average of the difference between confidence and accuracy in each bin.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score (lower is better, 0 is perfect calibration)
    """
    # Handle different input types
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.cpu().numpy()
    
    # For multi-class, extract the probability of the predicted class
    if len(y_probs.shape) > 1 and y_probs.shape[1] > 1:
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.array([y_probs[i, pred] for i, pred in enumerate(y_pred)])
    else:
        y_pred = (y_probs >= 0.5).astype(int)
        confidences = y_probs.copy()
        confidences[y_pred == 0] = 1 - confidences[y_pred == 0]
    
    # Create bins and find bin for each prediction
    bin_indices = np.minimum(n_bins - 1, np.floor(confidences * n_bins).astype(int))
    
    # Initialize bins
    bin_sums = np.zeros(n_bins)
    bin_true_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Fill bins
    for i in range(len(y_true)):
        bin_idx = bin_indices[i]
        bin_sums[bin_idx] += confidences[i]
        bin_true_sums[bin_idx] += (y_pred[i] == y_true[i])
        bin_counts[bin_idx] += 1
    
    # Calculate ECE
    ece = 0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_conf = bin_sums[i] / bin_counts[i]
            bin_acc = bin_true_sums[i] / bin_counts[i]
            bin_weight = bin_counts[i] / len(y_true)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    
    return ece


def calculate_ece_for_conformal(prediction_sets: List[set], true_labels: np.ndarray, 
                               confidence_levels: np.ndarray) -> float:
    """
    Calculate Expected Calibration Error (ECE) for conformal prediction.
    
    In conformal prediction, this measures how well the claimed confidence levels
    (e.g., 90% confidence) match the actual coverage of true labels in prediction sets.
    
    Args:
        prediction_sets: List of lists of prediction sets at different confidence levels
        true_labels: Ground truth labels
        confidence_levels: Array of confidence levels (1-alpha) used to generate the prediction sets
    
    Returns:
        ECE score (lower is better, 0 is perfect calibration)
    """
    # Handle different input types
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(confidence_levels, torch.Tensor):
        confidence_levels = confidence_levels.cpu().numpy()
    
    # Calculate actual coverage for each confidence level
    actual_coverages = []
    
    for level_idx, level_sets in enumerate(prediction_sets):
        correct = 0
        for i, label in enumerate(true_labels):
            if label in level_sets[i]:
                correct += 1
        
        coverage = correct / len(true_labels)
        actual_coverages.append(coverage)
    
    # Calculate ECE as the weighted average of |actual_coverage - claimed_confidence|
    ece = 0
    for i, (claimed, actual) in enumerate(zip(confidence_levels, actual_coverages)):
        # Weight by the number of samples (equal in this case)
        weight = 1.0 / len(confidence_levels)
        ece += weight * np.abs(actual - claimed)
    
    return ece


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                  model_names: Optional[List[str]] = None, 
                  ax: Optional[plt.Axes] = None,
                  title: str = "ROC Curve",
                  save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot the ROC curve for one or more models.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: List of score arrays for each model
        model_names: Names of the models for the legend
        ax: Matplotlib axes to plot on
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Handle single model case
    if not isinstance(y_scores[0], (list, np.ndarray, torch.Tensor)) or len(y_scores[0]) == 1:
        y_scores = [y_scores]
        
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(y_scores))]
    
    for i, scores in enumerate(y_scores):
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true
            
        fpr, tpr, _ = roc_curve(y_true_np, scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'{model_names[i]} (AUROC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax


def plot_auarc_curve(error_rates: List[float], set_sizes: List[float], 
                    model_names: Optional[List[str]] = None,
                    ax: Optional[plt.Axes] = None,
                    title: str = "Error Rate vs. Set Size",
                    log_scale: bool = False,
                    save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot the Error Rate vs. Set Size curve (for AUARC).
    
    Args:
        error_rates: List of error rate arrays for each model
        set_sizes: List of set size arrays for each model
        model_names: Names of the models for the legend
        ax: Matplotlib axes to plot on
        title: Plot title
        log_scale: Whether to use log scale for the x-axis
        save_path: Path to save the plot
    
    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Handle single model case
    if not isinstance(error_rates[0], (list, np.ndarray)):
        error_rates = [error_rates]
        set_sizes = [set_sizes]
        
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(error_rates))]
    
    for i, (errors, sizes) in enumerate(zip(error_rates, set_sizes)):
        # Sort by set size
        sorted_indices = np.argsort(sizes)
        sorted_errors = np.array(errors)[sorted_indices]
        sorted_sizes = np.array(sizes)[sorted_indices]
        
        # Calculate AUARC
        auarc = np.trapz(sorted_errors, sorted_sizes)
        max_size = np.max(sorted_sizes)
        if max_size > 0:
            auarc /= max_size
        
        ax.plot(sorted_sizes, sorted_errors, lw=2, 
                label=f'{model_names[i]} (AUARC = {auarc:.3f})')
    
    ax.set_xlabel('Average Set Size')
    ax.set_ylabel('Error Rate (1 - Coverage)')
    ax.set_title(title)
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax


def plot_reliability_diagram(y_true: np.ndarray, y_probs: np.ndarray, 
                            n_bins: int = 10,
                            model_names: Optional[List[str]] = None,
                            ax: Optional[plt.Axes] = None,
                            title: str = "Reliability Diagram",
                            save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot a reliability diagram (calibration curve) for one or more models.
    
    A reliability diagram shows the relationship between predicted probabilities
    and the actual fraction of positive samples. A perfectly calibrated model
    would have points along the diagonal.
    
    Args:
        y_true: Ground truth labels
        y_probs: List of probability arrays for each model
        n_bins: Number of bins for calibration
        model_names: Names of the models for the legend
        ax: Matplotlib axes to plot on
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Handle single model case
    if not isinstance(y_probs[0], (list, np.ndarray, torch.Tensor)) or len(y_probs[0]) == 1:
        y_probs = [y_probs]
        
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(y_probs))]
    
    for i, probs in enumerate(y_probs):
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true
            
        # For multi-class, extract the probability of the predicted class
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            y_pred = np.argmax(probs, axis=1)
            confidences = np.array([probs[i, pred] for i, pred in enumerate(y_pred)])
        else:
            confidences = probs.copy()
        
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        # Calculate accuracy and confidence for each bin
        for j in range(n_bins):
            bin_mask = (confidences >= bin_edges[j]) & (confidences < bin_edges[j+1])
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(y_pred[bin_mask] == y_true_np[bin_mask])
                bin_conf = np.mean(confidences[bin_mask])
                bin_count = np.sum(bin_mask)
                
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Calculate ECE
        ece = calculate_ece(y_true_np, probs, n_bins)
        
        # Plot calibration curve
        ax.plot(bin_centers, bin_accuracies, marker='o', linestyle='-', 
                label=f'{model_names[i]} (ECE = {ece:.3f})')
        
        # Plot histogram of confidence distribution
        ax2 = ax.twinx()
        ax2.bar(bin_centers, np.array(bin_counts) / np.sum(bin_counts), 
                alpha=0.1, width=1/n_bins, color='gray')
        ax2.set_ylabel('Fraction of Samples')
        ax2.set_ylim(0, 1)
    
    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax


def plot_conformal_calibration(confidence_levels: np.ndarray, actual_coverages: List[float],
                              model_names: Optional[List[str]] = None,
                              ax: Optional[plt.Axes] = None,
                              title: str = "Conformal Calibration",
                              save_path: Optional[str] = None) -> plt.Axes:
    """
    Plot a calibration curve for conformal prediction.
    
    This shows how well the claimed confidence levels match the actual coverage.
    
    Args:
        confidence_levels: Array of confidence levels (1-alpha)
        actual_coverages: List of actual coverage values for each model
        model_names: Names of the models for the legend
        ax: Matplotlib axes to plot on
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Handle single model case
    if not isinstance(actual_coverages[0], (list, np.ndarray)):
        actual_coverages = [actual_coverages]
        
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(actual_coverages))]
    
    for i, coverages in enumerate(actual_coverages):
        # Calculate ECE
        ece = np.mean(np.abs(np.array(coverages) - confidence_levels))
        
        ax.plot(confidence_levels, coverages, marker='o', linestyle='-', 
                label=f'{model_names[i]} (ECE = {ece:.3f})')
    
    # Plot the perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    ax.set_xlabel('Target Confidence Level (1-α)')
    ax.set_ylabel('Actual Coverage')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return ax


def plot_metrics_over_epochs(epochs: List[int], 
                            auroc_values: List[float], 
                            auarc_values: List[float],
                            ece_values: Optional[List[float]] = None,
                            model_names: Optional[List[str]] = None,
                            title: str = "Metrics Over Training Epochs",
                            log_scale: bool = False,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot AUROC, AUARC, and ECE metrics over training epochs.
    
    Args:
        epochs: List of epoch numbers
        auroc_values: List of AUROC values for each model over epochs
        auarc_values: List of AUARC values for each model over epochs
        ece_values: Optional list of ECE values for each model over epochs
        model_names: Names of the models for the legend
        title: Plot title
        log_scale: Whether to use log scale for the y-axis
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure with the plot
    """
    n_plots = 3 if ece_values is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 8, 6))
    
    # Handle single model case
    if not isinstance(auroc_values[0], (list, np.ndarray)):
        auroc_values = [auroc_values]
        auarc_values = [auarc_values]
        if ece_values is not None:
            ece_values = [ece_values]
        
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(auroc_values))]
    
    for i, (auroc, auarc) in enumerate(zip(auroc_values, auarc_values)):
        axes[0].plot(epochs, auroc, lw=2, marker='o', label=model_names[i])
        axes[1].plot(epochs, auarc, lw=2, marker='o', label=model_names[i])
        
        if ece_values is not None:
            axes[2].plot(epochs, ece_values[i], lw=2, marker='o', label=model_names[i])
    
    # AUROC subplot
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('AUROC (higher is better)')
    axes[0].set_title('AUROC Over Epochs')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUARC subplot
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUARC (lower is better)')
    axes[1].set_title('AUARC Over Epochs')
    if log_scale:
        axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # ECE subplot (if provided)
    if ece_values is not None:
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('ECE (lower is better)')
        axes[2].set_title('ECE Over Epochs')
        if log_scale:
            axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_epoch_metrics(epochs: List[int], 
                         auroc_values: List[float], 
                         auarc_values: List[float],
                         ece_values: Optional[List[float]] = None,
                         coverage_values: Optional[List[float]] = None,
                         size_values: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Analyze metrics over epochs to identify optimal epochs and trends.
    
    Args:
        epochs: List of epoch numbers
        auroc_values: List of AUROC values over epochs
        auarc_values: List of AUARC values over epochs
        ece_values: Optional list of ECE values over epochs
        coverage_values: Optional list of coverage values over epochs
        size_values: Optional list of set size values over epochs
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Find best epochs for each metric
    best_auroc_idx = np.argmax(auroc_values)
    best_auarc_idx = np.argmin(auarc_values)  # Lower AUARC is better
    
    results['best_auroc'] = {
        'epoch': epochs[best_auroc_idx],
        'value': auroc_values[best_auroc_idx]
    }
    
    results['best_auarc'] = {
        'epoch': epochs[best_auarc_idx],
        'value': auarc_values[best_auarc_idx]
    }
    
    # Add ECE analysis if provided
    if ece_values is not None:
        best_ece_idx = np.argmin(ece_values)  # Lower ECE is better
        results['best_ece'] = {
            'epoch': epochs[best_ece_idx],
            'value': ece_values[best_ece_idx]
        }
    
    # Calculate trends (improvement over epochs)
    if len(epochs) > 1:
        auroc_trend = (auroc_values[-1] - auroc_values[0]) / len(epochs)
        auarc_trend = (auarc_values[-1] - auarc_values[0]) / len(epochs)
        
        results['auroc_trend'] = auroc_trend
        results['auarc_trend'] = auarc_trend
        
        if ece_values is not None:
            ece_trend = (ece_values[-1] - ece_values[0]) / len(epochs)
            results['ece_trend'] = ece_trend
    
    # If coverage and size are provided, find optimal trade-off
    if coverage_values and size_values:
        # Define a simple trade-off metric: coverage / set_size
        trade_off = np.array(coverage_values) / np.array(size_values)
        best_trade_off_idx = np.argmax(trade_off)
        
        results['best_trade_off'] = {
            'epoch': epochs[best_trade_off_idx],
            'coverage': coverage_values[best_trade_off_idx],
            'size': size_values[best_trade_off_idx],
            'trade_off': trade_off[best_trade_off_idx]
        }
    
    return results


def save_metrics_to_csv(epochs: List[int], 
                       auroc_values: List[float], 
                       auarc_values: List[float],
                       ece_values: Optional[List[float]] = None,
                       coverage_values: Optional[List[float]] = None,
                       size_values: Optional[List[float]] = None,
                       model_name: str = "model",
                       save_dir: str = "./results") -> str:
    """
    Save metrics to a CSV file.
    
    Args:
        epochs: List of epoch numbers
        auroc_values: List of AUROC values over epochs
        auarc_values: List of AUARC values over epochs
        ece_values: Optional list of ECE values over epochs
        coverage_values: Optional list of coverage values over epochs
        size_values: Optional list of set size values over epochs
        model_name: Name of the model
        save_dir: Directory to save the CSV file
    
    Returns:
        Path to the saved CSV file
    """
    import pandas as pd
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    data = {
        'epoch': epochs,
        'auroc': auroc_values,
        'auarc': auarc_values
    }
    
    if ece_values is not None:
        data['ece'] = ece_values
    
    if coverage_values:
        data['coverage'] = coverage_values
    
    if size_values:
        data['set_size'] = size_values
    
    df = pd.DataFrame(data)
    
    # Calculate additional metrics
    if coverage_values and size_values:
        df['efficiency'] = df['coverage'] / df['set_size']
    
    file_path = os.path.join(save_dir, f"{model_name}_metrics.csv")
    df.to_csv(file_path, index=False)
    
    return file_path


if __name__ == "__main__":
    # This module is not meant to be run directly
    pass 