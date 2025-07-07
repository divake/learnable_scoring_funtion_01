"""
Advanced metrics for evaluating conformal prediction models.

This module provides functions to calculate and visualize:
1. AUROC (Area Under the Receiver Operating Characteristic curve)
2. ECE (Expected Calibration Error)

These metrics are particularly useful for evaluating conformal prediction models
and understanding calibration quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from typing import List, Dict, Tuple, Union, Optional, Any
import os
from sklearn.preprocessing import label_binarize


def calculate_auroc(y_true: np.ndarray, y_scores: np.ndarray, 
                multi_class: str = 'ovr', higher_is_better: bool = True,
                normalize_scores: bool = True, handle_errors: bool = True) -> float:
    """
    Calculate the Area Under the Receiver Operating Characteristic curve (AUROC).
    
    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores or probabilities
        multi_class: Strategy for multi-class classification:
                    'ovr' - One-vs-Rest (default)
                    'ovo' - One-vs-One
        higher_is_better: Whether higher scores indicate higher confidence.
                          Set to False for conformal scores where lower values
                          indicate higher confidence.
        normalize_scores: Whether to normalize scores to sum to 1.0. Set to False
                          if scores are already properly scaled or normalization
                          would distort their meaning.
        handle_errors: If True, catch and handle errors, returning 0.5 for failures.
                       If False, allow exceptions to propagate.
    
    Returns:
        AUROC score
    
    Note:
        For binary classification, y_scores should be the probability/score of the positive class.
        For multi-class, y_scores should be a 2D array of shape (n_samples, n_classes).
        
        For conformal prediction methods where lower scores indicate higher confidence:
        - Set higher_is_better=False to automatically invert the scores
        - For methods like '1-p', 'APS', 'LogMargin', or 'Sparsemax', lower scores 
          typically indicate higher confidence, so use higher_is_better=False
    """
    import logging
    from sklearn.metrics import roc_auc_score
    
    # Handle different input types
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Replace NaN or infinite values with safe defaults
    y_scores = np.nan_to_num(y_scores, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Verify that we have enough samples and unique classes
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        message = f"Need at least 2 classes for AUROC, found {len(unique_classes)}"
        if handle_errors:
            logging.warning(message)
            return 0.5
        else:
            raise ValueError(message)
    
    # Check for each class having examples
    for cls in unique_classes:
        if np.sum(y_true == cls) <= 1:
            message = f"Class {cls} has too few examples for reliable AUROC calculation"
            if handle_errors:
                logging.warning(message)
                # Continue with calculation but warn
            else:
                raise ValueError(message)
    
    # If higher scores are not better (like in conformal prediction lower scores indicate higher confidence)
    # invert the scores by negation or use 1-score depending on their range
    if not higher_is_better:
        # Check if scores are roughly in [0,1] range to decide transformation
        if np.min(y_scores) >= -0.1 and np.max(y_scores) <= 1.1:
            # For scores in [0,1], use 1-score to invert
            y_scores = 1.0 - y_scores
        else:
            # For scores outside [0,1], use negation to invert
            y_scores = -y_scores
        logging.info("Inverted scores since higher_is_better=False")
    
    # Determine if binary or multi-class classification
    is_binary = False
    
    # Case 1: Only 2 unique classes
    if len(unique_classes) == 2:
        is_binary = True
        
    # Case 2: One-dimensional scores array
    if len(y_scores.shape) == 1:
        is_binary = True
        
    # Case 3: Two-column scores array (typical binary classifier output)
    if len(y_scores.shape) > 1 and y_scores.shape[1] == 2:
        is_binary = True
        # Use probability of positive class (second column)
        y_scores = y_scores[:, 1]
    
    # Case 4: Single-column scores array
    if len(y_scores.shape) > 1 and y_scores.shape[1] == 1:
        is_binary = True
        y_scores = y_scores.flatten()
    
    if is_binary:
        # Binary classification
        try:
            # For binary, remap classes to 0 and 1
            # This handles cases where classes might be e.g., [2, 7] instead of [0, 1]
            binary_y_true = np.zeros_like(y_true)
            binary_y_true[y_true == unique_classes[1]] = 1
            
            return roc_auc_score(binary_y_true, y_scores)
        except Exception as e:
            message = f"Error calculating binary AUROC: {e}"
            if handle_errors:
                logging.error(message)
                return 0.5
            else:
                raise ValueError(message) from e
    else:
        # Multi-class classification
        # Check dimensions match
        if len(y_scores.shape) < 2 or y_scores.shape[1] != len(unique_classes):
            message = (f"Number of score columns ({y_scores.shape[1] if len(y_scores.shape) > 1 else 1}) "
                     f"doesn't match number of classes ({len(unique_classes)})")
            if handle_errors:
                logging.warning(message)
                # Try to continue by reshaping if possible
                if len(y_scores.shape) == 1 and len(y_scores) % len(unique_classes) == 0:
                    y_scores = y_scores.reshape(-1, len(unique_classes))
                else:
                    logging.error("Cannot reshape scores to match class count")
                    return 0.5
            else:
                raise ValueError(message)
        
        # Normalize scores if requested and necessary
        if normalize_scores:
            # Check if scores need normalization
            row_sums = np.sum(y_scores, axis=1)
            
            # Handle rows that sum to 0 or NaN
            zero_rows = np.isclose(row_sums, 0) | np.isnan(row_sums)
            if np.any(zero_rows):
                # For rows that sum to 0, replace with uniform distribution
                uniform_prob = 1.0 / len(unique_classes)
                for i in np.where(zero_rows)[0]:
                    y_scores[i, :] = uniform_prob
                
                # Recalculate row sums
                row_sums = np.sum(y_scores, axis=1)
                
            # Normalize scores to sum to 1.0
            if not np.allclose(row_sums, 1.0):
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                row_sums = row_sums + epsilon
                y_scores = y_scores / row_sums[:, np.newaxis]
        
        # Final check for any remaining NaN values
        y_scores = np.nan_to_num(y_scores, nan=1.0/len(unique_classes))
        
        try:
            # For one-vs-rest approach, binarize labels with explicit classes to handle non-consecutive indices
            if multi_class == 'ovr':
                # Create proper binary matrix for ground truth labels
                y_true_bin = np.zeros((len(y_true), len(unique_classes)))
                for i, cls in enumerate(unique_classes):
                    y_true_bin[:, i] = (y_true == cls).astype(int)
                
                # If only 2 classes, use simpler computation
                if len(unique_classes) == 2:
                    return roc_auc_score(y_true_bin[:, 1], y_scores[:, 1])
                else:
                    return roc_auc_score(y_true_bin, y_scores, multi_class=multi_class, average='macro')
            else:
                # For one-vs-one, map classes to consecutive integers first
                class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
                y_true_mapped = np.array([class_mapping[cls] for cls in y_true])
                
                return roc_auc_score(y_true_mapped, y_scores, multi_class=multi_class, average='macro')
                
        except Exception as e:
            message = f"Error calculating multi-class AUROC: {e}"
            if handle_errors:
                logging.error(message)
                return 0.5
            else:
                raise ValueError(message) from e


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
    Plot the ROC curve for one or more models, supporting both binary and multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_scores: Score arrays for each model. For multi-class, this should be of shape (n_samples, n_classes)
        model_names: Names of the models for the legend
        ax: Matplotlib axes to plot on
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Convert tensors to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Determine if we're dealing with multi-class classification
    multi_class = len(y_scores.shape) > 1 and y_scores.shape[1] > 2
    
    if multi_class:
        # Multi-class classification
        # Binarize the labels for multi-class ROC curve
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot micro-average ROC curve
        ax.plot(fpr["micro"], tpr["micro"], label=f'micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=2)
        
        # Plot ROC curves for first few classes (limited to 5 for readability)
        n_classes_to_plot = min(5, len(classes))
        colors = plt.cm.get_cmap('tab10', n_classes_to_plot)
        
        for i in range(n_classes_to_plot):
            ax.plot(fpr[i], tpr[i], color=colors(i),
                    lw=2, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.3f})')
    else:
        # Binary classification
        # If scores has two columns, use the second column (probability of positive class)
        if len(y_scores.shape) > 1 and y_scores.shape[1] == 2:
            y_scores = y_scores[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        model_label = model_names[0] if model_names else "Model"
        ax.plot(fpr, tpr, lw=2, label=f'{model_label} (AUC = {roc_auc:.3f})')
    
    # Add diagonal line for reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Add styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
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
        plt.close()
    
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
        plt.close()
    
    return ax


def plot_metrics_over_epochs(epochs: List[int], 
                            auroc_values: List[float], 
                            ece_values: Optional[List[float]] = None,
                            model_names: Optional[List[str]] = None,
                            title: str = "Metrics Over Training Epochs",
                            log_scale: bool = False,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot AUROC and ECE metrics over training epochs.
    
    Args:
        epochs: List of epoch numbers
        auroc_values: List of AUROC values for each model over epochs
        ece_values: Optional list of ECE values for each model over epochs
        model_names: Names of the models for the legend
        title: Plot title
        log_scale: Whether to use log scale for the y-axis
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure with the plot
    """
    n_plots = 2 if ece_values is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 8, 6))
    
    # Ensure axes is always a list
    if n_plots == 1:
        axes = [axes]
    
    # Handle single model case
    if not isinstance(auroc_values[0], (list, np.ndarray)):
        auroc_values = [auroc_values]
        if ece_values is not None:
            ece_values = [ece_values]
        
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(auroc_values))]
    
    for i, auroc in enumerate(auroc_values):
        axes[0].plot(epochs, auroc, lw=2, marker='o', label=model_names[i])
        
        if ece_values is not None:
            axes[1].plot(epochs, ece_values[i], lw=2, marker='o', label=model_names[i])
    
    # AUROC subplot
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('AUROC (higher is better)')
    axes[0].set_title('AUROC Over Epochs')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ECE subplot (if provided)
    if ece_values is not None:
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('ECE (lower is better)')
        axes[1].set_title('ECE Over Epochs')
        if log_scale:
            axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def analyze_epoch_metrics(epochs: List[int], 
                         auroc_values: List[float], 
                         ece_values: Optional[List[float]] = None,
                         coverage_values: Optional[List[float]] = None,
                         size_values: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Analyze metrics over epochs to identify optimal epochs and trends.
    
    Args:
        epochs: List of epoch numbers
        auroc_values: List of AUROC values over epochs
        ece_values: Optional list of ECE values over epochs
        coverage_values: Optional list of coverage values over epochs
        size_values: Optional list of set size values over epochs
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Find best epochs for each metric
    best_auroc_idx = np.argmax(auroc_values)
    
    results['best_auroc'] = {
        'epoch': epochs[best_auroc_idx],
        'value': auroc_values[best_auroc_idx]
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
        
        results['auroc_trend'] = auroc_trend
        
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
        'auroc': auroc_values
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


def calculate_conformal_metrics(y_true: np.ndarray, 
                               scores: np.ndarray,
                               base_probs: np.ndarray,
                               tau: float) -> Dict[str, float]:
    """
    Calculate all metrics for conformal prediction in one function.
    
    This function properly handles the different semantics of scores vs probabilities
    in conformal prediction methods.
    
    Args:
        y_true: Ground truth labels
        scores: Conformal scores (lower is better) of shape (n_samples, n_classes)
        base_probs: Original probabilities from base model of shape (n_samples, n_classes)
        tau: Current tau threshold for conformal prediction
    
    Returns:
        Dictionary containing AUROC, ECE, coverage, and average set size
    """
    # Handle tensor inputs
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(base_probs, torch.Tensor):
        base_probs = base_probs.cpu().numpy()
    
    # 1. AUROC - Use base model probabilities (already proper probabilities)
    auroc = calculate_auroc(y_true, base_probs)
    
    # 2. ECE - Use base model probabilities to measure calibration
    ece = calculate_ece(y_true, base_probs)
    
    # 4. Coverage and set size at current tau
    pred_sets = scores <= tau
    set_sizes = np.sum(pred_sets, axis=1)
    
    # Calculate coverage
    correct = 0
    for i, label in enumerate(y_true):
        if pred_sets[i, label]:
            correct += 1
    coverage = correct / len(y_true)
    avg_set_size = np.mean(set_sizes)
    
    return {
        'auroc': auroc,
        'ece': ece,
        'coverage': coverage,
        'avg_set_size': avg_set_size
    }


if __name__ == "__main__":
    # This module is not meant to be run directly
    pass 