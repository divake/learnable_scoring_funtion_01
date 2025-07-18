# src/core/metrics.py

import torch
import numpy as np

def compute_tau(cal_loader, scoring_fn, base_model, device, coverage_target=0.9, tau_config=None):
    """
    Compute tau threshold for desired coverage on calibration set with constraints and smoothing
    Using vectorized approach for efficiency
    
    Modified to handle cached outputs (base_model can be None if inputs are already probabilities)
    """
    if tau_config is None:
        raise ValueError("tau_config must be provided with min, max, and window_size values")
    
    scoring_fn.eval()
    all_scores = []
    
    # Check if we're using cached outputs (base_model is None)
    using_cached = base_model is None
    
    if not using_cached:
        base_model.eval()
    
    with torch.no_grad():
        for inputs, targets in cal_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get softmax probabilities from base model or use inputs directly if cached
            if using_cached:
                # Inputs are already probabilities
                probs = inputs
            else:
                # Compute probabilities from base model
                logits = base_model(inputs)
                probs = torch.softmax(logits, dim=1)
            
            # New approach: scoring function outputs scores for all classes directly
            scores = scoring_fn(probs)  # Shape: (batch_size, num_classes)
            
            # Extract only true class scores for tau calculation
            true_scores = scores[torch.arange(len(targets)), targets]
            all_scores.append(true_scores.cpu())
    
    # Concatenate all scores and labels
    all_scores = torch.cat(all_scores, dim=0)
    
    # Sort scores for quantile computation
    sorted_scores, _ = torch.sort(all_scores)
    
    # Compute index for the desired quantile
    # Use ceiling instead of floor to ensure we meet or exceed target coverage
    idx = int(np.ceil(coverage_target * len(sorted_scores))) - 1
    idx = max(0, min(idx, len(sorted_scores) - 1))  # Safety bounds check
    
    # Apply smoothing around the quantile with more emphasis on lower values
    # to ensure we maintain coverage
    window_size = tau_config['window_size']
    if window_size > 0:
        start_idx = max(0, idx - window_size)
        end_idx = min(len(sorted_scores) - 1, idx + window_size // 2)  # Asymmetric window
        window_scores = sorted_scores[start_idx:end_idx+1]
        # Weight lower scores more to ensure coverage
        # Get smoothing weights from tau_config
        weight_start = tau_config.get('smoothing_weights', {}).get('start', 1.5)
        weight_end = tau_config.get('smoothing_weights', {}).get('end', 1.0)
        weights = torch.linspace(weight_start, weight_end, len(window_scores))
        tau = (window_scores * weights).sum() / weights.sum()
    else:
        tau = sorted_scores[idx].item()
    
    # Clamp tau to reasonable range
    tau = max(tau_config['min'], min(tau_config['max'], tau))
    
    return tau

def compute_coverage_and_size(prediction_sets, targets):
    """
    Compute coverage and average set size
    """
    covered = 0
    total_size = 0
    
    for pred_set, target in zip(prediction_sets, targets):
        if target in pred_set:
            covered += 1
        total_size += len(pred_set)
    
    coverage = covered / len(targets)
    avg_size = total_size / len(targets)
    
    return coverage, avg_size

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count