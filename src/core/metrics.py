# src/core/metrics.py

import torch
import numpy as np

def compute_tau(cal_loader, scoring_fn, base_model, device, coverage_target=0.9, tau_config=None):
    """
    Compute tau threshold for desired coverage on calibration set with constraints and smoothing
    Using vectorized approach for efficiency
    """
    if tau_config is None:
        raise ValueError("tau_config must be provided with min, max, and window_size values")
    
    scoring_fn.eval()
    base_model.eval()
    all_scores = []
    
    with torch.no_grad():
        for inputs, targets in cal_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get softmax probabilities from base model
            logits = base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Vectorized approach to compute all scores at once
            batch_size, num_classes = probs.shape
            flat_probs = probs.reshape(-1, 1)
            flat_scores = scoring_fn(flat_probs)
            scores = flat_scores.reshape(batch_size, num_classes)
            
            # Extract only true class scores for tau calculation
            true_scores = scores[torch.arange(len(targets)), targets]
            all_scores.append(true_scores.cpu())
    
    # Concatenate all scores and labels
    all_scores = torch.cat(all_scores, dim=0)
    
    # Sort scores for quantile computation
    sorted_scores, _ = torch.sort(all_scores)
    
    # Compute index for the desired quantile
    idx = int(coverage_target * len(sorted_scores))
    
    # Apply smoothing around the quantile
    start_idx = max(0, idx - tau_config['window_size'])
    end_idx = min(len(sorted_scores), idx + tau_config['window_size'])
    tau = sorted_scores[start_idx:end_idx].mean().item()
    
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