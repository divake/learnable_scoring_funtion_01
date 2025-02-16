# src/core/metrics.py

import torch
import numpy as np

def compute_tau(cal_loader, scoring_fn, base_model, device, coverage_target=0.9):
    """
    Compute tau threshold for desired coverage on calibration set with constraints and smoothing
    """
    # Constants for tau bounds
    tau_min = 0.1
    tau_max = 0.9
    window_size = 5  # For smoothing
    
    scoring_fn.eval()
    base_model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in cal_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get softmax probabilities from base model
            logits = base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Calculate scores for true class
            true_probs = probs[torch.arange(len(targets)), targets].unsqueeze(1)
            scores = scoring_fn(true_probs)
            
            all_scores.append(scores.cpu())
            all_labels.append(targets.cpu())
    
    # Concatenate all scores and labels
    all_scores = torch.cat(all_scores, dim=0).squeeze()
    
    # Sort scores for quantile computation
    sorted_scores, _ = torch.sort(all_scores)
    
    # Compute index for the desired quantile
    idx = int(coverage_target * len(sorted_scores))
    
    # Apply smoothing around the quantile
    start_idx = max(0, idx - window_size)
    end_idx = min(len(sorted_scores), idx + window_size)
    tau = sorted_scores[start_idx:end_idx].mean().item()
    
    # Clamp tau to reasonable range
    tau = max(tau_min, min(tau_max, tau))
    
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