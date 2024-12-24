# src/utils/metrics.py

import torch
import numpy as np

def compute_tau(cal_loader, scoring_fn, base_model, device, coverage_target=0.9):
    """Compute tau with stricter constraints"""
    tau_min = 0.1
    tau_max = 0.5
    window_size = 3
    
    scoring_fn.eval()
    base_model.eval()
    all_scores = []
    all_sizes = []
    
    with torch.no_grad():
        for inputs, targets in cal_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            
            # Get probabilities from base model
            logits = base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Get scores for all classes
            scores = scoring_fn(probs)
            
            # Create proper index tensor for gathering target scores
            batch_indices = torch.arange(batch_size, device=device)
            
            # Safely get target scores
            if scores.size(1) > targets.max().item():
                target_scores = scores[batch_indices, targets]
                
                # Track set sizes
                trial_tau = 0.3
                set_sizes = (scores <= trial_tau).float().sum(dim=1)
                
                all_scores.append(target_scores.cpu())
                all_sizes.append(set_sizes.cpu())
            else:
                continue
    
    if not all_scores:
        raise ValueError("No valid scores collected during calibration")
        
    all_scores = torch.cat(all_scores, dim=0)
    all_sizes = torch.cat(all_sizes, dim=0)
    
    # Get initial tau from desired coverage
    sorted_scores, _ = torch.sort(all_scores)
    idx = int(coverage_target * len(sorted_scores))
    
    # Apply smoothing and constraints
    start_idx = max(0, idx - window_size)
    end_idx = min(len(sorted_scores), idx + window_size)
    tau = sorted_scores[start_idx:end_idx].mean().item()
    
    return max(tau_min, min(tau_max, tau))

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