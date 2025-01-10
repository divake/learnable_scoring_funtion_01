# src/utils/metrics.py

import torch
import numpy as np

def compute_tau(cal_loader, scoring_fn, base_model, device, coverage_target=0.9, epoch=0):
    """Compute tau with guaranteed minimum set size"""
    scoring_fn.eval()
    base_model.eval()
    all_scores = []
    max_scores = []  # Track maximum score for each sample
    
    with torch.no_grad():
        for inputs, targets in cal_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Compute scores for all classes
            batch_scores = []
            chunk_size = 1000
            flat_probs = probs.reshape(-1, 1)
            for i in range(0, flat_probs.size(0), chunk_size):
                end_idx = min(i + chunk_size, flat_probs.size(0))
                chunk_scores = scoring_fn(flat_probs[i:end_idx])
                batch_scores.append(chunk_scores)
            batch_scores = torch.cat(batch_scores).view(probs.size(0), -1)
            
            # Get scores for true classes
            true_scores = batch_scores[torch.arange(len(targets)), targets]
            all_scores.append(true_scores.cpu())
            
            # Track maximum scores
            max_scores.append(batch_scores.max(dim=1)[0].cpu())
    
    all_scores = torch.cat(all_scores, dim=0)
    max_scores = torch.cat(max_scores, dim=0)
    
    # Get initial tau from quantile
    sorted_scores, _ = torch.sort(all_scores)
    idx = int(coverage_target * len(sorted_scores))
    base_tau = sorted_scores[idx].item()
    
    # Ensure tau is higher than minimum score for each sample
    min_max_score = max_scores.min().item()
    base_tau = max(base_tau, min_max_score)
    
    # Apply exponential decay schedule
    decay_rate = 0.98
    decay_steps = 5
    current_step = epoch // decay_steps
    target_tau = 0.4
    tau = target_tau + (base_tau - target_tau) * (decay_rate ** current_step)
    
    # Final bounds check
    tau = max(0.01, min(0.9, tau))
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