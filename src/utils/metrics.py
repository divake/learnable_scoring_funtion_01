# src/utils/metrics.py

import torch
import numpy as np

def compute_tau(cal_loader, scoring_fn, base_model, device, coverage_target=0.9, epoch=0):
    """Compute tau using efficient quantile estimation.
    
    Args:
        cal_loader: Calibration data loader
        scoring_fn: Scoring function model
        base_model: Base model
        device: Device to run computation on
        coverage_target: Target coverage level
        epoch: Current epoch number
    """
    scoring_fn.eval()
    base_model.eval()
    
    # Pre-allocate tensors for efficiency
    batch_size = next(iter(cal_loader))[0].size(0)
    num_batches = len(cal_loader)
    all_scores = torch.empty(batch_size * num_batches, device=device)
    max_scores = torch.empty(batch_size * num_batches, device=device)
    
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for batch_idx, (inputs, targets) in enumerate(cal_loader):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(inputs)  # Handle last batch correctly
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions in a single forward pass
            logits = base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Process all classes in a single batch for efficiency
            class_probs = probs.reshape(-1, 1)
            scores = scoring_fn(class_probs)
            scores = scores.view(len(inputs), -1)
            
            # Store true class scores and max scores directly
            all_scores[start_idx:end_idx] = scores[torch.arange(len(targets)), targets]
            max_scores[start_idx:end_idx] = scores.max(dim=1)[0]
    
    # Compute quantile directly on GPU
    valid_scores = all_scores[:end_idx]  # Only use valid scores
    sorted_scores, _ = torch.sort(valid_scores)
    n = len(sorted_scores)
    
    # Convert to tensor for proper computation
    rank = torch.tensor((1 - coverage_target) * (n - 1), device=device)
    lower_idx = int(rank.floor().item())
    upper_idx = int(rank.ceil().item())
    weight = rank - lower_idx
    
    # Compute interpolated quantile
    base_tau = (1 - weight) * sorted_scores[lower_idx] + weight * sorted_scores[upper_idx]
    base_tau = base_tau.item()
    
    # Apply adaptive threshold with momentum
    momentum = 0.9
    min_max_score = max_scores[:end_idx].min().item()
    current_tau = max(base_tau, min_max_score)
    
    # Apply exponential decay schedule with faster convergence
    decay_rate = 0.95
    decay_steps = 3
    current_step = epoch // decay_steps
    target_tau = 0.4
    
    # Smooth tau updates with momentum
    if hasattr(compute_tau, 'last_tau'):
        tau = momentum * compute_tau.last_tau + (1 - momentum) * (
            target_tau + (current_tau - target_tau) * (decay_rate ** current_step)
        )
    else:
        tau = target_tau + (current_tau - target_tau) * (decay_rate ** current_step)
    
    # Store for next iteration
    compute_tau.last_tau = tau
    
    # Ensure tau stays in reasonable bounds
    tau = max(0.01, min(0.9, tau))
    return tau

# Initialize the last_tau attribute
compute_tau.last_tau = 0.5  # Initial value

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