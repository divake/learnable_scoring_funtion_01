#src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, test_loader, 
                 device, lambda1, lambda2):
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        
        # Core objectives
        self.target_coverage = 0.9
        self.min_set_size = 1.0
        self.max_set_size = 5.0
        
        # Loss weights (reusing lambda1 and lambda2)
        self.coverage_weight = lambda1  # Weight for coverage loss
        self.size_weight = lambda2      # Weight for set size loss
        
        # For numerical stability
        self.eps = 1e-6

    def compute_loss(self, target_scores, false_scores, tau):
        """Compute loss with improved set size calculation and stronger weights.
        
        Args:
            target_scores: Scores for true class (shape: [batch_size, 1])
            false_scores: Scores for false classes (shape: [batch_size, num_classes-1])
            tau: Current threshold
            
        Returns:
            tuple: (total_loss, coverage, average_set_size)
        """
        batch_size = target_scores.size(0)
        
        # 1. Coverage Loss: Ensure target class is in the set
        target_in_set = (target_scores <= tau).float()
        coverage = target_in_set.mean()
        
        # Use BCEWithLogitsLoss for better numerical stability with AMP
        coverage_loss = 10.0 * F.binary_cross_entropy_with_logits(
            coverage.unsqueeze(0),  # Add batch dimension
            torch.tensor([self.target_coverage], device=self.device, dtype=torch.float32)
        )
        
        # 2. Set Size Loss: Control total set size with stronger penalties
        false_in_set = (false_scores <= tau).float()
        # Correct set size calculation: count number of classes included
        set_sizes = false_in_set.sum(dim=1) + target_in_set.squeeze()
        
        # Normalize set sizes by total number of classes for better loss scaling
        num_classes = false_scores.size(1) + 1  # add 1 for target class
        normalized_set_sizes = set_sizes / num_classes
        
        # Penalize deviations from desired set size range
        min_size_loss = F.smooth_l1_loss(
            normalized_set_sizes,
            torch.ones_like(normalized_set_sizes) * (self.min_set_size / num_classes),
            beta=0.5
        )
        max_size_loss = torch.relu(normalized_set_sizes - (self.max_set_size / num_classes)).mean()
        
        # 3. Distribution Loss: Enforce separation between true and false scores
        score_margin = 0.2
        distribution_loss = F.relu(
            target_scores - false_scores.min(dim=1)[0] + score_margin
        ).mean()
        
        # 4. Variance Control: Encourage appropriate spread of scores
        target_std = target_scores.std()
        false_std = false_scores.std()
        variance_loss = F.mse_loss(target_std, torch.tensor(0.1, device=self.device)) + \
                       F.mse_loss(false_std, torch.tensor(0.1, device=self.device))
        
        # Combine all losses with weights
        size_loss = 5.0 * min_size_loss + 5.0 * max_size_loss
        total_loss = (
            coverage_loss + 
            self.size_weight * size_loss + 
            0.1 * distribution_loss + 
            0.05 * variance_loss
        )
        
        # Return actual (non-normalized) set sizes for logging
        return (
            total_loss,
            coverage.detach().item(),
            set_sizes.mean().detach().item()  # Return actual set sizes
        )

    def train_epoch(self, optimizer, tau, epoch, return_scores=False):
        self.scoring_fn.train()
        total_loss = 0
        avg_coverage = 0
        avg_set_size = 0
        n_batches = 0
        
        all_true_scores = []
        all_false_scores = []
        
        # Enable automatic mixed precision training with new API
        scaler = torch.amp.GradScaler()
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Get softmax probabilities from base model
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = self.base_model(data)
                probs = F.softmax(logits, dim=1)
            
            # Process all classes in batches
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # Process classes in chunks for memory efficiency
                chunk_size = 20  # Process 20 classes at a time
                all_scores = []
                
                for i in range(0, probs.size(1), chunk_size):
                    end_idx = min(i + chunk_size, probs.size(1))
                    chunk_probs = probs[:, i:end_idx].reshape(-1, 1)
                    chunk_scores = self.scoring_fn(chunk_probs)
                    all_scores.append(chunk_scores.view(batch_size, -1))
                
                all_scores = torch.cat(all_scores, dim=1)  # [batch_size, num_classes]
                
                # Extract target and false scores
                target_scores = all_scores[torch.arange(batch_size), target].unsqueeze(1)
                
                # Get false scores efficiently
                mask = torch.ones_like(all_scores, device=self.device)
                mask[torch.arange(batch_size), target] = 0
                false_scores = all_scores[mask.bool()].view(batch_size, -1)
                
                # Compute loss
                loss, coverage, set_size = self.compute_loss(
                    target_scores, false_scores, tau
                )
                
                # Add L2 regularization if present
                if hasattr(self.scoring_fn, 'l2_reg'):
                    loss = loss + self.scoring_fn.l2_reg
            
            # Update with gradient scaling
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            total_loss += loss.item()
            avg_coverage += coverage
            avg_set_size += set_size
            n_batches += 1
            
            if return_scores:
                all_true_scores.append(target_scores.detach().cpu())
                all_false_scores.append(false_scores.detach().cpu())
            
            # Clear GPU cache periodically
            if n_batches % 50 == 0:
                torch.cuda.empty_cache()
        
        if return_scores:
            true_scores = torch.cat(all_true_scores, dim=0)
            false_scores = torch.cat(all_false_scores, dim=0)
            return (
                total_loss / n_batches,
                avg_coverage / n_batches,
                avg_set_size / n_batches,
                true_scores,
                false_scores
            )
        
        return (
            total_loss / n_batches,
            avg_coverage / n_batches,
            avg_set_size / n_batches
        )

    def evaluate(self, loader, tau):
        """Evaluate model with raw threshold-based set construction"""
        self.scoring_fn.eval()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Get predictions
                logits = self.base_model(inputs)
                probs = F.softmax(logits, dim=1)
                
                # Score all classes in chunks for memory efficiency
                chunk_size = 20  # Process 20 classes at a time
                all_scores = []
                
                for i in range(0, probs.size(1), chunk_size):
                    end_idx = min(i + chunk_size, probs.size(1))
                    chunk_probs = probs[:, i:end_idx].reshape(-1, 1)
                    chunk_scores = self.scoring_fn(chunk_probs)
                    all_scores.append(chunk_scores.view(batch_size, -1))
                
                scores = torch.cat(all_scores, dim=1)
                
                # Create prediction sets
                pred_sets = (scores <= tau)
                
                # Calculate metrics
                covered = pred_sets[torch.arange(batch_size), targets]
                set_sizes = pred_sets.float().sum(dim=1)  # Count number of classes in each set
                
                # Update metrics with actual counts
                coverage_meter.update(covered.float().mean().item(), batch_size)
                size_meter.update(set_sizes.mean().item(), batch_size)
                
                # Clear memory
                del scores, pred_sets
                torch.cuda.empty_cache()
        
        return coverage_meter.avg, size_meter.avg