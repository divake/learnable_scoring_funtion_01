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
                 device, lambda1=0.1, lambda2=0.1):
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Enhanced margin parameters
        self.initial_margin = 0.1
        self.max_margin = 0.5
        self.margin_growth = 0.02
        self.focal_gamma = 2.0
        
        # Modified hyperparameters
        self.target_coverage = 0.9
        self.target_size = 5
        self.margin = 0.2
        
        # Loss weights
        self.coverage_weight = 2.0
        self.margin_weight = 0.5
        self.size_weight = 0.5
        
        # Bounds
        self.tau_min = 0.1
        self.tau_max = 2.0
        self.num_classes = 100

    def get_margin(self, epoch):
        """Dynamic margin with warmup schedule"""
        return min(self.max_margin, 
                  self.initial_margin + epoch * self.margin_growth)
    
    def compute_margin_loss(self, target_scores, false_scores, epoch):
        """Enhanced margin loss with top-k consideration"""
        current_margin = self.get_margin(epoch)
        k = min(5, false_scores.size(1))  # Top-k false scores
        top_k_false = torch.topk(false_scores, k, largest=False)[0]
        
        margin_losses = F.softplus(
            target_scores.unsqueeze(1) - top_k_false + current_margin
        )
        return margin_losses.mean()
    
    def distribution_matching_loss(self, true_scores, false_scores):
        """Control distribution separation and spread"""
        true_mean = true_scores.mean()
        false_mean = false_scores.mean()
        
        # Enforce mean separation
        separation_loss = F.relu(0.3 - (false_mean - true_mean))
        
        # Control variance
        true_std = true_scores.std()
        false_std = false_scores.std()
        spread_loss = (F.mse_loss(true_std, torch.tensor(0.1, device=self.device)) + 
                      F.mse_loss(false_std, torch.tensor(0.1, device=self.device)))
        
        return separation_loss + 0.1 * spread_loss
    
    def focal_weight(self, scores):
        """Focal weighting for hard examples"""
        return (1 - torch.sigmoid(scores)).pow(self.focal_gamma)
    
    def compute_batch_scores(self, probs):
        """Compute scores efficiently in chunks"""
        batch_size = probs.size(0)
        scores = torch.zeros(batch_size, self.num_classes, device=self.device)
        
        chunk_size = 1000
        flat_probs = probs.reshape(-1, 1)  # Ensure 2D shape
        
        for i in range(0, flat_probs.size(0), chunk_size):
            end_idx = min(i + chunk_size, flat_probs.size(0))
            chunk_scores = self.scoring_fn(flat_probs[i:end_idx])
            scores.view(-1)[i:end_idx] = chunk_scores.squeeze()
        
        return scores

    def train_epoch(self, optimizer, tau, epoch, return_scores=False):
        self.scoring_fn.train()
        total_loss = 0
        all_true_scores = []
        all_false_scores = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Get softmax probabilities from base model
            with torch.no_grad():
                logits = self.base_model(data)
                probs = F.softmax(logits, dim=1)
            
            # Get target probabilities and scores efficiently
            target_probs = torch.gather(probs, 1, target.unsqueeze(1))
            target_scores = self.scoring_fn(target_probs)
            
            # Create mask for false classes - more efficient batch operation
            mask = torch.ones_like(probs, device=self.device)
            mask.scatter_(1, target.unsqueeze(1), 0)
            
            # Process all false probabilities in one go
            false_probs = probs[mask.bool()].view(batch_size, -1)  # Shape: [batch_size, num_false_classes]
            
            # Process scores in chunks to avoid memory issues
            chunk_size = 1024
            false_scores_list = []
            
            for i in range(0, false_probs.size(1), chunk_size):
                end_idx = min(i + chunk_size, false_probs.size(1))
                chunk_probs = false_probs[:, i:end_idx].unsqueeze(-1)  # Add feature dimension
                chunk_scores = self.scoring_fn(chunk_probs)
                false_scores_list.append(chunk_scores)
            
            # Concatenate all chunks
            false_scores = torch.cat(false_scores_list, dim=1)
            
            # Get top-k smallest false scores efficiently
            k = 5  # Number of hard negatives to consider
            top_k_false_scores, _ = torch.topk(false_scores, k=min(k, false_scores.size(1)), 
                                             dim=1, largest=False)
            
            # Compute losses with proper weighting
            margin_loss = self.compute_margin_loss(target_scores, top_k_false_scores, epoch)
            dist_loss = self.distribution_matching_loss(target_scores, false_scores)
            weights = self.focal_weight(target_scores)
            
            # Total loss with weighted components
            loss = (self.margin_weight * margin_loss * weights.mean() +
                   self.coverage_weight * dist_loss +
                   self.size_weight * self.scoring_fn.l2_reg)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if return_scores:
                all_true_scores.append(target_scores.detach().cpu())
                all_false_scores.append(false_scores.detach().cpu())
        
        avg_loss = total_loss / len(self.train_loader)
        
        # Compute coverage and size on training set
        train_coverage, train_size = self.evaluate(self.train_loader, tau)
        
        if return_scores:
            true_scores = torch.cat(all_true_scores, dim=0)
            false_scores = torch.cat(all_false_scores, dim=0)
            return avg_loss, train_coverage, train_size, true_scores, false_scores
        
        return avg_loss, train_coverage, train_size

    def evaluate(self, loader, tau):
        """Evaluate model with guaranteed minimum set size"""
        self.scoring_fn.eval()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        # Convert tau to tensor and clamp
        tau = torch.tensor(tau, device=self.device)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)
            
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                scores = self.compute_batch_scores(probs)
                
                # Ensure minimum set size
                pred_sets = scores <= tau
                empty_preds = ~pred_sets.any(dim=1)
                if empty_preds.any():
                    min_scores, min_indices = scores[empty_preds].min(dim=1)
                    pred_sets[empty_preds, min_indices] = True
                
                target_scores = scores[torch.arange(len(targets)), targets]
                coverage = (target_scores <= tau).float().mean()
                set_sizes = pred_sets.float().sum(dim=1)
                avg_size = set_sizes.mean()
                
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg