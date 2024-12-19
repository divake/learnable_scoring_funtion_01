# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, lambda1=1.0, lambda2=2.0):  # Increased lambda2
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Constants
        self.tau_min = 0.1
        self.tau_max = 0.6  # Reduced max tau
        self.score_margin = 0.2  # Increased margin
        self.target_size = 1.2  # Target set size
        self.max_size = 3.0  # Maximum allowed set size

    def train_epoch(self, optimizer, tau):
        self.scoring_fn.train()
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        # Clamp tau to reasonable range
        tau = max(self.tau_min, min(self.tau_max, tau))
        
        pbar = tqdm(self.train_loader)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            with torch.no_grad():
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
            
            scores = torch.zeros_like(probs, device=self.device)
            for i in range(probs.size(1)):
                class_probs = probs[:, i:i+1]
                scores[:, i:i+1] = self.scoring_fn(class_probs)
            
            # Get true and false class scores
            target_scores = scores[torch.arange(batch_size), targets]
            mask = torch.ones_like(scores, dtype=bool)
            mask[torch.arange(batch_size), targets] = False
            false_scores = scores[mask].view(batch_size, -1)
            
            # Stronger margin loss
            margin_loss = torch.relu(target_scores - false_scores.min(dim=1)[0] + self.score_margin).mean()
            
            # Coverage loss with smoother transition
            coverage_indicators = torch.sigmoid(20.0 * (tau - target_scores))  # Reduced temperature
            coverage = coverage_indicators.mean()
            
            # Size penalty with stricter control
            pred_sets = scores <= tau
            set_sizes = pred_sets.float().sum(dim=1)
            avg_size = set_sizes.mean()
            
            # Progressive size penalty
            size_deviation = torch.abs(avg_size - self.target_size)
            quadratic_penalty = size_deviation ** 2
            exp_penalty = torch.exp(torch.relu(avg_size - self.max_size))
            size_penalty = quadratic_penalty + 0.1 * exp_penalty
            
            # Separation loss to encourage score spread
            score_spread = false_scores.mean(dim=1) - target_scores
            spread_loss = torch.relu(self.score_margin - score_spread).mean()
            
            # Combined loss with balanced terms
            loss = (
                self.lambda1 * (1 - coverage) +  # Coverage term
                self.lambda2 * size_penalty +    # Size penalty
                0.2 * margin_loss +             # Increased margin weight
                0.1 * spread_loss +             # Added spread loss
                0.01 * self.scoring_fn.l2_reg   # Reduced L2 weight
            )
            
            optimizer.zero_grad()
            loss.backward()
            # Reduced gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Update meters with exact metrics
            with torch.no_grad():
                exact_coverage = (target_scores <= tau).float().mean()
                exact_size = pred_sets.float().sum(dim=1).mean()
                
                loss_meter.update(loss.item())
                coverage_meter.update(exact_coverage.item())
                size_meter.update(exact_size.item())
            
            pbar.set_postfix({
                'Loss': f'{loss_meter.avg:.3f}',
                'Coverage': f'{coverage_meter.avg:.3f}',
                'Size': f'{size_meter.avg:.3f}'
            })
        
        return loss_meter.avg, coverage_meter.avg, size_meter.avg

    def evaluate(self, loader, tau):
        """Evaluate model"""
        self.scoring_fn.eval()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                scores = torch.zeros_like(probs, device=self.device)
                for i in range(probs.size(1)):
                    class_probs = probs[:, i:i+1]
                    scores[:, i:i+1] = self.scoring_fn(class_probs)
                
                target_scores = scores[torch.arange(len(targets)), targets]
                coverage = (target_scores <= tau).float().mean()
                
                pred_sets = scores <= tau
                set_sizes = pred_sets.float().sum(dim=1)
                avg_size = set_sizes.mean()
                
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg