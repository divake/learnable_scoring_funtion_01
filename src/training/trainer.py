# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, lambda1=1.0, lambda2=1.0):
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Initialize EMA
        self.loss_ema = None
        self.coverage_ema = None
        self.size_ema = None
        self.ema_alpha = 0.9
    
    def train_epoch(self, optimizer, tau):
        """Train for one epoch with corrected set size calculation"""
        self.scoring_fn.train()
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Get softmax probabilities from base model
            with torch.no_grad():
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
            
            # Get scores for each class probability
            scores = torch.zeros_like(probs, device=self.device)
            for i in range(probs.size(1)):
                class_probs = probs[:, i:i+1]
                scores[:, i:i+1] = self.scoring_fn(class_probs)
            
            # Compute coverage with smooth indicators
            target_scores = scores[torch.arange(batch_size), targets]
            coverage_indicators = torch.sigmoid(50.0 * (tau - target_scores))
            coverage = coverage_indicators.mean()
            
            # Compute size with smooth indicators - FIXED
            pred_sets = scores <= tau
            set_sizes = pred_sets.float().sum(dim=1)  # Count classes per sample
            avg_size = torch.maximum(set_sizes.mean(), 
                                   torch.ones_like(set_sizes.mean()))  # Ensure minimum size of 1
            
            # Progressive size penalty
            size_penalty = torch.max(torch.tensor(0.0).to(self.device), 
                                   avg_size - 1.5) ** 2
            
            # Compute loss components
            coverage_loss = self.lambda1 * (1 - coverage)
            size_loss = self.lambda2 * (avg_size + size_penalty)
            
            # Combined loss
            loss = coverage_loss + size_loss
            
            # Apply EMA smoothing
            if self.loss_ema is None:
                self.loss_ema = loss.item()
                self.coverage_ema = coverage.item()
                self.size_ema = avg_size.item()
            else:
                self.loss_ema = self.ema_alpha * self.loss_ema + (1 - self.ema_alpha) * loss.item()
                self.coverage_ema = self.ema_alpha * self.coverage_ema + (1 - self.ema_alpha) * coverage.item()
                self.size_ema = self.ema_alpha * self.size_ema + (1 - self.ema_alpha) * avg_size.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update meters with exact (not smoothed) values for accurate monitoring
            with torch.no_grad():
                exact_pred_sets = scores <= tau
                exact_coverage = (target_scores <= tau).float().mean()
                exact_size = torch.maximum(
                    exact_pred_sets.float().sum(dim=1).mean(),
                    torch.ones_like(exact_pred_sets.float().sum(dim=1).mean())
                )
                
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
        """Evaluate model with corrected set size calculation"""
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
                
                # Fixed set size calculation for evaluation
                pred_sets = scores <= tau
                set_sizes = pred_sets.float().sum(dim=1)
                avg_size = torch.maximum(set_sizes.mean(), 
                                       torch.ones_like(set_sizes.mean()))
                
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg