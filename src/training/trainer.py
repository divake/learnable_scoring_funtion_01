# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, lambda1=1.0, lambda2=1.0):
        # Store all parameters as instance variables
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Additional instance variables
        self.size_target = 1.0
        self.prev_losses = []

    def train_epoch(self, optimizer, tau):
        self.scoring_fn.train()
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
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
            
            # Coverage loss with smooth approximation
            target_scores = scores[torch.arange(batch_size), targets]
            coverage_indicators = torch.sigmoid(50.0 * (tau - target_scores))
            coverage = coverage_indicators.mean()
            
            # Progressive size penalty
            pred_sets = scores <= tau
            set_sizes = pred_sets.float().sum(dim=1)
            avg_size = set_sizes.mean()
            
            # Multi-component size penalty
            base_penalty = torch.relu(avg_size - 1)
            exp_penalty = torch.exp(torch.relu(avg_size - 3.0))     
            size_penalty = 0.5 * base_penalty + 0.05 * exp_penalty  
            
            # Adaptive coverage-size trade-off
            coverage_loss = self.lambda1 * (1 - coverage)
            size_loss = self.lambda2 * size_penalty
            
            # Combined loss with stability term
            loss = coverage_loss + size_loss
        
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update meters
            loss_meter.update(loss.item())
            coverage_meter.update(coverage.item())
            size_meter.update(avg_size.item())
            
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