# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.metrics import AverageMeter

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, lambda1=1.0, lambda2=0.1):
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def train_epoch(self, optimizer, tau):
        """Train for one epoch"""
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
                class_probs = probs[:, i:i+1]  # [batch_size, 1]
                scores[:, i:i+1] = self.scoring_fn(class_probs)  # [batch_size, 1]
            
            # Compute coverage loss (using smooth indicators)
            target_scores = scores[torch.arange(batch_size), targets]
            coverage_indicators = torch.sigmoid(50.0 * (tau - target_scores))
            coverage = coverage_indicators.mean()
            
            # Compute size loss (using smooth indicators)
            size_indicators = torch.sigmoid(50.0 * (tau - scores))
            avg_size = size_indicators.mean(dim=1).mean()
            
            # Compute combined loss
            loss = self.lambda1 * (1 - coverage) + self.lambda2 * avg_size
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update meters with exact metrics
            with torch.no_grad():
                exact_coverage = (target_scores <= tau).float().mean()
                exact_size = (scores <= tau).float().sum(dim=1).mean()
                
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
                batch_size = inputs.size(0)
                
                # Get softmax probabilities
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                # Get scores for each class probability
                scores = torch.zeros_like(probs, device=self.device)
                for i in range(probs.size(1)):
                    class_probs = probs[:, i:i+1]
                    scores[:, i:i+1] = self.scoring_fn(class_probs)
                
                # Compute coverage
                target_scores = scores[torch.arange(batch_size), targets]
                coverage = (target_scores <= tau).float().mean()
                
                # Compute set size
                sizes = (scores <= tau).float().sum(dim=1)
                avg_size = sizes.mean()
                
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg