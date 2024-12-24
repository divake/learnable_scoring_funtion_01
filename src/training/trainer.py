# src/training/trainer.py

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

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, lambda1=1.0, lambda2=2.0):
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
        self.tau_max = 0.7
        self.score_margin = 0.1
        self.target_size = 1.0
        self.max_size = 3.0

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
                probs = torch.softmax(logits, dim=1)  # [batch_size, 10]
            
            # Get scores for all classes at once
            scores = self.scoring_fn(probs)  # [batch_size, 10]
                
            # Get true and false class scores
            target_scores = scores[torch.arange(batch_size), targets]
            mask = torch.ones_like(scores, dtype=bool)
            mask[torch.arange(batch_size), targets] = False
            false_scores = scores[mask].view(batch_size, -1)
            
            # Coverage loss with smooth transition
            coverage_error = torch.mean(torch.relu(target_scores - tau))
            
            # Margin loss with distribution-aware weighting
            margin_loss = torch.mean(torch.relu(
                target_scores.unsqueeze(1) - false_scores + self.score_margin
            ))
            
            # Size control
            pred_sets = (scores <= tau).float()
            set_sizes = pred_sets.sum(dim=1)
            size_penalty = torch.mean((set_sizes - self.target_size) ** 2)
            
            # Combined loss with dynamic weighting
            loss = (
                2.0 * coverage_error +  # Coverage is important
                1.0 * margin_loss +     # Encourage separation
                0.5 * size_penalty +    # Control set size
                0.01 * self.scoring_fn.l2_reg  # L2 regularization
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                exact_coverage = (target_scores <= tau).float().mean()
                exact_size = set_sizes.mean()
                
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
        self.scoring_fn.eval()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Get probabilities
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                # Get scores for all classes at once
                scores = self.scoring_fn(probs)  # [batch_size, num_classes]
                
                # Get target scores
                target_scores = scores[torch.arange(batch_size), targets]
                coverage = (target_scores <= tau).float().mean()
                
                # Compute set sizes
                pred_sets = (scores <= tau).float()
                set_sizes = pred_sets.sum(dim=1)
                avg_size = set_sizes.mean()
                
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg