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
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, lambda1=1.0, lambda2=2.0):
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        
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

    def compute_batch_scores(self, probs):
        """Compute scores efficiently in chunks"""
        batch_size = probs.size(0)
        scores = torch.zeros(batch_size, self.num_classes, device=self.device)
        
        chunk_size = 1000
        flat_probs = probs.reshape(-1, 1)
        
        for i in range(0, flat_probs.size(0), chunk_size):
            end_idx = min(i + chunk_size, flat_probs.size(0))
            chunk_scores = self.scoring_fn(flat_probs[i:end_idx])
            scores.view(-1)[i:end_idx] = chunk_scores.squeeze()
        
        return scores


    def train_epoch(self, optimizer, tau, epoch):
        self.scoring_fn.train()
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        def compute_prediction_sets(scores, tau):
            pred_sets = scores <= tau
            empty_preds = ~pred_sets.any(dim=1)
            if empty_preds.any():
                min_scores, min_indices = scores[empty_preds].min(dim=1)
                pred_sets[empty_preds, min_indices] = True
            return pred_sets

        # Get tau with decay schedule
        tau = torch.tensor(tau, device=self.device)
        
        pbar = tqdm(self.train_loader)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Get probabilities and scores
            with torch.no_grad():
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
            
            scores = self.compute_batch_scores(probs)
            
            # Score consistency loss
            score_mean = scores.mean()
            score_std = scores.std()
            consistency_loss = F.mse_loss(score_mean, 
                                        self.scoring_fn.target_mean.clone().detach().to(self.device)) + \
                            F.mse_loss(score_std, 
                                        self.scoring_fn.target_std.clone().detach().to(self.device))
            
            # Original losses
            target_scores = scores[torch.arange(batch_size), targets]
            mask = torch.ones_like(scores, dtype=bool)
            mask[torch.arange(batch_size), targets] = False
            false_scores = scores[mask].view(batch_size, -1)
            
            margin_loss = F.softplus(target_scores - false_scores.min(dim=1)[0] + self.margin).mean()
            
            target_covered = (target_scores <= tau).float()
            coverage = target_covered.mean()
            coverage_error = coverage - self.target_coverage
            coverage_loss = coverage_error.pow(2)
            
            pred_sets = compute_prediction_sets(scores, tau)
            set_sizes = pred_sets.float().sum(dim=1)
            avg_size = set_sizes.mean()
            
            # Add strong penalty for violating minimum size constraint
            min_size_violation = F.relu(1.0 - set_sizes).mean()
            size_penalty = 10.0 * min_size_violation  # Large weight to enforce constraint

            size_loss = F.huber_loss(
                avg_size,
                torch.tensor(self.target_size, device=self.device, dtype=torch.float),
                delta=0.5
            )
            
            # Combined loss
            loss = (
                2.0 * coverage_loss +
                0.5 * margin_loss +
                1.0 * size_loss +  # Increased weight
                0.1 * consistency_loss +
                0.0001 * self.scoring_fn.l2_reg
            )
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            with torch.no_grad():
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