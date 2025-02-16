# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os

from src.utils.metrics import AverageMeter
from src.utils.visualization import (
    plot_training_curves, 
    plot_score_distributions,
    plot_set_size_distribution, 
    plot_scoring_function_behavior
)
from src.utils.metrics import compute_tau

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
        
    def train(self, num_epochs, target_coverage, tau_config, set_size_config, save_dir, plot_dir):
        """Main training loop"""
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.scoring_fn.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.2,
            div_factor=20,
            final_div_factor=100,
            anneal_strategy='cos'
        )
        
        # Training history
        history = {
            'epochs': [],
            'train_losses': [],
            'train_coverages': [],
            'train_sizes': [],
            'val_coverages': [],
            'val_sizes': [],
            'tau_values': []
        }
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Learning rate: {current_lr:.6f}")
            
            # Compute tau on calibration set
            tau = compute_tau(
                cal_loader=self.cal_loader,
                scoring_fn=self.scoring_fn,
                base_model=self.base_model,
                device=self.device,
                coverage_target=target_coverage
            )
            
            # Train epoch
            train_loss, train_coverage, train_size = self.train_epoch(
                optimizer=optimizer,
                tau=tau,
                tau_config=tau_config,
                set_size_config=set_size_config
            )
            
            # Evaluate
            val_coverage, val_size = self.evaluate(self.test_loader, tau)
            
            # Update scheduler
            scheduler.step()
            
            # Update history
            history['epochs'].append(epoch)
            history['train_losses'].append(train_loss)
            history['train_coverages'].append(train_coverage)
            history['train_sizes'].append(train_size)
            history['val_coverages'].append(val_coverage)
            history['val_sizes'].append(val_size)
            history['tau_values'].append(tau)
            
            # Log metrics
            logging.info(f"Train Loss: {train_loss:.4f}")
            logging.info(f"Train Coverage: {train_coverage:.4f}")
            logging.info(f"Train Set Size: {train_size:.4f}")
            logging.info(f"Val Coverage: {val_coverage:.4f}")
            logging.info(f"Val Set Size: {val_size:.4f}")
            logging.info(f"Tau: {tau:.4f}")
            
            # Update plots
            self._update_plots(history, tau, save_dir, plot_dir)
            
            # Save best model
            if train_loss < best_loss and train_size < set_size_config['max']:
                best_loss = train_loss
                torch.save(
                    self.scoring_fn.state_dict(),
                    os.path.join(save_dir, 'scoring_function_best.pth')
                )
                logging.info("Saved new best model")
    
    def train_epoch(self, optimizer, tau, tau_config, set_size_config):
        """Train for one epoch"""
        self.scoring_fn.train()
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        # Clamp tau to reasonable range
        tau = max(tau_config['min'], min(tau_config['max'], tau))
        
        pbar = tqdm(self.train_loader, desc='Training')
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
            
            # Margin loss
            margin_loss = torch.relu(
                target_scores - false_scores.min(dim=1)[0] + set_size_config['margin']
            ).mean()
            
            # Coverage loss
            coverage_indicators = (target_scores <= tau).float()
            coverage = coverage_indicators.mean()
            coverage_loss = (1 - coverage)
            
            # Size penalty
            pred_sets = scores <= tau
            set_sizes = pred_sets.float().sum(dim=1)
            avg_size = set_sizes.mean()
            
            size_deviation = torch.abs(avg_size - set_size_config['target'])
            size_penalty = size_deviation ** 2
            
            # Combined loss
            loss = (
                self.lambda1 * coverage_loss +
                self.lambda2 * size_penalty +
                0.2 * margin_loss
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scoring_fn.parameters(), max_norm=0.5)
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
    
    def _update_plots(self, history, tau, save_dir, plot_dir):
        """Update training plots"""
        # Collect data for distributions
        true_scores = []
        false_scores = []
        set_sizes = []
        
        self.scoring_fn.eval()
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                scores = torch.zeros_like(probs, device=self.device)
                for i in range(probs.size(1)):
                    class_probs = probs[:, i:i+1]
                    scores[:, i:i+1] = self.scoring_fn(class_probs)
                
                true_class_scores = scores[torch.arange(len(targets)), targets].cpu().numpy()
                true_scores.extend(true_class_scores)
                
                mask = torch.ones_like(scores, dtype=bool)
                mask[torch.arange(len(targets)), targets] = False
                false_class_scores = scores[mask].cpu().numpy()
                false_scores.extend(false_class_scores)
                
                pred_sets = (scores <= tau).sum(dim=1)
                set_sizes.extend(pred_sets.cpu().numpy())
        
        # Update plots
        plot_training_curves(
            epochs=history['epochs'],
            train_losses=history['train_losses'],
            train_coverages=history['train_coverages'],
            train_sizes=history['train_sizes'],
            val_coverages=history['val_coverages'],
            val_sizes=history['val_sizes'],
            tau_values=history['tau_values'],
            save_dir=plot_dir
        )
        
        plot_score_distributions(
            true_scores=true_scores,
            false_scores=false_scores,
            tau=tau,
            save_dir=plot_dir
        )
        
        plot_set_size_distribution(
            set_sizes=set_sizes,
            save_dir=plot_dir
        )
        
        plot_scoring_function_behavior(
            self.scoring_fn,
            self.device,
            plot_dir
        )