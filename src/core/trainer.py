# src/core/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import hashlib

from src.utils.visualization import (
    plot_training_curves, 
    plot_score_distributions,
    plot_set_size_distribution, 
    plot_scoring_function_behavior
)
from .metrics import AverageMeter, compute_tau
from .advanced_metrics import (
    calculate_auroc,
    calculate_auarc_from_scores,
    plot_metrics_over_epochs,
    save_metrics_to_csv,
    calculate_ece
)

class ScoringFunctionTrainer:
    def __init__(self, base_model, scoring_fn, train_loader, cal_loader, 
                 test_loader, device, config):
        """
        Initialize the trainer
        
        Args:
            base_model: Base classification model
            scoring_fn: Scoring function model
            train_loader: Training data loader
            cal_loader: Calibration data loader
            test_loader: Test data loader
            device: Device to run on
            config: Configuration dictionary containing training parameters
        """
        self.base_model = base_model
        self.scoring_fn = scoring_fn
        self.train_loader = train_loader
        self.cal_loader = cal_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config  # Store entire config
        
        # Get loss weights from config
        self.lambda1 = config['training']['loss_weights']['coverage']
        self.lambda2 = config['training']['loss_weights']['size']
        self.margin_weight = config['training']['loss_weights']['margin']
        
        # Get gradient clipping config
        self.grad_clip_config = config['training']['grad_clip']
        
        # Initialize caching attributes
        self.is_cached = False
        self.original_loaders = {
            'train': self.train_loader,
            'cal': self.cal_loader,
            'test': self.test_loader
        }
        self.cached_loaders = {}
        
        # Setup cache directory
        cache_config = config.get('cache', {'enabled': True, 'dir': 'cache'})
        self.use_cache = cache_config.get('enabled', True)
        self.cache_dir = os.path.join(config['base_dir'], cache_config.get('dir', 'cache'))
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _setup_optimizer(self, num_epochs):
        """Setup optimizer and scheduler based on configuration"""
        optimizer_config = self.config['optimizer']
        
        # Initialize optimizer
        if optimizer_config['name'] == 'AdamW':
            optimizer = optim.AdamW(
                self.scoring_fn.parameters(),
                **optimizer_config['params']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
        
        # Initialize scheduler
        scheduler_config = optimizer_config['scheduler']
        if scheduler_config['name'] == 'OneCycleLR':
            scheduler_params = scheduler_config['params'].copy()
            scheduler_params.update({
                'epochs': num_epochs,
                'steps_per_epoch': len(self.train_loader)
            })
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                **scheduler_params
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
            
        return optimizer, scheduler
    
    def _generate_cache_path(self):
        """Generate a unique path for caching based on model and dataset"""
        # Generate a unique identifier for the model
        model_str = str(self.base_model.__class__.__name__)
        # Use dataset name from config
        dataset_name = self.config['dataset']['name']
        # Create cache dir structure
        cache_subdir = os.path.join(self.cache_dir, dataset_name, model_str)
        os.makedirs(cache_subdir, exist_ok=True)
        return cache_subdir
    
    def _compute_model_hash(self):
        """Compute hash of model parameters to ensure cache validity"""
        model_state = self.base_model.state_dict()
        hasher = hashlib.md5()
        
        # Sort keys to ensure consistent order
        for key in sorted(model_state.keys()):
            # Convert tensor to bytes
            param_bytes = model_state[key].cpu().numpy().tobytes()
            hasher.update(param_bytes)
            
        return hasher.hexdigest()
    
    def _save_cache_metadata(self, cache_dir):
        """Save metadata about the cache to ensure validity"""
        metadata = {
            'model_name': self.base_model.__class__.__name__,
            'model_hash': self._compute_model_hash(),
            'dataset': self.config['dataset']['name'],
            'timestamp': str(np.datetime64('now')),
            'dataset_sizes': {
                'train': len(self.train_loader.dataset),
                'cal': len(self.cal_loader.dataset),
                'test': len(self.test_loader.dataset),
            }
        }
        
        metadata_path = os.path.join(cache_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
    
    def _check_cache_validity(self, cache_dir):
        """Check if existing cache is valid for current model and dataset"""
        metadata_path = os.path.join(cache_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            return False
            
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Check if model hash matches
            current_hash = self._compute_model_hash()
            if metadata.get('model_hash') != current_hash:
                logging.info("Model parameters have changed. Cache will be regenerated.")
                return False
                
            # Check if dataset sizes match
            if metadata.get('dataset_sizes', {}).get('train') != len(self.train_loader.dataset) or \
               metadata.get('dataset_sizes', {}).get('cal') != len(self.cal_loader.dataset) or \
               metadata.get('dataset_sizes', {}).get('test') != len(self.test_loader.dataset):
                logging.info("Dataset sizes have changed. Cache will be regenerated.")
                return False
                
            return True
            
        except Exception as e:
            logging.warning(f"Error checking cache validity: {e}")
            return False
    
    def cache_base_model_outputs(self):
        """Pre-compute and cache all base model outputs to disk"""
        if not self.use_cache:
            logging.info("Caching is disabled in config")
            return
            
        # Generate cache path
        cache_dir = self._generate_cache_path()
        
        # Check if valid cache exists
        if self._check_cache_validity(cache_dir):
            logging.info("Found valid cache. Loading cached outputs...")
            self._load_cached_outputs(cache_dir)
            return
            
        logging.info("Generating new cache for base model outputs...")
        self.base_model.eval()
        
        # Process each dataset
        for name, loader in [
            ('train', self.train_loader), 
            ('cal', self.cal_loader), 
            ('test', self.test_loader)
        ]:
            logging.info(f"Processing {name} dataset...")
            all_probs = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in tqdm(loader, desc=f"Caching {name}"):
                    inputs = inputs.to(self.device)
                    
                    # Get softmax probabilities from base model
                    logits = self.base_model(inputs)
                    probs = torch.softmax(logits, dim=1)
                    
                    # Store probabilities and targets
                    all_probs.append(probs.cpu())
                    all_targets.append(targets)
            
            # Concatenate all batches
            probs = torch.cat(all_probs, dim=0)
            targets = torch.cat(all_targets, dim=0)
            
            # Save to disk
            probs_path = os.path.join(cache_dir, f'{name}_probs.pt')
            targets_path = os.path.join(cache_dir, f'{name}_targets.pt')
            
            torch.save(probs, probs_path)
            torch.save(targets, targets_path)
            
            logging.info(f"Cached {len(targets)} samples for {name} dataset")
        
        # Save metadata
        self._save_cache_metadata(cache_dir)
        
        # Load the cached outputs
        self._load_cached_outputs(cache_dir)
        
        # Free up GPU memory by moving base model to CPU if needed
        if torch.cuda.is_available():
            self.base_model = self.base_model.cpu()
            torch.cuda.empty_cache()
            logging.info("Moved base model to CPU to free GPU memory")
    
    def _load_cached_outputs(self, cache_dir):
        """Load cached outputs from disk and create new data loaders"""
        from torch.utils.data import TensorDataset, DataLoader
        
        self.cached_datasets = {}
        self.cached_loaders = {}
        
        # Load each dataset
        for name in ['train', 'cal', 'test']:
            probs_path = os.path.join(cache_dir, f'{name}_probs.pt')
            targets_path = os.path.join(cache_dir, f'{name}_targets.pt')
            
            if not os.path.exists(probs_path) or not os.path.exists(targets_path):
                raise FileNotFoundError(f"Cache files for {name} dataset not found")
                
            probs = torch.load(probs_path)
            targets = torch.load(targets_path)
            
            # Create dataset
            dataset = TensorDataset(probs, targets)
            self.cached_datasets[name] = dataset
            
            # Create dataloader
            shuffle = (name == 'train')  # Only shuffle training data
            self.cached_loaders[name] = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=shuffle,
                num_workers=2,  # Reduced workers since data is smaller
                pin_memory=True
            )
        
        # Update the loaders
        self.train_loader = self.cached_loaders['train']
        self.cal_loader = self.cached_loaders['cal']
        self.test_loader = self.cached_loaders['test']
        
        self.is_cached = True
        logging.info("Successfully loaded cached outputs")
    
    def _init_history(self):
        """Initialize training history dictionary"""
        return {
            'epochs': [],
            'train_losses': [],
            'train_coverages': [],
            'train_sizes': [],
            'val_coverages': [],
            'val_sizes': [],
            'tau_values': [],
            'auroc_values': [],
            'auarc_values': [],
            'ece_values': []
        }
    
    def _log_metrics(self, epoch, num_epochs, current_lr, train_loss, train_coverage, 
                     train_size, val_coverage, val_size, tau, auroc=None, auarc=None, ece=None):
        """Log training metrics"""
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Learning rate: {current_lr:.6f}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Train Coverage: {train_coverage:.4f}")
        logging.info(f"Train Set Size: {train_size:.4f}")
        logging.info(f"Val Coverage: {val_coverage:.4f}")
        logging.info(f"Val Set Size: {val_size:.4f}")
        logging.info(f"Tau: {tau:.4f}")
        
        if auroc is not None:
            logging.info(f"AUROC: {auroc:.4f}")
        if auarc is not None:
            logging.info(f"AUARC: {auarc:.4f}")
        if ece is not None:
            logging.info(f"ECE: {ece:.4f}")
    
    def _update_history(self, history, epoch, train_loss, train_coverage, train_size,
                       val_coverage, val_size, tau, auroc=None, auarc=None, ece=None):
        """Update training history"""
        history['epochs'].append(epoch)
        history['train_losses'].append(train_loss)
        history['train_coverages'].append(train_coverage)
        history['train_sizes'].append(train_size)
        history['val_coverages'].append(val_coverage)
        history['val_sizes'].append(val_size)
        history['tau_values'].append(tau)
        
        if auroc is not None:
            history['auroc_values'].append(auroc)
        if auarc is not None:
            history['auarc_values'].append(auarc)
        if ece is not None:
            history['ece_values'].append(ece)
    
    def _save_model(self, train_loss, best_loss, set_size, set_size_config, save_dir):
        """Save model if it's the best so far"""
        if train_loss < best_loss and set_size < set_size_config['max']:
            best_loss = train_loss
            
            # Create dataset-specific filename
            dataset_name = self.config['dataset']['name']
            model_filename = f'scoring_function_{dataset_name}_best.pth'
            
            torch.save(
                self.scoring_fn.state_dict(),
                os.path.join(save_dir, model_filename)
            )
            logging.info(f"Saved new best model for {dataset_name}")
        return best_loss
    
    def _calculate_target_coverage_metrics(self, history, target_coverage=0.9, tolerance=0.02):
        """
        Calculate average set size and coverage for epochs where coverage is close to target.
        
        Args:
            history: Training history dictionary
            target_coverage: Target coverage (default: 0.9)
            tolerance: Tolerance around target (default: 0.02, meaning 88-92% for target=0.9)
            
        Returns:
            Dictionary with average metrics and number of qualifying epochs
        """
        qualifying_epochs = []
        qualifying_coverages = []
        qualifying_sizes = []
        qualifying_auroc = []
        qualifying_auarc = []
        qualifying_ece = []
        qualifying_tau = []
        qualifying_efficiency = []
        
        # Find epochs where coverage is within tolerance of target
        for i, coverage in enumerate(history['val_coverages']):
            if abs(coverage - target_coverage) <= tolerance:
                epoch_idx = i
                qualifying_epochs.append(history['epochs'][epoch_idx])
                qualifying_coverages.append(coverage)
                qualifying_sizes.append(history['val_sizes'][epoch_idx])
                
                # Add additional metrics if available
                if 'auroc_values' in history and len(history['auroc_values']) > epoch_idx:
                    qualifying_auroc.append(history['auroc_values'][epoch_idx])
                
                if 'auarc_values' in history and len(history['auarc_values']) > epoch_idx:
                    qualifying_auarc.append(history['auarc_values'][epoch_idx])
                
                if 'ece_values' in history and len(history['ece_values']) > epoch_idx:
                    qualifying_ece.append(history['ece_values'][epoch_idx])
                
                if 'tau_values' in history and len(history['tau_values']) > epoch_idx:
                    qualifying_tau.append(history['tau_values'][epoch_idx])
                
                # Calculate efficiency (coverage/size) for this epoch
                efficiency = coverage / history['val_sizes'][epoch_idx]
                qualifying_efficiency.append(efficiency)
        
        # Calculate averages if any qualifying epochs exist
        if qualifying_epochs:
            avg_coverage = sum(qualifying_coverages) / len(qualifying_coverages)
            avg_size = sum(qualifying_sizes) / len(qualifying_sizes)
            avg_efficiency = sum(qualifying_efficiency) / len(qualifying_efficiency)
            
            result = {
                'avg_coverage': avg_coverage,
                'avg_size': avg_size,
                'avg_efficiency': avg_efficiency,
                'num_epochs': len(qualifying_epochs),
                'epochs': qualifying_epochs
            }
            
            # Add averages for the additional metrics if available
            if qualifying_auroc:
                result['avg_auroc'] = sum(qualifying_auroc) / len(qualifying_auroc)
            
            if qualifying_auarc:
                result['avg_auarc'] = sum(qualifying_auarc) / len(qualifying_auarc)
            
            if qualifying_ece:
                result['avg_ece'] = sum(qualifying_ece) / len(qualifying_ece)
            
            if qualifying_tau:
                result['avg_tau'] = sum(qualifying_tau) / len(qualifying_tau)
            
            return result
        else:
            return {
                'avg_coverage': None,
                'avg_size': None,
                'avg_efficiency': None,
                'num_epochs': 0,
                'epochs': []
            }
    
    def train(self, num_epochs, target_coverage, tau_config, set_size_config, save_dir, plot_dir):
        """Main training loop"""
        # Cache base model outputs at the beginning
        if not self.is_cached and self.use_cache:
            self.cache_base_model_outputs()
        
        optimizer, scheduler = self._setup_optimizer(num_epochs)
        history = self._init_history()
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            current_lr = optimizer.param_groups[0]['lr']
            
            # Compute tau on calibration set
            tau = compute_tau(
                cal_loader=self.cal_loader,
                scoring_fn=self.scoring_fn,
                base_model=None if self.is_cached else self.base_model,  # Pass base_model when not cached
                device=self.device,
                coverage_target=target_coverage,
                tau_config=tau_config
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
            
            # Calculate AUROC, AUARC, and ECE
            auroc, auarc, ece = self._calculate_advanced_metrics(tau)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            self._log_metrics(
                epoch, num_epochs, current_lr, train_loss, train_coverage,
                train_size, val_coverage, val_size, tau, auroc, auarc, ece
            )
            
            # Update history
            self._update_history(
                history, epoch, train_loss, train_coverage, train_size,
                val_coverage, val_size, tau, auroc, auarc, ece
            )
            
            # Update plots
            self._update_plots(history, tau, save_dir, plot_dir)
            
            # Save best model
            best_loss = self._save_model(
                train_loss, best_loss, train_size, set_size_config, save_dir
            )
        
        # Save final metrics to CSV
        metrics_file = save_metrics_to_csv(
            epochs=history['epochs'],
            auroc_values=history['auroc_values'],
            auarc_values=history['auarc_values'],
            ece_values=history['ece_values'],
            coverage_values=history['val_coverages'],
            size_values=history['val_sizes'],
            model_name=self.config['dataset']['name'],
            save_dir=plot_dir
        )
        logging.info(f"Saved metrics to {metrics_file}")
        
        # Create a summary analysis
        from .advanced_metrics import analyze_epoch_metrics
        analysis = analyze_epoch_metrics(
            epochs=history['epochs'],
            auroc_values=history['auroc_values'],
            auarc_values=history['auarc_values'],
            ece_values=history['ece_values'],
            coverage_values=history['val_coverages'],
            size_values=history['val_sizes']
        )
        
        # Log analysis results
        logging.info("\nMetrics Analysis:")
        logging.info(f"Best AUROC: {analysis['best_auroc']['value']:.4f} at epoch {analysis['best_auroc']['epoch']}")
        logging.info(f"Best AUARC: {analysis['best_auarc']['value']:.4f} at epoch {analysis['best_auarc']['epoch']}")
        if 'best_ece' in analysis:
            logging.info(f"Best ECE: {analysis['best_ece']['value']:.4f} at epoch {analysis['best_ece']['epoch']}")
        
        if 'best_trade_off' in analysis:
            logging.info(f"Best trade-off at epoch {analysis['best_trade_off']['epoch']}:")
            logging.info(f"  Coverage: {analysis['best_trade_off']['coverage']:.4f}")
            logging.info(f"  Set Size: {analysis['best_trade_off']['size']:.4f}")
            logging.info(f"  Efficiency: {analysis['best_trade_off']['trade_off']:.4f}")
        
        # Calculate and log metrics for epochs with coverage close to target
        target_metrics = self._calculate_target_coverage_metrics(
            history, 
            target_coverage=target_coverage,
            tolerance=0.02  # ±2% tolerance (88-92% for target=90%)
        )
        
        if target_metrics['num_epochs'] > 0:
            logging.info("\nMetrics for epochs with coverage within ±2% of target:")
            logging.info(f"  Number of qualifying epochs: {target_metrics['num_epochs']}")
            logging.info(f"  Average Coverage: {target_metrics['avg_coverage']:.4f}")
            logging.info(f"  Average Set Size: {target_metrics['avg_size']:.4f}")
            logging.info(f"  Average Efficiency: {target_metrics['avg_efficiency']:.4f}")
            
            # Log additional metrics if available
            if 'avg_auroc' in target_metrics:
                logging.info(f"  Average AUROC: {target_metrics['avg_auroc']:.4f}")
            if 'avg_auarc' in target_metrics:
                logging.info(f"  Average AUARC: {target_metrics['avg_auarc']:.4f}")
            if 'avg_ece' in target_metrics:
                logging.info(f"  Average ECE: {target_metrics['avg_ece']:.4f}")
            if 'avg_tau' in target_metrics:
                logging.info(f"  Average Tau: {target_metrics['avg_tau']:.4f}")
                
            logging.info(f"  Qualifying epochs: {target_metrics['epochs']}")
        else:
            logging.info("\nNo epochs had coverage within ±2% of target coverage.")
            
            # Find closest epoch to target coverage
            closest_idx = min(range(len(history['val_coverages'])), 
                             key=lambda i: abs(history['val_coverages'][i] - target_coverage))
            
            logging.info(f"Closest epoch to target coverage: {history['epochs'][closest_idx]}")
            logging.info(f"  Coverage: {history['val_coverages'][closest_idx]:.4f}")
            logging.info(f"  Set Size: {history['val_sizes'][closest_idx]:.4f}")
        
        logging.info("Training completed!")
    
    def _calculate_advanced_metrics(self, tau):
        """Calculate AUROC, AUARC, and ECE metrics on the test set"""
        self.scoring_fn.eval()
        if not self.is_cached and self.base_model is not None:
            self.base_model.eval()
        
        all_true_labels = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get scores for all classes using the appropriate method for cached/non-cached
                scores, _, _ = self._compute_scores(inputs, targets)
                
                all_true_labels.extend(targets.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
        
        # Convert to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_scores = np.concatenate(all_scores, axis=0)
        
        # For AUROC, we need to convert scores (lower is better) to probabilities (higher is better)
        # In conformal prediction, lower scores are better (included in prediction set if score <= tau)
        all_probs = 1.0 - all_scores
        
        # Normalize probabilities to ensure they sum to 1.0 across classes for each sample
        # This is required for multiclass ROC AUC calculation
        row_sums = np.sum(all_probs, axis=1, keepdims=True)
        all_probs_normalized = all_probs / row_sums
        
        # Calculate AUROC using one-vs-rest approach for multi-class
        auroc = calculate_auroc(all_true_labels, all_probs_normalized)
        
        # Calculate AUARC using a range of tau values
        tau_values = np.linspace(0, 1, 20)  # 20 threshold values from 0 to 1
        auarc = calculate_auarc_from_scores(all_true_labels, all_scores, tau_values)
        
        # Calculate ECE
        ece = calculate_ece(all_true_labels, all_probs_normalized)
        
        return auroc, auarc, ece
    
    def _compute_scores(self, inputs, targets=None):
        """
        Helper method to compute scores for inputs
        Modified to handle both raw inputs and cached probabilities
        """
        # Inputs are already probabilities if using cached data
        if self.is_cached:
            probs = inputs
        else:
            with torch.no_grad():
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
        
        # Vectorized approach: reshape to process all probabilities at once
        batch_size, num_classes = probs.shape
        flat_probs = probs.reshape(-1, 1)  # Reshape to (batch_size * num_classes, 1)
        
        # Process all probabilities in a single forward pass
        # When in eval mode, this won't compute stability_loss
        flat_scores = self.scoring_fn(flat_probs)
        
        # Reshape back to original dimensions
        scores = flat_scores.reshape(batch_size, num_classes)
        
        if targets is not None:
            target_scores = scores[torch.arange(len(targets)), targets]
            return scores, target_scores, probs
        return scores, None, probs

    def train_epoch(self, optimizer, tau, tau_config, set_size_config):
        """Train for one epoch"""
        self.scoring_fn.train()
        loss_meter = AverageMeter()
        coverage_meter = AverageMeter()
        size_meter = AverageMeter()
        
        # Ensure tau is within reasonable bounds
        tau = max(tau_config['min'], min(tau_config['max'], tau))
        
        # Get target coverage from config
        target_coverage = self.config['target_coverage']
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            
            scores, target_scores, _ = self._compute_scores(inputs, targets)
            
            # Get false class scores
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
            
            # Add separation loss to encourage true scores near 0 and false scores near 1
            # This helps create the desired separation in the score distributions
            if hasattr(self.scoring_fn, 'separation_factor'):
                # Push true scores toward 0
                true_score_loss = torch.mean(target_scores ** 2)
                
                # Push false scores toward 1
                false_score_loss = torch.mean((1.0 - false_scores) ** 2)
                
                # Combined separation loss
                separation_loss = true_score_loss + false_score_loss
                self.scoring_fn.separation_loss = self.scoring_fn.separation_factor * separation_loss
            else:
                separation_loss = 0.0
            
            # Dynamically adjust coverage weight if below target
            coverage_boost = 1.0
            size_boost = 1.0
            
            # If coverage is good (close to target), boost size weight
            if abs(coverage.item() - target_coverage) < 0.01:  # Within 1% of target
                size_boost = 1.5  # Boost size weight moderately
            # If coverage is too low, prioritize coverage
            elif coverage.item() < target_coverage - 0.02:  # If more than 2% below target
                coverage_boost = 2.0  # Boost coverage weight
            
            # Combined loss with dynamic weights
            loss = (
                self.lambda1 * coverage_boost * coverage_loss +
                self.lambda2 * size_boost * size_penalty +
                self.margin_weight * margin_loss
            )
            
            # Add stability loss if available
            if hasattr(self.scoring_fn, 'stability_loss'):
                loss = loss + self.scoring_fn.stability_loss
            
            # Add separation loss if available
            if hasattr(self.scoring_fn, 'separation_loss'):
                loss = loss + self.scoring_fn.separation_loss
            
            # Add L2 regularization if available
            if hasattr(self.scoring_fn, 'l2_reg'):
                loss = loss + self.scoring_fn.l2_reg
            
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping if enabled
            if self.grad_clip_config['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    self.scoring_fn.parameters(), 
                    max_norm=self.grad_clip_config['max_norm']
                )
            
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
        
        # Ensure tau is within reasonable bounds
        tau_config = self.config['tau']
        tau = max(tau_config['min'], min(tau_config['max'], tau))
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                scores, target_scores, _ = self._compute_scores(inputs, targets)
                
                # Compute coverage (true class scores <= tau)
                coverage = (target_scores <= tau).float().mean()
                
                # Compute prediction sets and sizes
                pred_sets = scores <= tau
                set_sizes = pred_sets.float().sum(dim=1)
                avg_size = set_sizes.mean()
                
                # Update metrics
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg
    
    def _update_plots(self, history, tau, save_dir, plot_dir):
        """Update training plots"""
        # Collect data for distributions
        true_scores = []
        false_scores = []
        set_sizes = []
        
        # Ensure tau is within reasonable bounds
        tau_config = self.config['tau']
        tau = max(tau_config['min'], min(tau_config['max'], tau))
        
        self.scoring_fn.eval()
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                scores, target_scores, _ = self._compute_scores(inputs, targets)
                
                true_scores.extend(target_scores.cpu().numpy())
                
                mask = torch.ones_like(scores, dtype=bool)
                mask[torch.arange(len(targets)), targets] = False
                false_class_scores = scores[mask].cpu().numpy()
                false_scores.extend(false_class_scores)
                
                pred_sets = (scores <= tau).sum(dim=1)
                set_sizes.extend(pred_sets.cpu().numpy())
        
        # Update plots and close figures after saving
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
        plt.close()
        
        plot_score_distributions(
            true_scores=true_scores,
            false_scores=false_scores,
            tau=tau,
            save_dir=plot_dir
        )
        plt.close()
        
        plot_set_size_distribution(
            set_sizes=set_sizes,
            save_dir=plot_dir
        )
        plt.close()
        
        plot_scoring_function_behavior(
            self.scoring_fn,
            self.device,
            plot_dir
        )
        plt.close()
        
        # Plot AUROC, AUARC, and ECE metrics if available
        if 'auroc_values' in history and len(history['auroc_values']) > 0:
            self._plot_advanced_metrics(history, plot_dir)
            plt.close()
    
    def _plot_advanced_metrics(self, history, plot_dir):
        """Plot AUROC, AUARC, and ECE metrics over epochs"""
        if len(history['epochs']) < 2:
            return  # Need at least 2 epochs to plot
            
        plot_metrics_over_epochs(
            epochs=history['epochs'],
            auroc_values=history['auroc_values'],
            auarc_values=history['auarc_values'],
            ece_values=history['ece_values'],
            model_names=[self.config['dataset']['name']],
            title=f"AUROC, AUARC, and ECE Metrics for {self.config['dataset']['name']}",
            save_path=os.path.join(plot_dir, 'auroc_auarc_metrics.png')
        )