# src/core/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.utils.visualization import (
    plot_training_curves, 
    plot_score_distributions,
    plot_set_size_distribution, 
    plot_scoring_function_behavior
)
from .metrics import AverageMeter, compute_tau
from .advanced_metrics import (
    calculate_auroc,
    plot_metrics_over_epochs,
    save_metrics_to_csv,
    calculate_ece
)
from .cache_generator import HighQualityCacheGenerator

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
        
        # Initialize model saving tracking
        self.best_model_filename = None
        
        # Initialize caching attributes
        self.is_cached = False
        self.original_loaders = {
            'train': self.train_loader,
            'cal': self.cal_loader,
            'test': self.test_loader
        }
        self.cached_loaders = {}
        
        # Setup cache directory - no fallback, must be explicitly defined
        if 'cache' not in config:
            raise ValueError("config['cache'] must be explicitly defined")
        if 'enabled' not in config['cache']:
            raise ValueError("config['cache']['enabled'] must be explicitly defined")
        if 'dir' not in config['cache']:
            raise ValueError("config['cache']['dir'] must be explicitly defined")
        
        self.use_cache = config['cache']['enabled']
        self.cache_dir = os.path.join(config['base_dir'], config['cache']['dir'])
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_required_config(self, section: str, key: str):
        """Get config value with no fallback - fail if not present"""
        if section not in self.config:
            raise ValueError(f"config['{section}'] must be explicitly defined")
        if key not in self.config[section]:
            raise ValueError(f"config['{section}']['{key}'] must be explicitly defined")
        return self.config[section][key]
        
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
        elif scheduler_config['name'] == 'CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                **scheduler_config['params']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
            
        return optimizer, scheduler
    
    
    def cache_base_model_outputs(self, chunk_size=None, enable_memory_monitoring=True):
        """Pre-compute and cache all base model outputs using the HighQualityCacheGenerator"""
        if not self.use_cache:
            logging.info("Caching is disabled in config")
            return
            
        # Create cache generator
        cache_generator = HighQualityCacheGenerator(
            base_model=self.base_model,
            config=self.config,
            device=self.device
        )
        
        # Generate optimized cache using the dedicated module
        dataloaders = {
            'train': self.train_loader,
            'cal': self.cal_loader,
            'test': self.test_loader
        }
        
        cache_path = cache_generator.generate_cache_optimized(
            dataloaders=dataloaders,
            chunk_size=chunk_size,
            enable_memory_monitoring=enable_memory_monitoring
        )
        
        # Load the cached data
        cached_loaders = cache_generator.load_cache(cache_path)
        
        # Update our loaders to use cached data
        self.train_loader = cached_loaders['train']
        self.cal_loader = cached_loaders['cal']
        self.test_loader = cached_loaders['test']
        
        # Mark as cached
        self.is_cached = True
        
        # Free up GPU memory by moving base model to CPU if needed
        if torch.cuda.is_available():
            self.base_model = self.base_model.cpu()
            torch.cuda.empty_cache()
            logging.info("Moved base model to CPU to free GPU memory")
    
    
    def _init_history(self):
        """Initialize training history dictionary"""
        return {
            'epochs': [],
            'train_losses': [],
            'train_coverages': [],
            'train_sizes': [],
            'val_coverages': [],
            'val_sizes': [],
            'val_non_empty_sizes': [],
            'tau_values': [],
            'auroc_values': [],
            'ece_values': []
        }
    
    def _log_metrics(self, epoch, num_epochs, current_lr, train_loss, train_coverage, 
                     train_size, val_coverage, val_size, tau, auroc=None, ece=None, val_non_empty_size=None):
        """Log training metrics"""
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Learning rate: {current_lr:.6f}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Train Coverage: {train_coverage:.4f}")
        logging.info(f"Train Set Size: {train_size:.4f}")
        logging.info(f"Val Coverage: {val_coverage:.4f}")
        logging.info(f"Val Set Size: {val_size:.4f}")
        if val_non_empty_size is not None:
            logging.info(f"Val Non-Empty Set Size: {val_non_empty_size:.4f}")
        logging.info(f"Tau: {tau:.4f}")
        
        if auroc is not None:
            logging.info(f"AUROC: {auroc:.4f}")
        if ece is not None:
            logging.info(f"ECE: {ece:.4f}")
    
    def _update_history(self, history, epoch, train_loss, train_coverage, train_size,
                       val_coverage, val_size, tau, auroc=None, ece=None, val_non_empty_size=None):
        """Update training history"""
        history['epochs'].append(epoch)
        history['train_losses'].append(train_loss)
        history['train_coverages'].append(train_coverage)
        history['train_sizes'].append(train_size)
        history['val_coverages'].append(val_coverage)
        history['val_sizes'].append(val_size)
        if val_non_empty_size is not None:
            history['val_non_empty_sizes'].append(val_non_empty_size)
        history['tau_values'].append(tau)
        
        if auroc is not None:
            history['auroc_values'].append(auroc)
        if ece is not None:
            history['ece_values'].append(ece)
    
    def _save_metrics_to_csv_incremental(self, history, plot_dir):
        """Save metrics to CSV incrementally after each epoch"""
        import pandas as pd
        
        dataset_name = self.config['dataset']['name']
        csv_path = os.path.join(plot_dir, f'{dataset_name}_metrics.csv')
        
        # Create DataFrame with all metrics
        data = {
            'epoch': history['epochs'],
            'auroc': history['auroc_values'],
            'ece': history['ece_values'],
            'coverage': history['val_coverages'],
            'set_size': history['val_sizes'],
            'non_empty_set_size': history['val_non_empty_sizes'] if 'val_non_empty_sizes' in history else history['val_sizes'],
            'efficiency': [cov/size if size > 0 else 0 for cov, size in zip(history['val_coverages'], history['val_sizes'])]
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV (overwrite each time to avoid duplicates)
        df.to_csv(csv_path, index=False)
        logging.debug(f"Updated metrics CSV: {csv_path}")
    
    def _save_model(self, val_coverage, val_size, best_set_size, save_dir, epoch, test_loader, tau):
        """
        Save model based on validation metrics.
        Only saves when coverage is between 88-92% and set size is smaller than previous best.
        Replaces any previous best model for the dataset.
        
        Args:
            val_coverage: Validation coverage (e.g., 0.90)
            val_size: Validation set size (e.g., 4.96)
            best_set_size: Previous best set size
            save_dir: Directory to save model
            epoch: Current epoch number
            test_loader: Test data loader for visualization generation
            
        Returns:
            Tuple of (updated best_set_size, saved_filename or None)
        """
        # Check if coverage is within acceptable range (88-92%)
        if 0.88 <= val_coverage <= 0.92:
            # Check if this is a better (smaller) set size
            if val_size < best_set_size:
                dataset_name = self.config['dataset']['name']
                
                # First, remove any existing best model for this dataset
                existing_pattern = os.path.join(save_dir, f'scoring_function_{dataset_name}_*_*_*epoch.pth')
                for old_model in glob.glob(existing_pattern):
                    os.remove(old_model)
                    logging.info(f"Removed previous best model: {os.path.basename(old_model)}")
                
                # Create new filename with metrics and epoch
                model_filename = f'scoring_function_{dataset_name}_{val_coverage:.2f}_{val_size:.2f}_{epoch}epoch.pth'
                model_path = os.path.join(save_dir, model_filename)
                
                torch.save(
                    self.scoring_fn.state_dict(),
                    model_path
                )
                
                # Also save as 'best' for backward compatibility
                best_filename = f'scoring_function_{dataset_name}_best.pth'
                best_path = os.path.join(save_dir, best_filename)
                torch.save(
                    self.scoring_fn.state_dict(),
                    best_path
                )
                
                logging.info(f"Saved new best model for {dataset_name}:")
                logging.info(f"  Coverage: {val_coverage:.4f} (within 88-92% range)")
                logging.info(f"  Set Size: {val_size:.4f} (improved from {best_set_size:.4f})")
                logging.info(f"  Epoch: {epoch}")
                logging.info(f"  Filename: {model_filename}")
                
                # Store the filename for the final summary
                self.best_model_filename = model_filename
                
                # Generate high-quality visualizations for best model
                best_model_plot_dir = os.path.join(self.plot_dir, 'best_model_perf')
                logging.info(f"Generating high-quality visualizations in {best_model_plot_dir}")
                
                from src.utils.visualization import generate_best_model_visualizations
                generate_best_model_visualizations(
                    self.scoring_fn, 
                    test_loader, 
                    self.config,
                    best_model_plot_dir,
                    tau
                )
                logging.info("High-quality visualizations generated successfully")
                
                return val_size, model_filename
        else:
            logging.debug(f"Coverage {val_coverage:.4f} outside 88-92% range, not saving")
            
        return best_set_size, None
    
    def _calculate_target_coverage_metrics(self, history, target_coverage=0.9, tolerance=None):
        if tolerance is None:
            tolerance = self._get_required_config('training_dynamics', 'coverage_tolerance')
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
    
    def _check_class_agnostic_behavior(self):
        """Debug function to verify class-agnostic behavior"""
        self.scoring_fn.eval()
        
        # Create synthetic probability vectors to test symmetry
        test_probs = torch.tensor([
            [0.7, 0.2, 0.1],  # High confidence for class 0
            [0.2, 0.7, 0.1],  # Same pattern but for class 1
            [0.1, 0.2, 0.7],  # Same pattern but for class 2
        ], device=self.device)
        
        # Pad with zeros if we have more than 3 classes
        if self.scoring_fn.num_classes > 3:
            padding = torch.zeros(3, self.scoring_fn.num_classes - 3, device=self.device)
            test_probs = torch.cat([test_probs, padding], dim=1)
        
        with torch.no_grad():
            scores = self.scoring_fn(test_probs)
        
        # Check if scores follow the same pattern
        logging.info("=== Class-agnostic behavior check ===")
        logging.info(f"Prob vector 1 (high class 0): {test_probs[0][:3].cpu().numpy()}")
        logging.info(f"Scores: {scores[0][:3].cpu().numpy()}")
        logging.info(f"Prob vector 2 (high class 1): {test_probs[1][:3].cpu().numpy()}")
        logging.info(f"Scores: {scores[1][:3].cpu().numpy()}")
        logging.info(f"Prob vector 3 (high class 2): {test_probs[2][:3].cpu().numpy()}")
        logging.info(f"Scores: {scores[2][:3].cpu().numpy()}")
        
        # Check if the scores for high probability class are similar
        high_prob_scores = [scores[0, 0], scores[1, 1], scores[2, 2]]
        low_prob_scores = [scores[0, 2], scores[1, 0], scores[2, 1]]
        
        logging.info(f"High prob (0.7) scores: {[s.item() for s in high_prob_scores]}")
        logging.info(f"Low prob (0.1) scores: {[s.item() for s in low_prob_scores]}")
        logging.info("=====================================")
        
        self.scoring_fn.train()
    
    def train(self, num_epochs, target_coverage, tau_config, set_size_config, save_dir, plot_dir):
        """Main training loop"""
        # Store plot_dir as instance variable for use in other methods
        self.plot_dir = plot_dir
        
        # Cache base model outputs at the beginning
        if not self.is_cached and self.use_cache:
            self.cache_base_model_outputs()
        
        optimizer, scheduler = self._setup_optimizer(num_epochs)
        history = self._init_history()
        best_set_size = float('inf')  # Initialize best set size to infinity
        
        # Track best performance for stability
        best_coverage_close = float('inf')  # Best set size when coverage is close to target
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Store current epoch for adaptive regularization
            self.current_epoch = epoch + 1
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
            val_coverage, val_size, val_non_empty_size = self.evaluate(self.test_loader, tau)
            
            # Calculate AUROC and ECE
            auroc, ece = self._calculate_advanced_metrics(tau)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            self._log_metrics(
                epoch, num_epochs, current_lr, train_loss, train_coverage,
                train_size, val_coverage, val_size, tau, auroc, ece, val_non_empty_size
            )
            
            # Update history
            self._update_history(
                history, epoch, train_loss, train_coverage, train_size,
                val_coverage, val_size, tau, auroc, ece, val_non_empty_size
            )
            
            # Save metrics to CSV incrementally
            self._save_metrics_to_csv_incremental(history, plot_dir)
            
            # Update plots
            self._update_plots(history, tau, save_dir, plot_dir)
            
            # Check class-agnostic behavior every 5 epochs
            if epoch % 5 == 0:
                self._check_class_agnostic_behavior()
            
            # Save best model based on validation metrics
            best_set_size, saved_filename = self._save_model(
                val_coverage, val_size, best_set_size, save_dir, epoch + 1, self.test_loader, tau
            )
        
        # Save final metrics to CSV
        metrics_file = save_metrics_to_csv(
            epochs=history['epochs'],
            auroc_values=history['auroc_values'],
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
            ece_values=history['ece_values'],
            coverage_values=history['val_coverages'],
            size_values=history['val_sizes']
        )
        
        # Log analysis results
        logging.info("\nMetrics Analysis:")
        logging.info(f"Best AUROC: {analysis['best_auroc']['value']:.4f} at epoch {analysis['best_auroc']['epoch']}")
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
            tolerance=self._get_required_config('training_dynamics', 'coverage_tolerance')
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
        
        # Report best model saved
        if best_set_size < float('inf') and hasattr(self, 'best_model_filename'):
            logging.info(f"\nBest model saved:")
            logging.info(f"  Filename: {self.best_model_filename}")
            logging.info(f"  Final set size: {best_set_size:.4f}")
        else:
            logging.info("\nNo model saved - coverage never reached 88-92% range")
    
    def _calculate_advanced_metrics(self, tau):
        """Calculate AUROC and ECE metrics on the test set"""
        self.scoring_fn.eval()
        if not self.is_cached and self.base_model is not None:
            self.base_model.eval()
        
        all_true_labels = []
        all_scores = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get scores from our scoring function
                scores, target_scores, _ = self._compute_scores(inputs, targets)
                
                all_true_labels.extend(targets.cpu().numpy())
                all_scores.append(scores.cpu().numpy())
        
        # Convert to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_scores = np.concatenate(all_scores, axis=0)
        
        # 1. AUROC for scoring function:
        # We want to measure if the scoring function assigns lower scores to true classes
        # Create binary labels: 1 for true class, 0 for false classes
        n_samples, n_classes = all_scores.shape
        binary_labels = np.zeros_like(all_scores)
        for i in range(n_samples):
            binary_labels[i, all_true_labels[i]] = 1
        
        # Flatten for AUROC calculation
        # Our scoring function should produce: high prob classes → low scores, low prob classes → high scores
        # AUROC expects: true positives to have higher scores than false positives
        # So we negate our scores since our good predictions have low scores
        scores_flat = -all_scores.flatten()  # Negate: low conformal scores → high AUROC scores
        labels_flat = binary_labels.flatten()
        
        # Calculate AUROC for the scoring function
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(labels_flat, scores_flat)
        
        # 2. ECE for MLP conformal prediction:
        # Measure calibration across different coverage levels
        # This tests if our MLP gives proper coverage at various confidence levels
        num_points = self._get_required_config('training_dynamics', 'ece_test_points')
        target_coverages = np.linspace(0.5, 0.99, num_points)  # Test from 50% to 99% coverage
        actual_coverages = []
        
        # First, get calibration scores to find taus
        cal_true_scores = []
        with torch.no_grad():
            for inputs, targets in self.cal_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                scores = self.scoring_fn(inputs)
                batch_size = targets.shape[0]
                true_scores = scores[torch.arange(batch_size), targets]
                cal_true_scores.extend(true_scores.cpu().numpy())
        cal_true_scores = np.array(cal_true_scores)
        
        # For each target coverage, find tau and test
        for target_cov in target_coverages:
            # Find tau on calibration set
            tau_for_coverage = np.percentile(cal_true_scores, target_cov * 100)
            
            # Measure actual coverage on test set
            coverage = 0
            for i, true_label in enumerate(all_true_labels):
                if all_scores[i, true_label] <= tau_for_coverage:
                    coverage += 1
            actual_coverages.append(coverage / len(all_true_labels))
        
        # ECE is the mean absolute difference between target and actual coverage
        # This measures how well calibrated our MLP scoring function is
        ece = np.mean(np.abs(np.array(target_coverages) - np.array(actual_coverages)))
        
        return auroc, ece
    
    def _find_tau_for_coverage(self, target_coverage):
        """Find tau threshold that gives target coverage on calibration set"""
        self.scoring_fn.eval()
        cal_scores = []
        cal_labels = []
        
        with torch.no_grad():
            for inputs, targets in self.cal_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get scores from scoring function
                scores, target_scores, _ = self._compute_scores(inputs, targets)
                
                cal_scores.append(scores.cpu().numpy())
                cal_labels.extend(targets.cpu().numpy())
        
        # Concatenate all scores
        cal_scores = np.concatenate(cal_scores, axis=0)
        cal_labels = np.array(cal_labels)
        
        # Extract true class scores
        true_scores = []
        for i, label in enumerate(cal_labels):
            true_scores.append(cal_scores[i, label])
        true_scores = np.array(true_scores)
        
        # Find tau at the target coverage quantile
        # Since we use scores <= tau, we need target_coverage quantile
        tau = np.quantile(true_scores, target_coverage)
        
        return tau
    
    def _compute_scores(self, inputs, targets=None):
        """
        Compute non-conformity scores using the learnable scoring function.
        
        The MLP takes probability vectors and directly outputs non-conformity scores
        for each class. This is a purely data-driven approach without using any
        static scoring functions like (1-p) or APS.
        
        Args:
            inputs: Input data (images or cached probabilities)
            targets: Ground truth labels (optional)
            
        Returns:
            scores: Non-conformity scores for all classes (batch_size, num_classes)
            target_scores: Scores for the true classes if targets provided
            probs: Probability distributions from base model
        """
        # Inputs are already probabilities if using cached data
        if self.is_cached:
            probs = inputs
        else:
            with torch.no_grad():
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
        
        batch_size, num_classes = probs.shape
        
        # Pass probability vector to scoring function
        # The MLP now directly outputs scores for each class
        # No static scoring function like (1-p) is used!
        scores = self.scoring_fn(probs)  # Shape: (batch_size, num_classes)
        
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
            
            # Sort scores for regularization
            sorted_scores, sorted_indices = torch.sort(scores, dim=1)
            
            # Find position of true class in sorted scores
            true_positions = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
            for i in range(batch_size):
                true_positions[i] = (sorted_indices[i] == targets[i]).nonzero(as_tuple=True)[0]
            
            # NEW: Phase-based training strategy
            # Phase 1: Discrimination loss only (epochs 1-10)
            # Phase 2: Add coverage loss (epochs 11-20)  
            # Phase 3: Add size loss (epochs 21-30)
            
            current_epoch = getattr(self, 'current_epoch', 1)
            
            # PRIMARY: Margin-based discrimination loss
            # Explicitly teach: true_class_score < false_class_scores by margin δ
            delta = 0.8  # Fixed margin for separation
            
            true_scores = target_scores  # Scores for true classes [B]
            
            # Get all false class scores for each sample
            batch_size = scores.shape[0]
            false_scores_list = []
            for i in range(batch_size):
                # Mask to exclude true class
                false_mask = torch.ones(scores.shape[1], dtype=torch.bool, device=scores.device)
                false_mask[targets[i]] = False
                false_scores = scores[i][false_mask]  # [C-1] false class scores
                false_scores_list.append(false_scores.mean())  # Mean of false scores
            
            false_scores_mean = torch.stack(false_scores_list)  # [B]
            
            # Margin loss: ReLU(true_score - false_score_mean + δ)
            # This encourages: true_score < false_score_mean - δ
            margin = true_scores - false_scores_mean + delta
            discrimination_loss = torch.relu(margin).mean()
            
            # Compute basic metrics for all phases
            coverage_indicators = (target_scores <= tau).float()
            coverage = coverage_indicators.mean()
            pred_sets = scores <= tau
            set_sizes = pred_sets.float().sum(dim=1)
            avg_size = set_sizes.mean()
            
            # Phase-based loss combination with configurable weights
            if current_epoch <= 10:
                # Phase 1: ONLY discrimination loss
                loss = discrimination_loss * self.margin_weight
                coverage_loss = torch.tensor(0.0, device=scores.device)
                size_loss = torch.tensor(0.0, device=scores.device)
                
            elif current_epoch <= 20:
                # Phase 2: Discrimination + Coverage
                coverage_loss = (coverage - target_coverage).pow(2)
                loss = discrimination_loss * self.margin_weight + self.lambda1 * coverage_loss
                size_loss = torch.tensor(0.0, device=scores.device)
                
            else:
                # Phase 3: All losses with configured weights
                coverage_loss = (coverage - target_coverage).pow(2)
                size_loss = avg_size
                loss = discrimination_loss * self.margin_weight + self.lambda1 * coverage_loss + self.lambda2 * size_loss
            
            # Add stability loss if available
            if hasattr(self.scoring_fn, 'stability_loss'):
                loss = loss + self.scoring_fn.stability_loss
            
            # Add separation loss if available
            if hasattr(self.scoring_fn, 'separation_loss'):
                loss = loss + self.scoring_fn.separation_loss
            
            # Add L2 regularization if available
            if hasattr(self.scoring_fn, 'l2_reg'):
                loss = loss + self.scoring_fn.l2_reg
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss detected at batch {pbar.n}. Skipping update.")
                # Skip this batch
                continue
            
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
            # Store individual loss components for logging
            self.last_discrimination_loss = discrimination_loss.item()
            self.last_coverage_loss = coverage_loss.item() if not isinstance(coverage_loss, float) else coverage_loss
            self.last_size_loss = size_loss.item() if not isinstance(size_loss, float) else size_loss
            self.current_phase = "Phase 1 (Discrimination)" if current_epoch <= 10 else "Phase 2 (+ Coverage)" if current_epoch <= 20 else "Phase 3 (All losses)"
            
            loss_meter.update(loss.item())
            coverage_meter.update(coverage.item())
            size_meter.update(avg_size.item())
            
            pbar.set_postfix({
                'Phase': self.current_phase.split()[0] + self.current_phase.split()[1][1:2],  # "Phase X"
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
        non_empty_size_meter = AverageMeter()
        
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
                
                # Compute non-empty average set size
                non_empty_mask = set_sizes > 0
                if non_empty_mask.any():
                    non_empty_sizes = set_sizes[non_empty_mask]
                    non_empty_avg = non_empty_sizes.mean()
                    non_empty_size_meter.update(non_empty_avg.item(), n=non_empty_mask.sum().item())
                
                # Update metrics
                coverage_meter.update(coverage.item())
                size_meter.update(avg_size.item())
        
        return coverage_meter.avg, size_meter.avg, non_empty_size_meter.avg
    
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
            save_dir=plot_dir,
            val_non_empty_sizes=history['val_non_empty_sizes'] if 'val_non_empty_sizes' in history else None
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
        """Plot AUROC and ECE metrics over epochs"""
        if len(history['epochs']) < 2:
            return  # Need at least 2 epochs to plot
            
        plot_metrics_over_epochs(
            epochs=history['epochs'],
            auroc_values=history['auroc_values'],
            ece_values=history['ece_values'],
            model_names=[self.config['dataset']['name']],
            title=f"AUROC and ECE Metrics for {self.config['dataset']['name']}",
            save_path=os.path.join(plot_dir, 'auroc_ece_metrics.png')
        )