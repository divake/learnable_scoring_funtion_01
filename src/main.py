import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import timm
import torchvision.transforms as transforms
from typing import Dict, Any, List, Optional, Tuple

from utils.config import Config
from utils.metrics import compute_coverage_and_size, compute_tau
from utils.visualization import (plot_training_curves, plot_score_distributions,
                               plot_set_size_distribution, plot_scoring_function_behavior)
from utils.logger import Logger
from utils.experiment import Experiment
from utils.callbacks import ModelCheckpoint
from utils.exceptions import (ConfigurationError, ModelError, DataError,
                            TrainingError, ValidationError)
from models.scoring_function import ScoringFunction, ConformalPredictor
from training.trainer import ScoringFunctionTrainer
from cifar_split import setup_cifar100
from utils.seed import set_seed

def cleanup_temp_files(config):
    """Clean up temporary and backup files"""
    patterns = ['.tmp', '.bak']
    for pattern in patterns:
        temp_file = config.scoring_model_path + pattern
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logging.info(f"Cleaned up {temp_file}")
            except Exception as e:
                logging.warning(f"Could not remove {temp_file}: {e}")

def save_scoring_model(model, path, backup=True):
    """Safe model saving with backup"""
    temp_path = path + '.tmp'
    backup_path = path + '.bak'
    
    try:
        # Save to temp file first
        torch.save(model.state_dict(), temp_path)
        
        # If backup is requested and original file exists
        if backup and os.path.exists(path):
            os.replace(path, backup_path)
            
        # Move temp file to final location
        os.replace(temp_path, path)
        return True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def load_scoring_model(model, path):
    """Safe model loading with backup fallback"""
    backup_path = path + '.bak'
    
    try:
        # Try loading main file
        model.load_state_dict(torch.load(path))
        return True
    except Exception as e:
        if os.path.exists(backup_path):
            try:
                # Try loading backup
                model.load_state_dict(torch.load(backup_path))
                return True
            except:
                pass
        raise e

def setup_logging(config):
    """Setup logging configuration."""
    log_path = os.path.join(config.base_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def get_transforms():
    """Get the transforms used during ViT training"""
    return transforms.Compose([
        transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761])
    ])

def setup_experiment(config: Config) -> Tuple[Experiment, Logger]:
    """Setup experiment and logger.
    
    Args:
        config: Configuration object
        
    Returns:
        tuple: Experiment and Logger instances
    """
    # Initialize logger
    logger = Logger(config.log_file)
    
    # Create experiment
    experiment = Experiment(
        name="conformal_prediction",
        base_dir=config.base_dir,
        config=config.__dict__,
        logger=logger
    )
    
    return experiment, logger

def setup_model(config: Config, logger: Logger) -> Tuple[nn.Module, nn.Module]:
    """Setup base model and scoring function.
    
    Args:
        config: Configuration object
        logger: Logger instance
        
    Returns:
        tuple: Base model and scoring function
    """
    try:
        # Load pretrained ViT model
        base_model = timm.create_model(
            'vit_base_patch16_224_in21k',
            pretrained=False,
            num_classes=config.num_classes,
            img_size=96,
            drop_path_rate=0.1,
            drop_rate=0.1
        )
        
        # Enable gradient checkpointing
        base_model.set_grad_checkpointing(enable=True)
        
        # Load trained weights with weights_only=True
        base_model.load_state_dict(torch.load(config.vit_model_path, weights_only=True))
        base_model = base_model.to(config.device)
        base_model.eval()
        
        # Initialize scoring function
        scoring_fn = ScoringFunction(
            input_dim=1,
            hidden_dims=config.hidden_dims,
            output_dim=1
        ).to(config.device)
        
        logger.info("Models initialized successfully")
        logger.log_model_summary(scoring_fn)
        
        return base_model, scoring_fn
    
    except Exception as e:
        logger.error(f"Failed to setup models: {str(e)}")
        raise ModelError(f"Failed to setup models: {str(e)}")

def setup_data(config: Config, logger: Logger) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Setup data loaders and datasets.
    
    Args:
        config: Configuration object
        logger: Logger instance
        
    Returns:
        tuple: Data loaders and datasets
    """
    try:
        # Setup transforms
        transform = transforms.Compose([
            transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                              std=[0.2675, 0.2565, 0.2761])
        ])
        
        # Load data
        train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_cifar100(
            batch_size=config.batch_size
        )
        
        # Update transforms
        train_dataset.dataset.transform = transform
        cal_dataset.dataset.transform = transform
        test_dataset.transform = transform
        
        logger.info("Data setup completed successfully")
        logger.info(f"Train size: {len(train_dataset)}")
        logger.info(f"Calibration size: {len(cal_dataset)}")
        logger.info(f"Test size: {len(test_dataset)}")
        
        return train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset
    
    except Exception as e:
        logger.error(f"Failed to setup data: {str(e)}")
        raise DataError(f"Failed to setup data: {str(e)}")

def train(
    config: Config,
    experiment: Experiment,
    logger: Logger,
    base_model: nn.Module,
    scoring_fn: nn.Module,
    train_loader: Any,
    cal_loader: Any,
    test_loader: Any
) -> Dict[str, Any]:
    """Main training loop.
    
    Args:
        config: Configuration object
        experiment: Experiment instance
        logger: Logger instance
        base_model: Base model
        scoring_fn: Scoring function
        train_loader: Training data loader
        cal_loader: Calibration data loader
        test_loader: Test data loader
        
    Returns:
        dict: Training history containing metrics and values
    """
    try:
        # Initialize trainer
        trainer = ScoringFunctionTrainer(
            base_model=base_model,
            scoring_fn=scoring_fn,
            train_loader=train_loader,
            cal_loader=cal_loader,
            test_loader=test_loader,
            device=config.device,
            lambda1=config.lambda1,
            lambda2=config.lambda2
        )
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            scoring_fn.parameters(),
            lr=config.learning_rate,
            weight_decay=config.optimizer_config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=config.optimizer_config['scheduler']['pct_start'],
            div_factor=config.optimizer_config['scheduler']['div_factor'],
            final_div_factor=config.optimizer_config['scheduler']['final_div_factor'],
            anneal_strategy=config.optimizer_config['scheduler']['anneal_strategy']
        )
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=experiment.get_checkpoint_path('best_model'),
                monitor='val_coverage',
                mode='max',
                logger=logger
            )
        ]
        
        # Training history
        history = {
            'epochs': [],
            'train_losses': [],
            'train_coverages': [],
            'train_sizes': [],
            'val_coverages': [],
            'val_sizes': [],
            'tau_values': [],
            'true_scores': [],
            'false_scores': [],
            'set_sizes': []
        }
        
        logger.info("Starting training")
        
        # Training loop
        for epoch in range(config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{config.num_epochs}")
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Compute tau
            tau = compute_tau(
                cal_loader=cal_loader,
                scoring_fn=scoring_fn,
                base_model=base_model,
                device=config.device,
                coverage_target=config.target_coverage,
                epoch=epoch
            )
            
            # Train epoch
            train_loss, train_coverage, train_size = trainer.train_epoch(
                optimizer=optimizer,
                tau=tau,
                epoch=epoch
            )
            
            # Evaluate
            val_coverage, val_size = trainer.evaluate(test_loader, tau)
            
            # Update scheduler
            scheduler.step()
            
            # Update history
            metrics = {
                'train_loss': train_loss,
                'train_coverage': train_coverage,
                'train_size': train_size,
                'val_coverage': val_coverage,
                'val_size': val_size,
                'tau': tau
            }
            
            # Update callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch, {'model': scoring_fn, **metrics})
            
            # Log metrics
            logger.log_metrics(epoch, metrics)
            
            # Update history
            history['epochs'].append(epoch)
            for key, value in metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Save history
            experiment.save_metrics(history, 'history.json')
            
            # Plot training curves only if we have enough data points
            if len(history['epochs']) > 1:
                try:
                    plot_training_curves(
                        epochs=history['epochs'],
                        train_losses=history['train_losses'],
                        train_coverages=history['train_coverages'],
                        train_sizes=history['train_sizes'],
                        val_coverages=history['val_coverages'],
                        val_sizes=history['val_sizes'],
                        tau_values=history['tau_values'],
                        save_dir=experiment.plot_dir
                    )
                except Exception as plot_error:
                    logger.warning(f"Failed to plot training curves: {str(plot_error)}")
            
        logger.info("Training completed")
        return history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise TrainingError(f"Training failed: {str(e)}")

def main():
    logger = None  # Initialize logger as None at the start
    history = None  # Initialize history as None
    try:
        # Load configuration
        config = Config()
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Setup experiment and logger
        experiment, logger = setup_experiment(config)
        
        # Log system information
        logger.log_system_info(config)
        
        # Setup models
        base_model, scoring_fn = setup_model(config, logger)
        
        # Setup data
        train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_data(config, logger)
        
        # Train model
        history = train(
            config=config,
            experiment=experiment,
            logger=logger,
            base_model=base_model,
            scoring_fn=scoring_fn,
            train_loader=train_loader,
            cal_loader=cal_loader,
            test_loader=test_loader
        )
        
        # Final evaluation and visualization
        logger.info("Generating final visualizations")
        
        # Load best model
        scoring_fn.load_state_dict(torch.load(experiment.get_checkpoint_path('best_model'), weights_only=True))
        
        if history:  # Only generate visualizations if we have history
            # Generate visualizations
            plot_score_distributions(
                true_scores=np.array(history['true_scores']),
                false_scores=np.array(history['false_scores']),
                tau=history['tau_values'][-1],
                save_dir=experiment.plot_dir
            )
            
            plot_set_size_distribution(
                set_sizes=np.array(history['set_sizes']),
                save_dir=experiment.plot_dir
            )
            
            plot_scoring_function_behavior(
                scoring_fn,
                config.device,
                experiment.plot_dir
            )
        
        logger.info("Experiment completed successfully")
        
        # Archive experiment
        experiment.archive()
        
    except Exception as e:
        if logger:
            logger.critical(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()