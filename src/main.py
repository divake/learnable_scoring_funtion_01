import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import timm
import torchvision.transforms as transforms

from utils.config import Config
from utils.metrics import compute_coverage_and_size, compute_tau
from utils.visualization import (plot_training_curves, plot_score_distributions,
                               plot_set_size_distribution, plot_scoring_function_behavior)
from models.scoring_function import ScoringFunction, ConformalPredictor
from training.trainer import ScoringFunctionTrainer
from cifar_split import setup_cifar100
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR
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

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Load configuration
    config = Config()
    
    # Clean up any temporary files from previous runs
    cleanup_temp_files(config)
    
    # Verify write permissions
    for dir_path in [config.model_dir, config.plot_dir]:
        try:
            test_file = os.path.join(dir_path, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            raise RuntimeError(f"Cannot write to {dir_path}: {e}")
    
    setup_logging(config)
    logging.info("Starting training process")
    logging.info(f"Model directory: {config.model_dir}")
    logging.info(f"Plot directory: {config.plot_dir}")
    
    # Setup transforms matching ViT training
    transform = get_transforms()
    
    # Load data with correct transforms
    train_loader, cal_loader, test_loader, train_dataset, cal_dataset, test_dataset = setup_cifar100(
        batch_size=config.batch_size
    )
    
    # Update transforms
    train_dataset.dataset.transform = transform
    cal_dataset.dataset.transform = transform
    test_dataset.transform = transform
    
    # Load pretrained ViT model with correct configuration
    base_model = timm.create_model(
        'vit_base_patch16_224_in21k',
        pretrained=False,
        num_classes=100,
        img_size=96,
        drop_path_rate=0.1,
        drop_rate=0.1
    )
    
    # Enable gradient checkpointing
    base_model.set_grad_checkpointing(enable=True)
    
    # Load trained weights
    base_model.load_state_dict(torch.load(config.vit_model_path))
    base_model = base_model.to(config.device)
    base_model.eval()
    
    # Initialize scoring function
    scoring_fn = ScoringFunction(
        input_dim=1,
        hidden_dims=[32, 16],  # Fixed architecture
        output_dim=1
    ).to(config.device)
    logging.info("Scoring function initialized")
    
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
    
    # Training setup
    optimizer = optim.AdamW(scoring_fn.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=20,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    # early_stopping = EarlyStopping()

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
    for epoch in range(config.num_epochs):
        logging.info(f"Epoch {epoch+1}/{config.num_epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")
        
        # Compute tau on calibration set
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
            tau=tau, epoch=epoch
        )
        
        # Evaluate
        val_coverage, val_size = trainer.evaluate(test_loader, tau)
        
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
        
        # Save best model
        if train_loss < best_loss and train_size < 2.0:
            best_loss = train_loss
            torch.save(scoring_fn.state_dict(),
                      os.path.join(config.model_dir, 'scoring_function_best.pth'))
            logging.info("Saved new best model")
        
        # Plot training curves
        plot_training_curves(
            epochs=history['epochs'],
            train_losses=history['train_losses'],
            train_coverages=history['train_coverages'],
            train_sizes=history['train_sizes'],
            val_coverages=history['val_coverages'],
            val_sizes=history['val_sizes'],
            tau_values=history['tau_values'],
            save_dir=config.plot_dir
        )
        
        # Early stopping check
        # early_stopping(train_loss, val_coverage, val_size)
        # if early_stopping.early_stop:
        #     logging.info("Early stopping triggered!")
        #     break
    
    # In main.py, after training completes and before final logging

    logging.info("Training completed!")

    # Load best model for final evaluation
    scoring_fn.load_state_dict(torch.load(os.path.join(config.model_dir, 'scoring_function_best.pth')))

    # Collect data for distributions
    true_scores = []
    false_scores = []
    set_sizes = []

    scoring_fn.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            # Get probabilities
            logits = base_model(inputs)
            probs = torch.softmax(logits, dim=1)
            batch_size = probs.size(0)
            
            # Process probabilities in chunks like during training
            scores = torch.zeros(batch_size, config.num_classes, device=config.device)
            chunk_size = 1000
            flat_probs = probs.reshape(-1, 1)  # Reshape to [N, 1]
            
            for i in range(0, flat_probs.size(0), chunk_size):
                end_idx = min(i + chunk_size, flat_probs.size(0))
                chunk_scores = scoring_fn(flat_probs[i:end_idx])
                scores.view(-1)[i:end_idx] = chunk_scores.squeeze()
            
            # Collect true class scores
            true_class_scores = scores[torch.arange(len(targets)), targets].cpu().numpy()
            true_scores.extend(true_class_scores)
            
            # Collect false class scores (all other classes)
            mask = torch.ones_like(scores, dtype=bool)
            mask[torch.arange(len(targets)), targets] = False
            false_class_scores = scores[mask].cpu().numpy()
            false_scores.extend(false_class_scores)
            
            # Compute set sizes using the final tau
            pred_sets = (scores <= tau).sum(dim=1)
            set_sizes.extend(pred_sets.cpu().numpy())

    # Plot all distributions
    logging.info("Plotting score distributions...")
    plot_score_distributions(
        true_scores=np.array(true_scores),
        false_scores=np.array(false_scores),
        tau=tau,
        save_dir=config.plot_dir
    )

    logging.info("Plotting set size distribution...")
    plot_set_size_distribution(
        set_sizes=np.array(set_sizes),
        save_dir=config.plot_dir
    )

    # Plot scoring function behavior last
    logging.info("Plotting scoring function behavior...")
    plot_scoring_function_behavior(scoring_fn, config.device, config.plot_dir)

    logging.info("All visualizations saved!")

if __name__ == "__main__":
    main()