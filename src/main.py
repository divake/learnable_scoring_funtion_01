# src/main.py

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import logging
import os
import torchvision.models as models

from utils.config import Config
from utils.metrics import compute_coverage_and_size, compute_tau
from utils.visualization import (plot_training_curves, plot_score_distributions,
                               plot_set_size_distribution)
from models.scoring_function import ScoringFunction, ConformalPredictor
from training.trainer import ScoringFunctionTrainer
from cifar_split import setup_cifar10
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR
from utils.visualization import (plot_training_curves, plot_score_distributions,
                               plot_set_size_distribution, plot_scoring_function_behavior)
from utils.seed import set_seed

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

# def inspect_saved_model(model_path):
#     """Inspect the architecture of a saved model."""
#     state_dict = torch.load(model_path)
#     print("\nSaved model state_dict keys:")
#     for key in state_dict.keys():
#         print(f"  {key}")

def main():
    # Set seed
    set_seed(42)

    # Load configuration
    config = Config()
    setup_logging(config)
    logging.info("Starting training process")
    
    # Inspect saved model first
    model_path = os.path.join(config.model_dir, 'resnet18_cifar10_best.pth')
    # logging.info("Inspecting saved model architecture...")
    # inspect_saved_model(model_path)
    
    # Load data
    train_loader, cal_loader, test_loader, _, _, _ = setup_cifar10(batch_size=config.batch_size)
    logging.info("Data loaded successfully")
    
    # Load pretrained ResNet model
    base_model = models.resnet18(weights=None)
    # Modify fc layer to match the saved architecture
    base_model.fc = nn.Sequential(
        nn.Identity(),  # This will be fc.0
        nn.Linear(base_model.fc.in_features, 10)  # This will be fc.1
    )
    base_model.load_state_dict(torch.load(model_path))
    base_model = base_model.to(config.device)
    base_model.eval()
    logging.info("Base model loaded successfully")
    
    # Initialize scoring function
    scoring_fn = ScoringFunction(
        input_dim=1,
        hidden_dims=config.hidden_dims,
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
    optimizer = optim.AdamW(scoring_fn.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Longer warm-up
        div_factor=20,
        final_div_factor=100,
        anneal_strategy='cos'
    )
    
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    
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
            coverage_target=config.target_coverage
        )
        
        # Train epoch
        train_loss, train_coverage, train_size = trainer.train_epoch(
            optimizer=optimizer,
            tau=tau
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
                
                # Get scores for all classes
                scores = torch.zeros_like(probs, device=config.device)
                for i in range(probs.size(1)):
                    class_probs = probs[:, i:i+1]
                    scores[:, i:i+1] = scoring_fn(class_probs)
                
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

        # Update plots every epoch
        logging.info("Updating plots...")
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
        
        plot_score_distributions(
            true_scores=np.array(true_scores),
            false_scores=np.array(false_scores),
            tau=tau,
            save_dir=config.plot_dir
        )

        plot_set_size_distribution(
            set_sizes=np.array(set_sizes),
            save_dir=config.plot_dir
        )

        plot_scoring_function_behavior(scoring_fn, config.device, config.plot_dir)
        
        # Save best model
        if train_loss < best_loss and train_size < 2.0:
            best_loss = train_loss
            torch.save(scoring_fn.state_dict(),
                      os.path.join(config.model_dir, 'scoring_function_best.pth'))
            logging.info("Saved new best model")
        
        # Early stopping check
        early_stopping(train_loss, val_coverage, val_size)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered!")
            break

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
            
            # Get scores for all classes
            scores = torch.zeros_like(probs, device=config.device)
            for i in range(probs.size(1)):
                class_probs = probs[:, i:i+1]
                scores[:, i:i+1] = scoring_fn(class_probs)
            
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