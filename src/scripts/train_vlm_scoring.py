#!/usr/bin/env python
"""
Script to train the MLP scoring function for VLM logits
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(src_dir))

import torch
from src.core import ScoringFunction, ScoringFunctionTrainer, ConfigManager
from src.utils import set_seed
from src.datasets.vlm import Dataset

def setup_logging(config, model_name, dataset_name):
    """Setup logging configuration"""
    # Create log directory if it doesn't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['log_dir'], f'vlm_{model_name}_{dataset_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train scoring function for VLM logits')
    parser.add_argument('--config', type=str, default='src/config/vlm.yaml',
                      help='Path to config file (e.g., src/config/vlm.yaml)')
    parser.add_argument('--model', type=str, default='cogagent-vqa-hf',
                      help='VLM model to use (default: cogagent-vqa-hf)')
    parser.add_argument('--dataset', type=str, default='ai2d',
                      help='Dataset to use (default: ai2d)')
    parser.add_argument('--gpu', type=int, default=None,
                      help='GPU ID to use (overrides config)')
    args = parser.parse_args()
    
    # Load and setup configuration
    config = ConfigManager(args.config)
    config.setup_paths()
    config.setup_device()
    
    # Override GPU if specified
    if args.gpu is not None:
        config.config['device'] = f"cuda:{args.gpu}"
    
    # Override model and dataset in config
    config.config['dataset']['default_model'] = args.model
    config.config['dataset']['default_dataset'] = args.dataset
    
    # Setup logging
    setup_logging(
        config.config, 
        config.config['dataset']['default_model'], 
        config.config['dataset']['default_dataset']
    )
    
    logging.info(f"Using config: {args.config}")
    logging.info(f"Using device: {config.config['device']}")
    logging.info(f"Training scoring function for VLM model {args.model} on dataset {args.dataset}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize dataset
    dataset = Dataset(config.config)
    dataloaders = dataset.get_dataloaders()
    
    # Get the dummy model (for VLM we use precomputed logits)
    base_model = dataset.get_model()
    logging.info("Base model (dummy) loaded successfully")
    
    # Initialize scoring function with appropriate dimensions for compatibility with core/metrics.py
    scoring_fn = ScoringFunction(
        input_dim=1,  # Changed to 1 to match the expected input in core/metrics.py
        hidden_dims=config.config['scoring_function']['hidden_dims'],
        output_dim=1,  # Changed to 1 to match the expected output in core/metrics.py
        config=config.config
    ).to(config.config['device'])
    logging.info("Scoring function initialized for compatibility with core/metrics.py")
    
    # Initialize trainer
    trainer = ScoringFunctionTrainer(
        base_model=base_model,
        scoring_fn=scoring_fn,
        train_loader=dataloaders['train'],
        cal_loader=dataloaders['calibration'],
        test_loader=dataloaders['test'],
        device=config.config['device'],
        config=config.config
    )
    
    # Make sure save directory exists
    save_dir = os.path.join(config.config.get('model_dir', 'models'), 'vlm')
    os.makedirs(save_dir, exist_ok=True)
    
    # Make sure plot directory exists
    plot_dir = config.config.get('plot_dir', 'plots/vlm')
    os.makedirs(plot_dir, exist_ok=True)
    
    logging.info(f"Created directories: {save_dir} and {plot_dir}")
    
    # Training loop
    results = trainer.train(
        num_epochs=config.config['num_epochs'],
        target_coverage=config.config['target_coverage'],
        tau_config=config.config['tau'],
        set_size_config=config.config['set_size'],
        save_dir=save_dir,
        plot_dir=plot_dir
    )
    
    logging.info(f"Training results: {results}")
    logging.info("Training completed!")

if __name__ == '__main__':
    main() 