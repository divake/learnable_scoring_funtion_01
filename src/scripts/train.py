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

def setup_logging(config):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['log_dir'], f'training_{timestamp}.log')
    
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
    parser = argparse.ArgumentParser(description='Train scoring function')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file (e.g., src/config/cifar10.yaml)')
    args = parser.parse_args()
    
    # Load and setup configuration
    config = ConfigManager(args.config)
    config.setup_paths()
    config.setup_device()
    
    # Setup logging
    setup_logging(config)
    logging.info(f"Using config: {args.config}")
    logging.info(f"Using device: {config['device']}")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize dataset
    dataset_class = config.get_dataset_class()
    dataset = dataset_class(config)
    dataloaders = dataset.get_dataloaders()
    
    # Load base model
    base_model = dataset.get_model()
    logging.info("Base model loaded successfully")
    
    # Initialize scoring function
    scoring_fn = ScoringFunction(
        input_dim=1,
        hidden_dims=config['scoring_function']['hidden_dims'],
        output_dim=1,
        config=config
    ).to(config['device'])
    logging.info("Scoring function initialized")
    
    # Initialize trainer
    trainer = ScoringFunctionTrainer(
        base_model=base_model,
        scoring_fn=scoring_fn,
        train_loader=dataloaders['train'],
        cal_loader=dataloaders['calibration'],
        test_loader=dataloaders['test'],
        device=config['device'],
        config=config
    )
    
    # Training loop
    trainer.train(
        num_epochs=config['num_epochs'],
        target_coverage=config['target_coverage'],
        tau_config=config['tau'],
        set_size_config=config['set_size'],
        save_dir=config['model_dir'],
        plot_dir=config['plot_dir']
    )
    
    logging.info("Training completed!")

if __name__ == '__main__':
    main() 