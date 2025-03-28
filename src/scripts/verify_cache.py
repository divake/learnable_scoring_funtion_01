import os
import sys
import argparse
import logging
import time

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(src_dir))

import torch
from src.core import ScoringFunction, ScoringFunctionTrainer, ConfigManager
from src.utils import set_seed

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Verify caching of base model outputs')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file (e.g., src/config/cifar10.yaml)')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load and setup configuration
    config = ConfigManager(args.config)
    config.setup_paths()
    config.setup_device()
    
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
    
    # First run: Generate cache
    logging.info("First run: Testing cache generation...")
    start_time = time.time()
    trainer.cache_base_model_outputs()
    first_run_time = time.time() - start_time
    logging.info(f"First run completed in {first_run_time:.2f} seconds")
    
    # Verify if cache was created
    cache_dir = trainer._generate_cache_path()
    logging.info(f"Cache directory: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        logging.error(f"Cache directory {cache_dir} was not created!")
        return
        
    # Check if cache files exist
    cache_files = [
        f"{split}_probs.pt" for split in ["train", "cal", "test"]
    ] + [
        f"{split}_targets.pt" for split in ["train", "cal", "test"]
    ] + ["metadata.json"]
    
    for file in cache_files:
        file_path = os.path.join(cache_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            logging.info(f"Found cache file: {file} ({file_size:.2f} MB)")
        else:
            logging.error(f"Cache file {file} is missing!")
    
    # Second run: Load from cache
    logging.info("\nSecond run: Testing cache loading...")
    
    # Reinitialize trainer (simulating a new run)
    trainer = ScoringFunctionTrainer(
        base_model=base_model,
        scoring_fn=scoring_fn,
        train_loader=dataloaders['train'],
        cal_loader=dataloaders['calibration'],
        test_loader=dataloaders['test'],
        device=config['device'],
        config=config
    )
    
    # Load cache
    start_time = time.time()
    trainer.cache_base_model_outputs()
    second_run_time = time.time() - start_time
    logging.info(f"Second run completed in {second_run_time:.2f} seconds")
    
    # Verify cache was loaded
    if trainer.is_cached:
        speedup = first_run_time / max(second_run_time, 0.001)  # Avoid division by zero
        logging.info(f"Cache was successfully loaded! Speedup: {speedup:.2f}x")
    else:
        logging.error("Cache was not loaded correctly!")
    
    # Verify using the cache for prediction scoring
    logging.info("\nTesting model with cached data...")
    
    # Get a small batch from the test loader for testing
    inputs, targets = next(iter(trainer.test_loader))
    inputs = inputs.to(config['device'])
    targets = targets.to(config['device'])
    
    # Test that scoring works with cached data
    trainer.scoring_fn.eval()
    with torch.no_grad():
        scores, target_scores, _ = trainer._compute_scores(inputs, targets)
    
    logging.info(f"Successfully computed scores using cached data!")
    logging.info(f"- Scores shape: {scores.shape}")
    logging.info(f"- Target scores shape: {target_scores.shape}")
    logging.info(f"- Average true class score: {target_scores.mean().item():.4f}")
    
    logging.info("Cache verification completed!")

if __name__ == '__main__':
    main() 