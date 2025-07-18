#!/usr/bin/env python3
"""
Universal caching script with chunked processing support for large datasets.
This script can handle any dataset configuration and prevents memory exhaustion.
"""

import os
import sys
import argparse
import logging
import time
import psutil
import torch

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(src_dir))

from src.core import ScoringFunction, ScoringFunctionTrainer, ConfigManager
from src.utils import set_seed

def setup_logging(verbose=False, log_file=None):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )

def get_memory_info():
    """Get current memory usage information"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # System memory
    virtual_memory = psutil.virtual_memory()
    
    return {
        'process_rss_gb': memory_info.rss / 1024 / 1024 / 1024,
        'process_vms_gb': memory_info.vms / 1024 / 1024 / 1024,
        'system_total_gb': virtual_memory.total / 1024 / 1024 / 1024,
        'system_available_gb': virtual_memory.available / 1024 / 1024 / 1024,
        'system_percent': virtual_memory.percent
    }

def log_memory_stats(stage=""):
    """Log current memory statistics"""
    mem_info = get_memory_info()
    logging.info(f"Memory stats {stage}:")
    logging.info(f"  Process RSS: {mem_info['process_rss_gb']:.2f} GB")
    logging.info(f"  System: {mem_info['system_available_gb']:.2f}/{mem_info['system_total_gb']:.2f} GB available ({100-mem_info['system_percent']:.1f}% free)")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            logging.info(f"  GPU {i}: {allocated:.2f}/{reserved:.2f} GB allocated/reserved")

def estimate_optimal_chunk_size(dataset_size, num_classes=1000):
    """Estimate optimal chunk size based on available memory"""
    mem_info = get_memory_info()
    available_gb = mem_info['system_available_gb']
    
    # Conservative estimate: use at most 50% of available memory
    safe_memory_gb = available_gb * 0.5
    
    # Estimate memory per sample based on number of classes
    # Each probability vector is num_classes * 4 bytes (float32)
    # Plus overhead for processing (~2x for safety)
    bytes_per_sample = num_classes * 4 * 2
    memory_per_sample_mb = bytes_per_sample / (1024 * 1024)
    
    # Calculate chunk size
    chunk_size = int((safe_memory_gb * 1024) / memory_per_sample_mb)
    
    # Apply reasonable bounds
    chunk_size = max(1000, min(chunk_size, 50000))
    
    # For smaller datasets, use a larger percentage of the dataset
    if dataset_size < 10000:
        chunk_size = min(chunk_size, dataset_size)
    
    logging.info(f"Estimated optimal chunk size: {chunk_size} samples")
    logging.info(f"Based on {safe_memory_gb:.1f} GB safe memory limit")
    logging.info(f"Estimated {memory_per_sample_mb:.3f} MB per sample")
    
    return chunk_size

def run_cache_verification(trainer, chunk_size, enable_memory_monitoring):
    """Run cache verification test (load from cache if exists)"""
    logging.info("\nRunning cache verification test...")
    
    # Reinitialize trainer (simulating a new run)
    new_trainer = ScoringFunctionTrainer(
        base_model=trainer.base_model,
        scoring_fn=trainer.scoring_fn,
        train_loader=trainer.train_loader,
        cal_loader=trainer.cal_loader,
        test_loader=trainer.test_loader,
        device=trainer.device,
        config=trainer.config
    )
    
    # Load cache
    start_time = time.time()
    new_trainer.cache_base_model_outputs(
        chunk_size=chunk_size,
        enable_memory_monitoring=enable_memory_monitoring
    )
    load_time = time.time() - start_time
    
    # Verify cache was loaded
    if new_trainer.is_cached:
        logging.info(f"Cache was successfully loaded in {load_time:.2f} seconds!")
        
        # Test that scoring works with cached data
        logging.info("\nTesting model with cached data...")
        inputs, targets = next(iter(new_trainer.test_loader))
        inputs = inputs.to(trainer.device)
        targets = targets.to(trainer.device)
        
        new_trainer.scoring_fn.eval()
        with torch.no_grad():
            scores, target_scores, _ = new_trainer._compute_scores(inputs, targets)
        
        logging.info(f"Successfully computed scores using cached data!")
        logging.info(f"- Scores shape: {scores.shape}")
        logging.info(f"- Target scores shape: {target_scores.shape}")
        logging.info(f"- Average true class score: {target_scores.mean().item():.4f}")
        
        return True, load_time
    else:
        logging.error("Cache was not loaded correctly!")
        return False, load_time

def main():
    parser = argparse.ArgumentParser(
        description='Cache dataset with chunked processing to prevent memory issues'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., src/config/imagenet.yaml)')
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Number of samples per chunk (default: auto-detect based on memory)')
    parser.add_argument('--no-memory-monitoring', action='store_true',
                       help='Disable memory monitoring')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of cache even if it exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log to file (in addition to console)')
    parser.add_argument('--verify', action='store_true',
                       help='Run verification test after caching')
    parser.add_argument('--dry-run', action='store_true',
                       help='Estimate memory usage without actually caching')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Log initial memory state
    log_memory_stats("at startup")
    
    # Load configuration
    logging.info(f"Loading config from: {args.config}")
    config = ConfigManager(args.config)
    config.setup_paths()
    config.setup_device()
    
    logging.info(f"Using config: {args.config}")
    logging.info(f"Using device: {config['device']}")
    logging.info(f"Dataset: {config['dataset']['name']}")
    
    # Set random seed
    set_seed(42)
    
    # Initialize dataset
    logging.info("Initializing dataset...")
    dataset_class = config.get_dataset_class()
    dataset = dataset_class(config)
    dataloaders = dataset.get_dataloaders()
    
    # Log dataset statistics
    train_size = len(dataloaders['train'].dataset)
    cal_size = len(dataloaders['calibration'].dataset)
    test_size = len(dataloaders['test'].dataset)
    total_size = train_size + cal_size + test_size
    
    logging.info(f"Dataset sizes:")
    logging.info(f"  Train: {train_size:,} samples")
    logging.info(f"  Calibration: {cal_size:,} samples")
    logging.info(f"  Test: {test_size:,} samples")
    logging.info(f"  Total: {total_size:,} samples")
    
    # Determine chunk size
    num_classes = config['dataset'].get('num_classes', 1000)
    if args.chunk_size is None:
        chunk_size = estimate_optimal_chunk_size(total_size, num_classes)
    else:
        chunk_size = args.chunk_size
        logging.info(f"Using user-specified chunk size: {chunk_size}")
    
    # Calculate number of chunks
    num_chunks_train = (train_size + chunk_size - 1) // chunk_size
    num_chunks_cal = (cal_size + chunk_size - 1) // chunk_size
    num_chunks_test = (test_size + chunk_size - 1) // chunk_size
    total_chunks = num_chunks_train + num_chunks_cal + num_chunks_test
    
    logging.info(f"Processing strategy:")
    logging.info(f"  Train: {num_chunks_train} chunks")
    logging.info(f"  Calibration: {num_chunks_cal} chunks")
    logging.info(f"  Test: {num_chunks_test} chunks")
    logging.info(f"  Total: {total_chunks} chunks")
    
    if args.dry_run:
        logging.info("Dry run complete. Exiting without caching.")
        return
    
    # Load base model
    logging.info("Loading base model...")
    base_model = dataset.get_model()
    logging.info("Base model loaded successfully")
    log_memory_stats("after loading model")
    
    # Initialize scoring function
    scoring_fn = ScoringFunction(
        input_dim=None,  # Will be taken from config
        hidden_dims=config['scoring_function']['hidden_dims'],
        output_dim=None,  # Will default to num_classes
        config=config
    ).to(config['device'])
    logging.info(f"Scoring function initialized with input_dim={num_classes}, output_dim={num_classes}")
    
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
    
    # Check if cache exists
    cache_dir = trainer._generate_cache_path()
    cache_exists = os.path.exists(cache_dir) and trainer._check_cache_validity(cache_dir)
    
    if cache_exists and not args.force:
        logging.info(f"Valid cache already exists at: {cache_dir}")
        if args.verify:
            logging.info("Running verification test on existing cache...")
            success, load_time = run_cache_verification(trainer, chunk_size, not args.no_memory_monitoring)
            if success:
                logging.info(f"Verification passed! Cache loaded in {load_time:.2f} seconds")
        else:
            logging.info("Use --force to regenerate cache or --verify to test existing cache")
        return
    
    if cache_exists and args.force:
        logging.info("Force flag set, regenerating cache...")
    elif not cache_exists:
        logging.info("No valid cache found, generating new cache...")
    
    # Start caching process
    logging.info("=" * 60)
    logging.info("Starting chunked caching process...")
    logging.info(f"Cache will be saved to: {cache_dir}")
    logging.info("=" * 60)
    
    start_time = time.time()
    
    try:
        trainer.cache_base_model_outputs(
            chunk_size=chunk_size,
            enable_memory_monitoring=not args.no_memory_monitoring
        )
        
        first_run_time = time.time() - start_time
        logging.info("=" * 60)
        logging.info(f"Caching completed successfully in {first_run_time/60:.1f} minutes")
        logging.info("=" * 60)
        
        # Verify cache files
        cache_files = [
            f"{split}_probs.pt" for split in ["train", "cal", "test"]
        ] + [
            f"{split}_targets.pt" for split in ["train", "cal", "test"]
        ] + ["metadata.json"]
        
        logging.info("Cache contents:")
        total_size_mb = 0
        for file in cache_files:
            file_path = os.path.join(cache_dir, file)
            if os.path.exists(file_path):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size_mb += file_size_mb
                logging.info(f"  {file}: {file_size_mb:.1f} MB")
            else:
                logging.error(f"  {file}: MISSING!")
        
        logging.info(f"Total cache size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
        
        # Run verification if requested
        if args.verify:
            success, load_time = run_cache_verification(trainer, chunk_size, not args.no_memory_monitoring)
            if success:
                speedup = first_run_time / max(load_time, 0.001)
                logging.info(f"Cache verification passed! Speedup: {speedup:.2f}x")
            else:
                logging.error("Cache verification failed!")
        
        # Final memory stats
        log_memory_stats("after completion")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Caching failed after {elapsed_time/60:.1f} minutes")
        logging.error(f"Error: {str(e)}")
        log_memory_stats("at error")
        raise

if __name__ == '__main__':
    main()