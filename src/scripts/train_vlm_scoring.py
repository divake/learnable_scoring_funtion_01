#!/usr/bin/env python
"""
Script to train the MLP scoring function for VLM logits
"""

import os
import sys
import argparse
import logging
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback
from datetime import datetime

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(src_dir))

import torch
from src.core import ScoringFunction, ScoringFunctionTrainer, ConfigManager
from src.utils import set_seed
from src.datasets.vlm import Dataset

def setup_logging(config, model_name=None, dataset_name=None):
    """Setup logging configuration"""
    # Create log directory if it doesn't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config['log_dir'], f'vlm_{model_name or "all"}_{dataset_name or "all"}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def train_model(model_name, dataset_name, config, gpu_id=None):
    """Train a single model and return the results"""
    model_config = config.copy()
    
    # Set device appropriately based on provided GPU ID
    if gpu_id is not None:
        model_config['device'] = f"cuda:{gpu_id}"
        # Also set CUDA_VISIBLE_DEVICES for torch
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Ensure device is set correctly
    if 'device' not in model_config or model_config['device'] is None:
        model_config['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Update config with current model and dataset
    model_config['dataset']['default_model'] = model_name
    model_config['dataset']['default_dataset'] = dataset_name
    
    # Set up model-specific logging
    model_log_file = os.path.join(model_config['log_dir'], f'vlm_{model_name}_{dataset_name}.log')
    model_logger = logging.getLogger(f"{model_name}_{dataset_name}")
    model_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(model_log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    model_logger.addHandler(file_handler)
    model_logger.propagate = False
    
    model_logger.info(f"Starting training for VLM model {model_name} on dataset {dataset_name}")
    model_logger.info(f"Using device: {model_config['device']}")
    
    # Check available CUDA devices
    if torch.cuda.is_available():
        model_logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
        model_logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        model_logger.warning("CUDA is not available, using CPU")
    
    try:
        # Initialize dataset
        model_logger.info("Initializing dataset...")
        dataset = Dataset(model_config)
        dataloaders = dataset.get_dataloaders()
        
        # Get the dummy model
        model_logger.info("Creating base model...")
        base_model = dataset.get_model()
        model_logger.info("Base model (dummy) loaded successfully")
        
        # Initialize scoring function
        model_logger.info("Initializing scoring function...")
        scoring_fn = ScoringFunction(
            input_dim=1,
            hidden_dims=model_config['scoring_function']['hidden_dims'],
            output_dim=1,
            config=model_config
        ).to(model_config['device'])
        
        # Initialize trainer
        model_logger.info("Setting up trainer...")
        trainer = ScoringFunctionTrainer(
            base_model=base_model,
            scoring_fn=scoring_fn,
            train_loader=dataloaders['train'],
            cal_loader=dataloaders['calibration'],
            test_loader=dataloaders['test'],
            device=model_config['device'],
            config=model_config
        )
        
        # Make sure save directory exists
        save_dir = os.path.join(model_config.get('model_dir', 'models/vlm'))
        os.makedirs(save_dir, exist_ok=True)
        
        # Make sure plot directory exists
        plot_dir = os.path.join(model_config.get('plot_dir', 'plots/vlm'), model_name)
        os.makedirs(plot_dir, exist_ok=True)
        
        # Training loop
        model_logger.info(f"Starting training for {model_config['num_epochs']} epochs...")
        results = trainer.train(
            num_epochs=model_config['num_epochs'],
            target_coverage=model_config['target_coverage'],
            tau_config=model_config['tau'],
            set_size_config=model_config['set_size'],
            save_dir=save_dir,
            plot_dir=plot_dir
        )
        
        # Print final results with clear formatting for easy parsing
        model_logger.info(f"=" * 50)
        model_logger.info(f"FINAL RESULTS FOR MODEL {model_name} ON DATASET {dataset_name}")
        model_logger.info(f"=" * 50)
        model_logger.info(f"Coverage: {results.get('coverage', float('nan')):.4f}")
        model_logger.info(f"Set Size: {results.get('avg_set_size', float('nan')):.4f}")
        model_logger.info(f"AUROC: {results.get('auroc', float('nan')):.4f}")
        model_logger.info(f"AUARC: {results.get('auarc', float('nan')):.4f}")
        model_logger.info(f"ECE: {results.get('ece', float('nan')):.4f}")
        model_logger.info(f"Tau: {results.get('tau', float('nan'))}")
        
        if 'avg_set_size' in results and results['avg_set_size'] > 0:
            efficiency = results['coverage'] / results['avg_set_size'] 
            model_logger.info(f"Efficiency: {efficiency:.4f}")
            results['efficiency'] = efficiency
        
        model_logger.info(f"=" * 50)
        
        return {
            'model': model_name,
            'dataset': dataset_name,
            'results': results,
            'success': True
        }
    
    except Exception as e:
        model_logger.error(f"Error training {model_name} on {dataset_name}: {str(e)}")
        model_logger.error(traceback.format_exc())
        return {
            'model': model_name,
            'dataset': dataset_name,
            'success': False,
            'error': str(e)
        }

def plot_comparison(all_results, plot_dir, dataset_name):
    """Generate comparison plots for all models"""
    # Check if we have any successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    if not successful_results:
        logging.warning("No successful results to plot")
        return None
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame([
        {
            'model': result['model'],
            'avg_set_size': result['results'].get('avg_set_size', float('nan')),
            'set_size': result['results'].get('avg_set_size', float('nan')),  # Add both names for compatibility
            'coverage': result['results'].get('coverage', float('nan')),
            'auroc': result['results'].get('auroc', float('nan')),
            'efficiency': result['results'].get('efficiency', float('nan')),
            'margin': result['results'].get('margin', float('nan'))
        }
        for result in successful_results
    ])
    
    # Check if we have any valid data to plot
    if df.empty:
        logging.warning("DataFrame is empty after filtering successful results")
        return None
    
    # Sort by set size for better visualization - handle case when column might be missing
    sort_column = 'avg_set_size' if 'avg_set_size' in df.columns else 'model'
    df_sorted = df.sort_values(sort_column)
    
    # Plot comparison metrics - only plot metrics that are actually present
    metrics = {}
    if 'avg_set_size' in df.columns and not df['avg_set_size'].isna().all():
        metrics['avg_set_size'] = 'Average Set Size'
    if 'set_size' in df.columns and not df['set_size'].isna().all() and 'avg_set_size' not in metrics:
        metrics['set_size'] = 'Average Set Size'
    if 'coverage' in df.columns and not df['coverage'].isna().all():
        metrics['coverage'] = 'Coverage (target 90%)'
    if 'auroc' in df.columns and not df['auroc'].isna().all():
        metrics['auroc'] = 'AUROC Score'
    if 'efficiency' in df.columns and not df['efficiency'].isna().all():
        metrics['efficiency'] = 'Efficiency'
    if 'margin' in df.columns and not df['margin'].isna().all():
        metrics['margin'] = 'Margin'
    
    if not metrics:
        logging.warning("No valid metrics to plot")
        return df_sorted
    
    for metric, title in metrics.items():
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_sorted['model'], df_sorted[metric])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f'{title} Comparison Across Models - {dataset_name}')
        plt.xlabel('VLM Model')
        plt.ylabel(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(os.path.join(plot_dir, f'comparison_{metric}_{dataset_name}.png'), dpi=300)
        plt.close()
    
    # Return the sorted DataFrame for summary statistics
    return df_sorted

def generate_summary(all_results, dataset_name):
    """Generate summary of best performing models"""
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if not successful_results:
        return "No successful training runs to summarize."
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            'model': result['model'],
            'set_size': result['results'].get('avg_set_size', float('nan')),
            'coverage': result['results'].get('coverage', float('nan')),
            'auroc': result['results'].get('auroc', float('nan')),
            'auarc': result['results'].get('auarc', float('nan')),
            'ece': result['results'].get('ece', float('nan')),
            'efficiency': result['results'].get('efficiency', float('nan')),
        }
        for result in successful_results
    ])
    
    if df.empty:
        return "No valid results to summarize."
        
    # Check if necessary columns exist
    required_columns = ['set_size', 'coverage', 'auroc']
    for col in required_columns:
        if col not in df.columns or df[col].isna().all():
            logging.warning(f"Column {col} is missing or all NaN in summary data")
            df[col] = float('nan')
    
    # Format summary text
    summary = f"SUMMARY FOR DATASET: {dataset_name}\n"
    summary += "=" * 50 + "\n\n"
    
    # Add a section comparing models to target coverage 90%
    target_coverage = 0.9
    summary += f"MODELS EVALUATION (Target Coverage: {target_coverage:.2f})\n"
    summary += "-" * 50 + "\n"
    summary += f"{'Model':<25} | {'Coverage':<10} | {'Set Size':<10} | {'AUROC':<10} | {'Efficiency':<10}\n"
    summary += "-" * 80 + "\n"
    
    # Sort models by how close they are to the target coverage
    if not df['coverage'].isna().all():
        df['coverage_distance'] = abs(df['coverage'] - target_coverage)
        df_sorted_by_coverage = df.sort_values('coverage_distance')
        
        for _, row in df_sorted_by_coverage.iterrows():
            coverage = row.get('coverage', float('nan'))
            set_size = row.get('set_size', float('nan'))
            auroc = row.get('auroc', float('nan'))
            efficiency = row.get('efficiency', float('nan'))
            
            coverage_str = f"{coverage:.4f}" if not np.isnan(coverage) else "N/A"
            set_size_str = f"{set_size:.4f}" if not np.isnan(set_size) else "N/A"
            auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "N/A"
            efficiency_str = f"{efficiency:.4f}" if not np.isnan(efficiency) else "N/A"
            
            summary += f"{row['model']:<25} | {coverage_str:<10} | {set_size_str:<10} | {auroc_str:<10} | {efficiency_str:<10}\n"
    else:
        summary += "No coverage data available for any model.\n"
    
    # Find best models in different categories
    summary += "\n\nBEST MODELS BY CATEGORY\n"
    summary += "-" * 50 + "\n"
    
    try:
        if not df['set_size'].isna().all():
            best_setsize = df.loc[df['set_size'].idxmin()]
            summary += f"1. Minimum Set Size: {best_setsize['model']} "
            summary += f"(Size: {best_setsize['set_size']:.4f}, "
            summary += f"Coverage: {best_setsize['coverage']:.4f if not np.isnan(best_setsize['coverage']) else 'N/A'}, "
            summary += f"AUROC: {best_setsize['auroc']:.4f if not np.isnan(best_setsize['auroc']) else 'N/A'})\n"
    except Exception as e:
        logging.warning(f"Error finding minimum set size model: {str(e)}")
        
    try:
        if not df['coverage'].isna().all():
            best_coverage = df.loc[(df['coverage'] - target_coverage).abs().idxmin()]
            summary += f"2. Closest to Target Coverage ({target_coverage:.2f}): {best_coverage['model']} "
            summary += f"(Coverage: {best_coverage['coverage']:.4f}, "
            summary += f"Size: {best_coverage['set_size']:.4f if not np.isnan(best_coverage['set_size']) else 'N/A'}, "
            summary += f"AUROC: {best_coverage['auroc']:.4f if not np.isnan(best_coverage['auroc']) else 'N/A'})\n"
    except Exception as e:
        logging.warning(f"Error finding best coverage model: {str(e)}")
        
    try:
        if not df['auroc'].isna().all():
            best_auroc = df.loc[df['auroc'].idxmax()]
            summary += f"3. Best AUROC: {best_auroc['model']} "
            summary += f"(AUROC: {best_auroc['auroc']:.4f}, "
            summary += f"Size: {best_auroc['set_size']:.4f if not np.isnan(best_auroc['set_size']) else 'N/A'}, "
            summary += f"Coverage: {best_auroc['coverage']:.4f if not np.isnan(best_auroc['coverage']) else 'N/A'})\n"
    except Exception as e:
        logging.warning(f"Error finding best AUROC model: {str(e)}")
        
    try:
        if not df['efficiency'].isna().all():
            best_efficiency = df.loc[df['efficiency'].idxmax()]
            summary += f"4. Best Efficiency: {best_efficiency['model']} "
            summary += f"(Efficiency: {best_efficiency['efficiency']:.4f}, "
            summary += f"Size: {best_efficiency['set_size']:.4f if not np.isnan(best_efficiency['set_size']) else 'N/A'}, "
            summary += f"Coverage: {best_efficiency['coverage']:.4f if not np.isnan(best_efficiency['coverage']) else 'N/A'})\n"
    except Exception as e:
        logging.warning(f"Error finding best efficiency model: {str(e)}")
    
    # Add data for failed models
    failed_results = [r for r in all_results if not r.get('success', False)]
    if failed_results:
        summary += "\n\nFAILED MODELS\n"
        summary += "-" * 50 + "\n"
        for result in failed_results:
            summary += f"{result['model']} - Error: {result.get('error', 'Unknown error')}\n"
    
    return summary

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train scoring function for VLM logits')
    parser.add_argument('--config', type=str, default='src/config/vlm.yaml',
                      help='Path to config file (e.g., src/config/vlm.yaml)')
    parser.add_argument('--model', type=str, default=None,
                      help='VLM model to use (default: all models)')
    parser.add_argument('--dataset', type=str, default='ai2d',
                      help='Dataset to use (default: ai2d)')
    parser.add_argument('--gpu', type=int, default=None,
                      help='Single GPU ID to use (overrides config)')
    parser.add_argument('--gpus', type=str, default=None,
                      help='Comma-separated list of GPU IDs to use for parallel training')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with more verbose output')
    args = parser.parse_args()
    
    try:
        # Load and setup configuration
        config_manager = ConfigManager(args.config)
        config_manager.setup_paths()
        config_manager.setup_device()
        config = config_manager.config
        
        # Override GPU if specified in command line
        if args.gpu is not None:
            config['device'] = f"cuda:{args.gpu}"
            # Also set CUDA_VISIBLE_DEVICES env var
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
        # Setup logging
        log_file = setup_logging(config, args.model, args.dataset)
        
        # Enable debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.debug("Debug logging enabled")
        
        # Print environment information
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logging.info(f"CUDA available: Yes, version {torch.version.cuda}")
            logging.info(f"CUDA devices: {torch.cuda.device_count()}")
            device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            logging.info(f"CUDA device names: {device_names}")
            logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            logging.info("CUDA not available, using CPU")
        
        logging.info(f"Using config: {args.config}")
        logging.info(f"Using device: {config['device']}")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Determine models to train
        models_to_train = [args.model] if args.model else config['dataset']['models']
        dataset_name = args.dataset
        
        logging.info(f"Training scoring functions for models: {models_to_train}")
        logging.info(f"Using dataset: {dataset_name}")
        
        # Parse GPU IDs for parallel training
        gpu_ids = None
        if args.gpus:
            gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
            logging.info(f"Using GPUs: {gpu_ids}")
        
        # Train all models (in parallel if GPU IDs are provided)
        all_results = []
        
        try:
            if gpu_ids and len(gpu_ids) > 1:
                # Print warning about multiprocessing with CUDA
                logging.warning("Using multiprocessing with CUDA can sometimes cause issues.")
                logging.warning("If you encounter problems, try running without the --gpus option.")
                
                # Parallel training using multiple GPUs
                with concurrent.futures.ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
                    future_to_model = {
                        executor.submit(train_model, model_name, dataset_name, config.copy(), gpu_id): model_name
                        for model_name, gpu_id in zip(models_to_train, gpu_ids * (len(models_to_train) // len(gpu_ids) + 1))[:len(models_to_train)]
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_model):
                        model_name = future_to_model[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                            logging.info(f"Completed training for {model_name}")
                        except Exception as e:
                            logging.error(f"Training failed for {model_name}: {str(e)}")
                            # Add failure entry
                            all_results.append({
                                'model': model_name,
                                'dataset': dataset_name,
                                'success': False,
                                'error': str(e)
                            })
            else:
                # Sequential training on single GPU
                for model_name in models_to_train:
                    try:
                        # Each training gets the GPU specified in config or command line
                        gpu_id = args.gpu  # This might be None if not specified
                        logging.info(f"Starting training for {model_name} on GPU {gpu_id or config.get('device', 'from config')}")
                        
                        result = train_model(model_name, dataset_name, config.copy(), gpu_id)
                        all_results.append(result)
                        logging.info(f"Completed training for {model_name}")
                    except Exception as e:
                        logging.error(f"Training failed for {model_name}: {str(e)}")
                        traceback.print_exc()
                        # Add failure entry
                        all_results.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'success': False,
                            'error': str(e)
                        })
        except Exception as e:
            logging.error(f"Error in training process: {str(e)}")
            traceback.print_exc()
        
        # Make sure we have at least empty results
        if not all_results:
            logging.warning("No results collected, creating empty results for each model")
            all_results = [
                {
                    'model': model_name,
                    'dataset': dataset_name,
                    'success': False,
                    'error': "Training not attempted"
                }
                for model_name in models_to_train
            ]
        
        # Generate comparison plots
        plot_dir = config.get('plot_dir', 'plots/vlm')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Only attempt plotting if we have some results
        df_sorted = None
        if all_results:
            try:
                df_sorted = plot_comparison(all_results, plot_dir, dataset_name)
            except Exception as e:
                logging.error(f"Error generating comparison plots: {str(e)}")
                traceback.print_exc()
        
        # Generate and save summary
        try:
            summary = generate_summary(all_results, dataset_name)
            logging.info("\n" + summary)
            
            # Save summary to file
            summary_file = os.path.join(plot_dir, f'summary_{dataset_name}.txt')
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            logging.info(f"Summary saved to {summary_file}")
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            traceback.print_exc()
        
        logging.info("All training completed!")
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main() 