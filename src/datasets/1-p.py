import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from tqdm import tqdm
import copy
import argparse
import yaml
import json
import sys
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns

class OneMinus_P_Scorer:
    """
    Implementation of the 1-p scoring function for conformal prediction.
    This class handles calibration and prediction using the 1-p scoring approach.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config['device'] >= 0 else "cpu")
        self.target_coverage = config['target_coverage']
        self.tau = None  # Will be set during calibration
        
        # Initialize dataset based on configuration
        dataset_name = config['dataset']['name']
        logging.info(f"Initializing 1-p scorer for {dataset_name} dataset")
        
        # Create a copy of the config to modify for dataset-specific settings
        dataset_config = copy.deepcopy(config)
        
        # Apply dataset-specific model path
        if 'model_paths' in config and dataset_name in config['model_paths']:
            dataset_config['model']['pretrained_path'] = config['model_paths'][dataset_name]
            logging.info(f"Using model path: {dataset_config['model']['pretrained_path']}")
        
        # Apply model-specific configurations
        if 'model_configs' in config and dataset_name in config['model_configs']:
            for key, value in config['model_configs'][dataset_name].items():
                dataset_config['model'][key] = value
            logging.info(f"Applied {dataset_name}-specific model configuration")
        
        # Apply dataset-specific data directory if available
        if dataset_name in config and 'data_dir' in config[dataset_name]:
            dataset_config['data_dir'] = config[dataset_name]['data_dir']
            logging.info(f"Using dataset-specific data directory: {dataset_config['data_dir']}")
        
        # Fix the device configuration to use a proper device object instead of an integer
        dataset_config['device'] = self.device
        
        # Patch torch.load to handle integer device IDs
        original_torch_load = torch.load
        
        def patched_torch_load(path, map_location=None, **kwargs):
            if isinstance(map_location, int):
                # Convert integer to proper device string
                map_location = f'cuda:{map_location}' if torch.cuda.is_available() else 'cpu'
            return original_torch_load(path, map_location=map_location, **kwargs)
        
        # Replace torch.load with our patched version
        torch.load = patched_torch_load
        
        # Import the appropriate dataset class directly
        if dataset_name == 'cifar10':
            from src.datasets.cifar10 import Dataset
        elif dataset_name == 'cifar100':
            from src.datasets.cifar100 import Dataset
        elif dataset_name == 'imagenet':
            from src.datasets.imagenet import Dataset
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        # Initialize the dataset with the modified config
        self.dataset = Dataset(dataset_config)
        self.dataset.setup()
        
        # Get the model from the dataset
        try:
            self.model = self.dataset.get_model()
            logging.info("Successfully loaded model")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
    def calibrate(self):
        """
        Calibrate the model using the calibration dataset to determine tau.
        """
        logging.info("Calibrating using 1-p scoring function...")
        self.model.eval()
        nonconformity_scores = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.dataset.cal_loader, desc="Calibration"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                
                # For each sample, get the probability of the true class
                for i in range(len(targets)):
                    true_class_prob = probabilities[i, targets[i]].item()
                    # 1-p scoring function
                    nonconformity_score = 1 - true_class_prob
                    nonconformity_scores.append(nonconformity_score)
        
        # Calculate tau as the percentile corresponding to target coverage
        # For 90% coverage, we use the 90th percentile
        self.tau = np.percentile(nonconformity_scores, 100 * self.target_coverage)
        logging.info(f"Calibration complete. Tau value: {self.tau:.4f}")
        
        # Store nonconformity scores for plotting
        self.nonconformity_scores = nonconformity_scores
        
        # Plot the scoring function
        self.plot_scoring_function()
        
        return self.tau
    
    def evaluate(self):
        """
        Evaluate the model on the test set and compute metrics.
        """
        if self.tau is None:
            logging.warning("Tau not calibrated. Running calibration first.")
            self.calibrate()
        
        self.model.eval()
        total_samples = 0
        covered_samples = 0
        set_sizes = []
        
        # For score distribution plot
        true_class_scores = []
        false_class_scores = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.dataset.test_loader, desc="Evaluation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                
                for i in range(len(targets)):
                    true_class = targets[i].item()
                    
                    # Collect scores for true and false classes
                    for class_idx in range(probabilities.size(1)):
                        class_prob = probabilities[i, class_idx].item()
                        # 1-p scoring function
                        score = 1 - class_prob
                        
                        if class_idx == true_class:
                            true_class_scores.append(score)
                        else:
                            false_class_scores.append(score)
                    
                    # Create prediction set
                    prediction_set = []
                    for class_idx in range(probabilities.size(1)):
                        class_prob = probabilities[i, class_idx].item()
                        # 1-p scoring function
                        score = 1 - class_prob
                        if score <= self.tau:
                            prediction_set.append(class_idx)
                    
                    # Check if true class is in the prediction set
                    is_covered = true_class in prediction_set
                    covered_samples += int(is_covered)
                    set_sizes.append(len(prediction_set))
                    total_samples += 1
        
        # Calculate metrics
        empirical_coverage = covered_samples / total_samples
        average_set_size = np.mean(set_sizes)
        median_set_size = np.median(set_sizes)
        
        # Debug: print set size distribution
        set_size_counts = np.bincount(set_sizes)
        logging.info(f"Debug - Set size distribution:")
        for size, count in enumerate(set_size_counts):
            if count > 0:
                logging.info(f"  Size {size}: {count} samples ({count/total_samples*100:.2f}%)")
        
        results = {
            "dataset": self.config['dataset']['name'],
            "tau": self.tau,
            "target_coverage": self.target_coverage,
            "empirical_coverage": empirical_coverage,
            "average_set_size": average_set_size,
            "median_set_size": median_set_size,
            "set_size_std": np.std(set_sizes),
            "set_size_min": np.min(set_sizes),
            "set_size_max": np.max(set_sizes),
        }
        
        logging.info(f"Evaluation Results for {self.config['dataset']['name']}:")
        logging.info(f"  Target Coverage: {self.target_coverage:.4f}")
        logging.info(f"  Empirical Coverage: {empirical_coverage:.4f}")
        logging.info(f"  Average Set Size: {average_set_size:.4f}")
        logging.info(f"  Median Set Size: {median_set_size:.4f}")
        
        # Plot score distributions
        self.plot_score_distributions(true_class_scores, false_class_scores)
        
        return results
    
    def plot_scoring_function(self):
        """
        Plot the 1-p scoring function based on calibration data.
        This shows the distribution of nonconformity scores and the threshold tau.
        """
        if not hasattr(self, 'nonconformity_scores'):
            logging.warning("No nonconformity scores available. Run calibration first.")
            return
        
        dataset_name = self.config['dataset']['name']
        plot_dir = os.path.join(self.config.get('plot_dir', 'plots/1-p'))
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of nonconformity scores
        sns.histplot(self.nonconformity_scores, kde=True, bins=50)
        
        # Add vertical line for tau
        plt.axvline(x=self.tau, color='r', linestyle='--', 
                   label=f'τ = {self.tau:.4f} (target coverage: {self.target_coverage:.2f})')
        
        # Add labels and title
        plt.xlabel('Nonconformity Score (1-p)')
        plt.ylabel('Frequency')
        plt.title(f'1-p Scoring Function Distribution - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(plot_dir, f'{dataset_name}_scoring_function.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Scoring function plot saved to {plot_path}")
    
    def plot_score_distributions(self, true_class_scores, false_class_scores):
        """
        Plot the distribution of non-conformity scores for true and false classes.
        This helps visualize how well the scoring function separates correct from incorrect predictions.
        
        Args:
            true_class_scores: List of non-conformity scores for true classes
            false_class_scores: List of non-conformity scores for false classes
        """
        dataset_name = self.config['dataset']['name']
        plot_dir = os.path.join(self.config.get('plot_dir', 'plots/1-p'))
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Set style similar to the provided image
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Plot distributions with lines instead of filled areas
        sns.kdeplot(true_class_scores, label='True Class Scores', color='blue')
        sns.kdeplot(false_class_scores, label='False Class Scores', color='orange')
        
        # Add vertical line for tau
        plt.axvline(x=self.tau, color='red', linestyle='--', 
                   label='Tau Threshold')
        
        # Add labels and title
        plt.xlabel('Non-Conformity Score')
        plt.ylabel('Density/Frequency')
        plt.title('Distribution of Conformity Scores')
        
        # Add legend with custom position
        plt.legend(loc='upper left')
        
        # Set axis limits based on data
        min_score = min(min(true_class_scores) if true_class_scores else 0, 
                        min(false_class_scores) if false_class_scores else 0)
        max_score = max(max(true_class_scores) if true_class_scores else 1, 
                        max(false_class_scores) if false_class_scores else 1)
        plt.xlim(min_score - 0.2, max_score + 0.2)
        
        # Save the plot
        plot_path = os.path.join(plot_dir, f'{dataset_name}_score_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Score distribution plot saved to {plot_path}")

def run_dataset(config, dataset_name, scoring_name="1-p"):
    """
    Run the scoring function evaluation for a specific dataset.
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset to evaluate
        scoring_name: Name of the scoring function (default: "1-p")
    
    Returns:
        Results dictionary
    """
    # Update config with dataset name
    dataset_config = copy.deepcopy(config)
    dataset_config['dataset']['name'] = dataset_name
    
    # Setup logging for this dataset
    log_dir = os.path.join(config['log_dir'], dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{scoring_name.lower()}_evaluation_{dataset_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add the file handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    try:
        logging.info(f"=== Starting evaluation for {dataset_name} dataset using {scoring_name} scoring function ===")
        
        # For ImageNet, load the dataset-specific configuration
        if dataset_name == 'imagenet':
            # Load ImageNet-specific configuration
            imagenet_config_path = os.path.join(config['base_dir'], 'src', 'config', 'imagenet.yaml')
            if os.path.exists(imagenet_config_path):
                with open(imagenet_config_path, 'r') as f:
                    imagenet_config = yaml.safe_load(f)
                    # Update data_dir from imagenet.yaml
                    if 'data_dir' in imagenet_config:
                        dataset_config['data_dir'] = imagenet_config['data_dir']
                        logging.info(f"Using data_dir from imagenet.yaml: {dataset_config['data_dir']}")
                    # Update batch_size from imagenet.yaml
                    if 'batch_size' in imagenet_config:
                        dataset_config['batch_size'] = imagenet_config['batch_size']
                        logging.info(f"Using batch_size from imagenet.yaml: {dataset_config['batch_size']}")
        
        # Initialize and run the scorer
        try:
            scorer = OneMinus_P_Scorer(dataset_config)
            scorer.calibrate()
            results = scorer.evaluate()
            
            # Convert NumPy types to Python native types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, np.integer):
                    json_results[key] = int(value)
                elif isinstance(value, np.floating):
                    json_results[key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = value
            
            # Save results
            results_file = os.path.join(log_dir, f'{scoring_name.lower()}_results_{dataset_name}.json')
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=4)
            
            logging.info(f"Results saved to {results_file}")
            logging.info(f"=== Completed evaluation for {dataset_name} dataset ===\n")
            
            return json_results
        except ValueError as e:
            if "num_samples" in str(e):
                logging.error(f"Error running {scoring_name} evaluation for {dataset_name}: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                return {"error": str(e)}
            else:
                raise
    except Exception as e:
        logging.error(f"Error running {scoring_name} evaluation for {dataset_name}: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}
    finally:
        # Remove the file handler to avoid duplicate log entries
        root_logger.removeHandler(file_handler)

def main():
    """
    Main function to run the scoring function evaluation.
    """
    parser = argparse.ArgumentParser(description='Run scoring function evaluation')
    parser.add_argument('--config', type=str, default='src/config/1-p.yaml', help='Path to config file')
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all', 'cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--scoring', type=str, default='1-p',
                        choices=['1-p', 'APS'],
                        help='Scoring function to use (default: 1-p)')
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Load configuration based on scoring function
    if args.scoring == 'APS':
        config_path = args.config.replace('1-p.yaml', 'aps.yaml')
        if not os.path.exists(config_path):
            config_path = 'src/config/aps.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Import APS scorer
        from src.datasets.aps import APS_Scorer as Scorer
        from src.datasets.aps import run_dataset as run_dataset_func
        
        logging.info(f"Using APS scoring function with config from {config_path}")
    else:  # Default to 1-p
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Use 1-p scorer (default)
        Scorer = OneMinus_P_Scorer
        run_dataset_func = run_dataset
        
        logging.info(f"Using 1-p scoring function with config from {args.config}")
    
    # Create summary directory
    summary_dir = os.path.join(config['log_dir'], 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Determine which datasets to run
    if args.dataset == 'all':
        datasets = ['cifar10', 'cifar100', 'imagenet']
        logging.info(f"Running evaluation for all datasets: CIFAR-10, CIFAR-100, and ImageNet")
    else:
        datasets = [args.dataset]
        logging.info(f"Running evaluation for {args.dataset} dataset")
    
    # Run evaluation for each dataset
    all_results = {}
    for dataset in datasets:
        try:
            results = run_dataset_func(config, dataset, args.scoring)
            all_results[dataset] = results
        except Exception as e:
            logging.error(f"Failed to run evaluation for {dataset}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            all_results[dataset] = {"error": str(e)}
    
    # Save summary of all results
    summary_file = os.path.join(summary_dir, f'{args.scoring.lower()}_summary.json')
    
    # Convert NumPy types to Python native types for JSON serialization
    json_results = {}
    for dataset, results in all_results.items():
        json_results[dataset] = {}
        for key, value in results.items():
            if isinstance(value, np.integer):
                json_results[dataset][key] = int(value)
            elif isinstance(value, np.floating):
                json_results[dataset][key] = float(value)
            elif isinstance(value, np.ndarray):
                json_results[dataset][key] = value.tolist()
            else:
                json_results[dataset][key] = value
    
    with open(summary_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    logging.info(f"Summary results saved to {summary_file}")
    
    # Print summary table
    logging.info(f"\n=== {args.scoring} Scoring Function Evaluation Summary ===")
    logging.info(f"{'Dataset':<10} | {'Coverage':<10} | {'Avg Set Size':<15} | {'Median Set Size':<15}")
    logging.info("-" * 60)
    
    for dataset, results in all_results.items():
        if "error" in results:
            logging.info(f"{dataset:<10} | {'ERROR':<10} | {'N/A':<15} | {'N/A':<15}")
        else:
            coverage = results.get("empirical_coverage", "N/A")
            avg_size = results.get("average_set_size", "N/A")
            median_size = results.get("median_set_size", "N/A")
            
            if isinstance(coverage, float):
                coverage = f"{coverage:.4f}"
            if isinstance(avg_size, float):
                avg_size = f"{avg_size:.4f}"
            if isinstance(median_size, float):
                median_size = f"{median_size:.4f}"
                
            logging.info(f"{dataset:<10} | {coverage:<10} | {avg_size:<15} | {median_size:<15}")
    
    return all_results

if __name__ == "__main__":
    main() 