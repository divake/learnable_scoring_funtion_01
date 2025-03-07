import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from tqdm import tqdm

from src.datasets.cifar10 import Dataset as CIFAR10Dataset

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
        
        # Initialize dataset
        if config['dataset']['name'] == 'cifar10':
            self.dataset = CIFAR10Dataset(config)
            self.dataset.setup()
        else:
            raise ValueError(f"Dataset {config['dataset']['name']} not supported yet")
        
        # Load the base model
        self.model = self.dataset.get_model()
        self.model.eval()  # Set to evaluation mode
        
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
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.dataset.test_loader, desc="Evaluation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probabilities = F.softmax(outputs, dim=1)
                
                for i in range(len(targets)):
                    prediction_set = []
                    for class_idx in range(probabilities.size(1)):
                        # 1-p scoring function for each potential class
                        nonconformity_score = 1 - probabilities[i, class_idx].item()
                        if nonconformity_score <= self.tau:
                            prediction_set.append(class_idx)
                    
                    # Check if true class is in the prediction set
                    is_covered = targets[i].item() in prediction_set
                    covered_samples += int(is_covered)
                    set_sizes.append(len(prediction_set))
                    total_samples += 1
        
        # Calculate metrics
        empirical_coverage = covered_samples / total_samples
        average_set_size = np.mean(set_sizes)
        median_set_size = np.median(set_sizes)
        
        results = {
            "tau": self.tau,
            "target_coverage": self.target_coverage,
            "empirical_coverage": empirical_coverage,
            "average_set_size": average_set_size,
            "median_set_size": median_set_size,
            "set_size_std": np.std(set_sizes),
            "set_size_min": np.min(set_sizes),
            "set_size_max": np.max(set_sizes),
        }
        
        logging.info(f"Evaluation Results:")
        logging.info(f"  Target Coverage: {self.target_coverage:.4f}")
        logging.info(f"  Empirical Coverage: {empirical_coverage:.4f}")
        logging.info(f"  Average Set Size: {average_set_size:.4f}")
        logging.info(f"  Median Set Size: {median_set_size:.4f}")
        
        return results

def main(config):
    """
    Main function to run the 1-p scoring evaluation.
    """
    # Setup logging
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '1-p_evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    # Initialize and run the 1-p scorer
    scorer = OneMinus_P_Scorer(config)
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
    import json
    with open(os.path.join(log_dir, '1-p_results.json'), 'w') as f:
        json.dump(json_results, f, indent=4)
    
    return results

if __name__ == "__main__":
    import yaml
    import argparse
    
    parser = argparse.ArgumentParser(description='Run 1-p scoring evaluation')
    parser.add_argument('--config', type=str, default='src/config/1-p.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config) 