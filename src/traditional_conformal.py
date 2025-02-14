import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import logging
import os
from cifar_split import setup_cifar10
from utils.visualization import (plot_training_curves, plot_score_distributions,
                               plot_set_size_distribution)

class TraditionalConformalPredictor:
    def __init__(self, base_model, device, coverage_target=0.9):
        """
        Traditional conformal predictor using 1-softmax as non-conformity score
        """
        self.base_model = base_model
        self.device = device
        self.coverage_target = coverage_target
        self.tau = None
        
    def calibrate(self, cal_loader):
        """Compute tau using calibration set"""
        self.base_model.eval()
        all_scores = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(cal_loader, desc="Calibrating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get softmax probabilities
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                # Get non-conformity scores (1 - probability) for true class
                true_probs = probs[torch.arange(len(targets)), targets]
                scores = 1 - true_probs
                
                all_scores.extend(scores.cpu().numpy())
        
        # Compute tau as quantile
        all_scores = np.array(all_scores)
        self.tau = np.quantile(all_scores, self.coverage_target)
        return self.tau
    
    def evaluate(self, loader):
        """Evaluate on test set"""
        if self.tau is None:
            raise ValueError("Must calibrate first!")
            
        self.base_model.eval()
        all_set_sizes = []
        coverages = []
        true_scores = []
        false_scores = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get softmax probabilities
                logits = self.base_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                # Compute non-conformity scores for all classes
                scores = 1 - probs
                
                # Generate prediction sets
                pred_sets = scores <= self.tau
                set_sizes = pred_sets.sum(dim=1).cpu().numpy()
                
                # Check coverage
                covered = pred_sets[torch.arange(len(targets)), targets]
                coverages.extend(covered.cpu().numpy())
                
                # Collect scores for distribution
                true_class_scores = scores[torch.arange(len(targets)), targets].cpu().numpy()
                true_scores.extend(true_class_scores)
                
                mask = torch.ones_like(scores, dtype=bool)
                mask[torch.arange(len(targets)), targets] = False
                false_class_scores = scores[mask].cpu().numpy()
                false_scores.extend(false_class_scores)
                
                all_set_sizes.extend(set_sizes)
        
        results = {
            'coverage': np.mean(coverages),
            'avg_set_size': np.mean(all_set_sizes),
            'set_sizes': np.array(all_set_sizes),
            'true_scores': np.array(true_scores),
            'false_scores': np.array(false_scores),
            'tau': self.tau
        }
        
        return results

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = '/ssd_4TB/divake/vision_cp/learnable_scoring_funtion_01'
    plot_dir = os.path.join(base_path, 'plots_traditional')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load only calibration and test data
    _, cal_loader, test_loader, _, _, _ = setup_cifar10(batch_size=128)
    print("Data loaded successfully")
    
    # Load pretrained ResNet model
    base_model = models.resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 10)
    base_model.load_state_dict(torch.load(os.path.join(base_path, 'models/resnet18_cifar10_best.pth')))
    base_model = base_model.to(device)
    base_model.eval()
    print("Model loaded successfully")
    
    # Create traditional conformal predictor
    predictor = TraditionalConformalPredictor(base_model, device)
    
    # Calibrate
    tau = predictor.calibrate(cal_loader)
    print(f"\nCalibrated tau: {tau:.4f}")
    
    # Evaluate
    results = predictor.evaluate(test_loader)
    print("\nTest Set Results:")
    print(f"Coverage: {results['coverage']:.4f}")
    print(f"Average Set Size: {results['avg_set_size']:.4f}")
    
    # Plot distributions
    plot_score_distributions(
        results['true_scores'],
        results['false_scores'],
        results['tau'],
        plot_dir
    )
    
    plot_set_size_distribution(
        results['set_sizes'],
        plot_dir
    )
    
    print("\nPlots saved in:", plot_dir)

if __name__ == "__main__":
    main()