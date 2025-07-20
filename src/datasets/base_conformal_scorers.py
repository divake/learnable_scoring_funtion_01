#!/usr/bin/env python3
"""
Unified implementation of conformal prediction scoring functions.

This module consolidates various conformal prediction scoring functions,
providing a single interface for calibration, evaluation, and visualization.
"""

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
from abc import ABC, abstractmethod
from datetime import datetime
import random
import glob
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
from pathlib import Path
import concurrent.futures
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from src.core.advanced_metrics import calculate_auroc, plot_roc_curve

# ==================== Registry System ====================

SCORER_REGISTRY = {}

def register_scorer(name):
    """
    Register a scoring function class with the given name.
    
    Args:
        name: Name of the scoring function
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        if name in SCORER_REGISTRY:
            logging.warning(f"Scoring function {name} already registered. Overwriting.")
        SCORER_REGISTRY[name] = cls
        return cls
    return decorator

def get_scorer(name, config, dataset=None):
    """
    Get a scoring function by name.
    
    Args:
        name: Name of the scoring function
        config: Configuration dictionary
        dataset: Optional dataset object
        
    Returns:
        Initialized scorer object
        
    Raises:
        ValueError: If the scorer is not registered
    """
    if name not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scoring function: {name}")
    if dataset is None:
        return SCORER_REGISTRY[name](config)
    return SCORER_REGISTRY[name](config, dataset)

def list_scorers():
    """
    List all registered scoring functions.
    
    Returns:
        Dictionary of registered scoring functions
    """
    return SCORER_REGISTRY.copy()

# ==================== Utility Functions ====================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def setup_logging(config: Dict[str, Any]) -> None:
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_format = config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Only setup console logging here - file logging will be handled per dataset
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logging.info("Logging initialized for console output")

def get_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Get dataset-specific configuration.
    
    Args:
        config: Base configuration
        dataset_name: Name of the dataset
        
    Returns:
        Dataset-specific configuration
    """
    # Create a copy of the config to modify
    dataset_config = copy.deepcopy(config)
    
    # Update dataset name
    if 'dataset' in dataset_config:
        dataset_config['dataset']['name'] = dataset_name
    else:
        dataset_config['dataset'] = {'name': dataset_name}
    
    # Apply dataset-specific overrides if available
    if dataset_name in config:
        dataset_overrides = config[dataset_name]
        if not isinstance(dataset_config['dataset'], dict):
            dataset_config['dataset'] = {}
        for key, value in dataset_overrides.items():
            dataset_config['dataset'][key] = value
    
    # Apply model path if available
    if 'model_paths' in config and dataset_name in config['model_paths']:
        if 'model' not in dataset_config:
            dataset_config['model'] = {}
        dataset_config['model']['pretrained_path'] = config['model_paths'][dataset_name]
    
    # Apply model-specific configurations
    if 'model_configs' in config and dataset_name in config['model_configs']:
        if 'model' not in dataset_config:
            dataset_config['model'] = {}
        for key, value in config['model_configs'][dataset_name].items():
            dataset_config['model'][key] = value
    
    return dataset_config

# ==================== Base Scorer Class ====================

class BaseScorer(ABC):
    """
    Base class for all conformal prediction scoring functions.
    
    This abstract class defines the interface that all scoring functions must implement
    and provides common functionality for calibration, evaluation, and visualization.
    """
    
    def __init__(self, config: Dict[str, Any], dataset=None):
        """
        Initialize the scorer with configuration.
        
        Args:
            config: Configuration dictionary
            dataset: Optional pre-initialized dataset
        """
        self.config = config
        self.device = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() and config['device'] >= 0 else "cpu")
        self.target_coverage = config.get('target_coverage', 0.9)
        self.tau = None  # Will be set during calibration
        self.nonconformity_scores = None  # Will be set during calibration
        
        # Apply scorer-specific configurations if available
        scorer_name = self.__class__.__name__
        if 'scoring_functions' in config and scorer_name in config['scoring_functions']:
            for key, value in config['scoring_functions'][scorer_name].items():
                setattr(self, key, value)
        
        # Set up unified directory structure for results
        dataset_name = config['dataset']['name']
        self.base_output_dir = os.path.join(config.get('base_dir', '.'), 'results_static_conformal', dataset_name)
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # Set plot directory within the unified structure
        self.plot_dir = self.base_output_dir
        
        # Set up log directory within the unified structure
        self.log_dir = self.base_output_dir
        
        # Initialize dataset if not provided
        if dataset is None:
            dataset_name = config['dataset']['name']
            logging.info(f"Initializing {self.__class__.__name__} scorer for {dataset_name} dataset")
            
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
            elif dataset_name == 'vlm':
                from src.datasets.vlm import Dataset
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
        else:
            self.dataset = dataset
            self.model = dataset.get_model()
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Check if cached outputs are available
        self.use_cache = False
        self.cached_data = {}
        self._load_cached_outputs()
    
    def _load_cached_outputs(self):
        """Load cached model outputs if available."""
        # Determine model type from architecture name
        model_arch = self.config.get('model', {}).get('architecture', 'resnet18')
        if 'vit' in model_arch.lower() or 'vision' in model_arch.lower():
            model_type = 'VisionTransformer'
        else:
            model_type = 'ResNet'
        
        cache_dir = Path(self.config['base_dir']) / 'cache' / self.config['dataset']['name'] / model_type
        
        if cache_dir.exists():
            try:
                # Load calibration data
                cal_probs_path = cache_dir / 'cal_probs.pt'
                cal_targets_path = cache_dir / 'cal_targets.pt'
                if cal_probs_path.exists() and cal_targets_path.exists():
                    self.cached_data['cal_probs'] = torch.load(cal_probs_path)
                    self.cached_data['cal_targets'] = torch.load(cal_targets_path)
                    logging.info(f"Loaded cached calibration outputs from {cache_dir}")
                    
                # Load test data
                test_probs_path = cache_dir / 'test_probs.pt'
                test_targets_path = cache_dir / 'test_targets.pt'
                if test_probs_path.exists() and test_targets_path.exists():
                    self.cached_data['test_probs'] = torch.load(test_probs_path)
                    self.cached_data['test_targets'] = torch.load(test_targets_path)
                    logging.info(f"Loaded cached test outputs from {cache_dir}")
                    
                if self.cached_data:
                    self.use_cache = True
                    logging.info("Using cached model outputs for fair comparison")
            except Exception as e:
                logging.warning(f"Failed to load cached outputs: {e}. Will use direct model inference.")
                self.use_cache = False
                self.cached_data = {}
        else:
            logging.info(f"Cache directory {cache_dir} not found. Will use direct model inference.")
    
    @abstractmethod
    def compute_nonconformity_score(self, probabilities: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Compute nonconformity scores based on model outputs and true targets.
        
        This method must be implemented by each specific scoring function.
        
        Args:
            probabilities: Softmax probabilities of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            List of nonconformity scores, one for each sample in the batch
        """
        pass
    
    def lower_is_better(self) -> bool:
        """
        Indicate whether lower scores are better (more conforming) for this scoring function.
        
        Returns:
            True if lower scores indicate more conformity (default for most scorers),
            False if higher scores indicate more conformity.
        """
        # By default, most conformal methods use lower scores to indicate higher conformity
        # Subclasses can override this if they have the opposite behavior
        return True
    
    def calibrate(self) -> float:
        """
        Calibrate the model using the calibration dataset to determine tau.
        
        Returns:
            The calibrated threshold value tau
        """
        logging.info(f"Calibrating using {self.__class__.__name__} scoring function...")
        nonconformity_scores = []
        
        if self.use_cache and 'cal_probs' in self.cached_data:
            # Use cached outputs
            probabilities = self.cached_data['cal_probs'].to(self.device)
            targets = self.cached_data['cal_targets'].to(self.device)
            
            # Add realistic noise if specified
            noise_std = self.config.get('noise_std', 0.0)
            noise_type = self.config.get('noise_type', 'logit_gaussian')
            
            if noise_std > 0:
                epsilon = 1e-8  # Small value to avoid log(0)
                original_entropy = torch.mean(-torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1))
                
                if noise_type == 'temperature':
                    print(f"\n🌡️ ADDING TEMPERATURE NOISE: std={noise_std} to calibration data")
                    print(f"   Original prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                    
                    # Temperature scaling with noise (more realistic)
                    logits = torch.log(probabilities + epsilon)
                    temp_range = self.config.get('temperature_range', [0.8, 1.2])
                    temperature = 1.0 + (torch.rand(1).item() - 0.5) * 2 * noise_std
                    temperature = torch.clamp(torch.tensor(temperature), temp_range[0], temp_range[1]).item()
                    
                    scaled_logits = logits / temperature
                    probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)
                    
                elif noise_type == 'quantization':
                    print(f"\n⚙️ ADDING QUANTIZATION NOISE: std={noise_std} to calibration data")
                    print(f"   Original prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                    
                    # Simulate quantization noise (edge deployment)
                    logits = torch.log(probabilities + epsilon)
                    quantization_step = noise_std * 0.1  # Realistic quantization
                    quantized_logits = torch.round(logits / quantization_step) * quantization_step
                    quantized_logits += torch.randn_like(logits) * quantization_step * 0.1
                    
                    probabilities = torch.nn.functional.softmax(quantized_logits, dim=1)
                    
                else:  # Default: logit_gaussian
                    print(f"\n🔊 ADDING LOGIT GAUSSIAN NOISE: std={noise_std} to calibration data")
                    print(f"   Original prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                    
                    # Convert probabilities back to logits, add noise, then softmax
                    logits = torch.log(probabilities + epsilon)
                    noise = torch.randn_like(logits) * noise_std
                    noisy_logits = logits + noise
                    probabilities = torch.nn.functional.softmax(noisy_logits, dim=1)
                
                new_entropy = torch.mean(-torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1))
                print(f"   After noise prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                print(f"   Entropy increase: {original_entropy:.4f} → {new_entropy:.4f} (+{new_entropy-original_entropy:.4f})")
            else:
                print(f"\n🔇 No noise applied (noise_std={noise_std})")
            
            # Process in batches to match original behavior
            batch_size = self.dataset.cal_loader.batch_size
            num_samples = len(targets)
            
            for i in tqdm(range(0, num_samples, batch_size), desc="Calibration (cached)"):
                end_idx = min(i + batch_size, num_samples)
                batch_probs = probabilities[i:end_idx]
                batch_targets = targets[i:end_idx]
                
                # Compute nonconformity scores using the specific scoring function
                batch_scores = self.compute_nonconformity_score(batch_probs, batch_targets)
                nonconformity_scores.extend(batch_scores)
        else:
            # Original implementation using direct model inference
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in tqdm(self.dataset.cal_loader, desc="Calibration"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    # Compute nonconformity scores using the specific scoring function
                    batch_scores = self.compute_nonconformity_score(probabilities, targets)
                    nonconformity_scores.extend(batch_scores)
        
        # Calculate tau as the percentile corresponding to target coverage
        # The percentile depends on whether lower or higher scores are better
        if self.lower_is_better():
            # For lower-is-better scoring functions (1-p, APS, LogMargin, Sparsemax)
            # We want tau such that P(score ≤ tau) = target_coverage
            self.tau = np.percentile(nonconformity_scores, 100 * self.target_coverage)
            logging.info(f"Calibration complete. Tau value: {self.tau:.4f} (lower scores are better)")
        else:
            # For higher-is-better scoring functions
            # We want tau such that P(score ≥ tau) = target_coverage
            # Equivalent to P(score < tau) = 1 - target_coverage
            self.tau = np.percentile(nonconformity_scores, 100 * (1 - self.target_coverage))
            logging.info(f"Calibration complete. Tau value: {self.tau:.4f} (higher scores are better)")
        
        # Store nonconformity scores for plotting
        self.nonconformity_scores = nonconformity_scores
        
        # Plot the scoring function
        self.plot_scoring_function()
        
        return self.tau
    
    def plot_scoring_function(self) -> None:
        """
        Plot the scoring function based on calibration data.
        This shows the distribution of nonconformity scores and the threshold tau.
        """
        if not hasattr(self, 'nonconformity_scores') or self.nonconformity_scores is None:
            logging.warning("No nonconformity scores available. Run calibration first.")
            return
        
        dataset_name = self.config['dataset']['name']
        
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of nonconformity scores
        sns.histplot(self.nonconformity_scores, kde=True, bins=50)
        
        # Add vertical line for tau
        plt.axvline(x=self.tau, color='r', linestyle='--', 
                   label=f'τ = {self.tau:.4f} (target coverage: {self.target_coverage:.2f})')
        
        # Add labels and title
        plt.xlabel('Nonconformity Score')
        plt.ylabel('Frequency')
        plt.title(f'{self.__class__.__name__} Scoring Function Distribution - {dataset_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plot_path = os.path.join(self.plot_dir, f'{self.__class__.__name__}_{dataset_name}_scoring_function.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Scoring function plot saved to {plot_path}")
    
    def plot_score_distributions(self, true_class_scores: List[float], false_class_scores: List[float]) -> None:
        """
        Plot the distribution of non-conformity scores for true and false classes.
        This helps visualize how well the scoring function separates correct from incorrect predictions.
        
        Args:
            true_class_scores: List of non-conformity scores for true classes
            false_class_scores: List of non-conformity scores for false classes
        """
        dataset_name = self.config['dataset']['name']
        
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
        plt.legend(loc="upper left")
        
        # Set axis limits based on data (filter out inf values)
        true_finite = [s for s in true_class_scores if not np.isinf(s)]
        false_finite = [s for s in false_class_scores if not np.isinf(s)]
        
        if true_finite or false_finite:
            min_score = min(min(true_finite) if true_finite else 0, 
                            min(false_finite) if false_finite else 0)
            max_score = max(max(true_finite) if true_finite else 1, 
                            max(false_finite) if false_finite else 1)
            plt.xlim(min_score - 0.2, max_score + 0.2)
        
        # Save the plot
        plot_path = os.path.join(self.plot_dir, f'{self.__class__.__name__}_{dataset_name}_score_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Score distribution plot saved to {plot_path}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> None:
        """
        Plot the ROC curve for the model using the imported function.
        
        Args:
            y_true: Ground truth labels
            y_scores: Predicted probabilities
        """
        dataset_name = self.config['dataset']['name']
        
        try:
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use the imported function that now handles multi-class data properly
            plot_roc_curve(y_true, y_scores, title=f'Receiver Operating Characteristic - {dataset_name}', ax=ax)
            
            # Save the plot
            plot_path = os.path.join(self.plot_dir, f'{self.__class__.__name__}_{dataset_name}_roc_curve.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"ROC curve plot saved to {plot_path}")
        except Exception as e:
            logging.warning(f"Could not plot ROC curve: {str(e)}")
            logging.warning("This is non-critical and the evaluation will continue.")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set and compute metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.tau is None:
            logging.warning("Tau not calibrated. Running calibration first.")
            self.calibrate()
        
        total_samples = 0
        covered_samples = 0
        set_sizes = []
        empty_sets = 0
        non_empty_sizes = []
        
        # For score distribution plot
        true_class_scores = []
        false_class_scores = []
        
        # For AUROC calculation
        all_true_labels = []
        all_probabilities = []
        all_sample_scores = []  # Store scores for all classes for each sample
        
        if self.use_cache and 'test_probs' in self.cached_data:
            # Use cached outputs
            probabilities = self.cached_data['test_probs'].to(self.device)
            targets = self.cached_data['test_targets'].to(self.device)
            
            # Add realistic noise if specified (same as calibration)
            noise_std = self.config.get('noise_std', 0.0)
            noise_type = self.config.get('noise_type', 'logit_gaussian')
            
            if noise_std > 0:
                epsilon = 1e-8  # Small value to avoid log(0)
                original_entropy = torch.mean(-torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1))
                
                if noise_type == 'temperature':
                    print(f"\n🌡️ ADDING TEMPERATURE NOISE: std={noise_std} to test data")
                    print(f"   Original prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                    
                    # Temperature scaling with noise (more realistic)
                    logits = torch.log(probabilities + epsilon)
                    temp_range = self.config.get('temperature_range', [0.8, 1.2])
                    temperature = 1.0 + (torch.rand(1).item() - 0.5) * 2 * noise_std
                    temperature = torch.clamp(torch.tensor(temperature), temp_range[0], temp_range[1]).item()
                    
                    scaled_logits = logits / temperature
                    probabilities = torch.nn.functional.softmax(scaled_logits, dim=1)
                    
                elif noise_type == 'quantization':
                    print(f"\n⚙️ ADDING QUANTIZATION NOISE: std={noise_std} to test data")
                    print(f"   Original prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                    
                    # Simulate quantization noise (edge deployment)
                    logits = torch.log(probabilities + epsilon)
                    quantization_step = noise_std * 0.1  # Realistic quantization
                    quantized_logits = torch.round(logits / quantization_step) * quantization_step
                    quantized_logits += torch.randn_like(logits) * quantization_step * 0.1
                    
                    probabilities = torch.nn.functional.softmax(quantized_logits, dim=1)
                    
                else:  # Default: logit_gaussian
                    print(f"\n🔊 ADDING LOGIT GAUSSIAN NOISE: std={noise_std} to test data")
                    print(f"   Original prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                    
                    # Convert probabilities back to logits, add noise, then softmax
                    logits = torch.log(probabilities + epsilon)
                    noise = torch.randn_like(logits) * noise_std
                    noisy_logits = logits + noise
                    probabilities = torch.nn.functional.softmax(noisy_logits, dim=1)
                
                new_entropy = torch.mean(-torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1))
                print(f"   After noise prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
                print(f"   Entropy increase: {original_entropy:.4f} → {new_entropy:.4f} (+{new_entropy-original_entropy:.4f})")
            else:
                print(f"\n🔇 No noise applied to test data (noise_std={noise_std})")
            
            # Process in batches to match original behavior
            batch_size = self.dataset.test_loader.batch_size
            num_samples = len(targets)
            
            for i in tqdm(range(0, num_samples, batch_size), desc="Evaluation (cached)"):
                end_idx = min(i + batch_size, num_samples)
                batch_probs = probabilities[i:end_idx]
                batch_targets = targets[i:end_idx]
                
                # Store true labels and probabilities for AUROC calculation
                all_true_labels.extend(batch_targets.cpu().numpy())
                all_probabilities.append(batch_probs.cpu().numpy())
                
                # Create prediction sets and collect metrics
                # Pass probabilities directly - create_prediction_sets will detect they are probabilities
                prediction_sets, batch_true_scores, batch_false_scores, batch_all_scores = self.create_prediction_sets(batch_probs, batch_targets)
                
                # Store all class scores for each sample
                all_sample_scores.extend(batch_all_scores)
                
                # Update metrics
                for j, pred_set in enumerate(prediction_sets):
                    true_class = batch_targets[j].item()
                    set_size = len(pred_set)
                    is_covered = true_class in pred_set
                    covered_samples += int(is_covered)
                    set_sizes.append(set_size)
                    total_samples += 1
                    
                    # Track empty sets
                    if set_size == 0:
                        empty_sets += 1
                    else:
                        non_empty_sizes.append(set_size)
                
                # Collect scores for distribution plots
                true_class_scores.extend(batch_true_scores)
                false_class_scores.extend(batch_false_scores)
        else:
            # Original implementation using direct model inference
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in tqdm(self.dataset.test_loader, desc="Evaluation"):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                
                # Store true labels and probabilities for AUROC calculation
                all_true_labels.extend(targets.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                
                # Create prediction sets and collect metrics
                prediction_sets, batch_true_scores, batch_false_scores, batch_all_scores = self.create_prediction_sets(outputs, targets)
                
                # Store all class scores for each sample
                all_sample_scores.extend(batch_all_scores)
                
                # Update metrics
                for i, pred_set in enumerate(prediction_sets):
                    true_class = targets[i].item()
                    set_size = len(pred_set)
                    is_covered = true_class in pred_set
                    covered_samples += int(is_covered)
                    set_sizes.append(set_size)
                    total_samples += 1
                    
                    # Track empty sets
                    if set_size == 0:
                        empty_sets += 1
                    else:
                        non_empty_sizes.append(set_size)
                
                # Collect scores for distribution plots
                true_class_scores.extend(batch_true_scores)
                false_class_scores.extend(batch_false_scores)
        
        # Calculate metrics
        empirical_coverage = covered_samples / total_samples
        
        # Calculate average set size excluding empty sets (honest evaluation)
        if non_empty_sizes:
            average_set_size = np.mean(non_empty_sizes)
            median_set_size = np.median(non_empty_sizes)
        else:
            # All sets are empty - this is a complete failure
            average_set_size = 0.0
            median_set_size = 0.0
            
        # Also calculate average including empty sets for reference
        average_set_size_with_empty = np.mean(set_sizes)
        empty_set_percentage = (empty_sets / total_samples) * 100
        
        # Convert to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_probabilities = np.vstack(all_probabilities)
        
        # Handle NaN values
        all_probabilities = np.nan_to_num(all_probabilities, nan=1.0/all_probabilities.shape[1])
        
        # Calculate AUROC from base model probabilities
        base_auroc = calculate_auroc(all_true_labels, all_probabilities)
        
        # Convert all_sample_scores to a proper numpy array
        # Ensure there are no NaN or Inf values
        score_matrix = np.array(all_sample_scores)
        score_matrix = np.nan_to_num(score_matrix, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Calculate AUROC directly from conformal scores
        # Pass higher_is_better based on the scoring function's property
        # For most conformal methods, lower scores are better (more conforming)
        score_auroc = calculate_auroc(
            all_true_labels, 
            score_matrix, 
            higher_is_better=not self.lower_is_better(),  # Invert based on scorer's property
            normalize_scores=False  # Don't normalize conformal scores as they have special meaning
        )
        
        logging.info(f"AUROC from base probabilities: {base_auroc:.4f}")
        logging.info(f"AUROC from {self.__class__.__name__} nonconformity scores: {score_auroc:.4f}")
        
        # Use the scoring-specific AUROC
        auroc = score_auroc
        
        # Debug: print set size distribution
        set_size_counts = np.bincount(set_sizes)
        logging.info(f"Set size distribution:")
        for size, count in enumerate(set_size_counts):
            if count > 0:
                logging.info(f"  Size {size}: {count} samples ({count/total_samples*100:.2f}%)")
        
        # Log empty set statistics
        logging.info(f"Empty sets: {empty_sets} ({empty_set_percentage:.2f}%)")
        if empty_sets > 0:
            logging.info(f"Average set size (excluding empty): {average_set_size:.4f}")
            logging.info(f"Average set size (including empty): {average_set_size_with_empty:.4f}")
        else:
            logging.info(f"Average set size: {average_set_size:.4f}")
        
        results = {
            "dataset": self.config['dataset']['name'],
            "scoring_function": self.__class__.__name__,
            "tau": self.tau,
            "target_coverage": self.target_coverage,
            "empirical_coverage": empirical_coverage,
            "average_set_size": average_set_size,
            "median_set_size": median_set_size,
            "set_size_std": np.std(set_sizes),
            "set_size_min": np.min(set_sizes),
            "set_size_max": np.max(set_sizes),
            "auroc": auroc,
            "base_auroc": base_auroc,
            "score_auroc": score_auroc,
            "empty_sets": empty_sets,
            "empty_set_percentage": empty_set_percentage,
            "average_set_size_with_empty": average_set_size_with_empty,
        }
        
        logging.info(f"Evaluation Results for {self.config['dataset']['name']} with {self.__class__.__name__}:")
        logging.info(f"  Target Coverage: {self.target_coverage:.4f}")
        logging.info(f"  Empirical Coverage: {empirical_coverage:.4f}")
        logging.info(f"  Average Set Size (excluding empty): {average_set_size:.4f}")
        if empty_sets > 0:
            logging.info(f"  Average Set Size (including empty): {average_set_size_with_empty:.4f}")
        logging.info(f"  Empty Sets: {empty_sets} ({empty_set_percentage:.2f}%)")
        logging.info(f"  Median Set Size: {median_set_size:.4f}")
        logging.info(f"  AUROC (scoring function): {auroc:.4f}")
        logging.info(f"  AUROC (base model): {base_auroc:.4f}")
        
        # Plot score distributions
        self.plot_score_distributions(true_class_scores, false_class_scores)
        
        # Plot ROC curve using the same parameters used for score_auroc calculation
        # Prepare the same scores used for AUROC calculation
        plot_scores = score_matrix.copy()
        if self.lower_is_better():
            # If lower scores are better, use the same transformation as in calculate_auroc
            # This ensures the plot is consistent with the AUROC calculation
            plot_scores = -plot_scores if np.max(np.abs(plot_scores)) > 1.1 else 1.0 - plot_scores
            
        self.plot_roc_curve(all_true_labels, plot_scores)
        
        return results

    def create_prediction_sets(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[List[List[int]], List[float], List[float], List[List[float]]]:
        """
        Create prediction sets based on model outputs and the calibrated threshold.
        
        This method may be overridden by specific scoring functions if needed.
        
        Args:
            outputs: Model output logits of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            Tuple of (prediction_sets, true_class_scores, false_class_scores, all_class_scores)
        """
        # Check if outputs are already probabilities (sum to 1)
        if torch.allclose(outputs.sum(dim=1), torch.ones(outputs.size(0)).to(outputs.device), atol=1e-3):
            probabilities = outputs
        else:
            probabilities = F.softmax(outputs, dim=1)
        batch_size = targets.size(0)
        num_classes = probabilities.size(1)
        
        prediction_sets = []
        true_class_scores = []
        false_class_scores = []
        all_class_scores = []  # Store scores for all classes for each sample
        
        for i in range(batch_size):
            true_class = targets[i].item()
            prediction_set = []
            sample_all_scores = []  # Scores for all classes for this sample
            
            for class_idx in range(num_classes):
                # Generic computation of nonconformity score for this class
                score = self.compute_class_nonconformity_score(probabilities[i], class_idx)
                sample_all_scores.append(score)
                
                # Collect scores for true and false classes
                if class_idx == true_class:
                    true_class_scores.append(score)
                else:
                    false_class_scores.append(score)
                
                # Check inclusion criterion based on whether lower is better
                should_include = False
                if self.lower_is_better():
                    # Include class if score <= tau for lower-is-better functions
                    should_include = score <= self.tau
                else:
                    # Include class if score >= tau for higher-is-better functions
                    should_include = score >= self.tau
                
                # Add to prediction set if meets criterion
                if should_include:
                    prediction_set.append(class_idx)
            
            # Handle empty prediction sets based on configuration
            if len(prediction_set) == 0:
                empty_set_handling = self.config.get('empty_set_handling', 'keep_empty')
                
                if empty_set_handling == 'keep_empty':
                    # Keep as empty - most honest approach
                    logging.debug(f"Empty prediction set for sample {i}")
                elif empty_set_handling == 'add_most_likely':
                    # Add most likely class - maintains some coverage
                    most_likely_class = torch.argmax(probabilities[i]).item()
                    prediction_set.append(most_likely_class)
                    logging.warning(f"Empty set detected, added most likely class: {most_likely_class}")
                elif empty_set_handling == 'add_top_k':
                    # Add top-k classes where k is configurable
                    k = self.config.get('empty_set_top_k', 3)
                    top_k = torch.topk(probabilities[i], k).indices.tolist()
                    prediction_set.extend(top_k)
                    logging.warning(f"Empty set detected, added top-{k} classes")
                else:
                    raise ValueError(f"Unknown empty_set_handling: {empty_set_handling}")
            
            prediction_sets.append(prediction_set)
            all_class_scores.append(sample_all_scores)
        
        return prediction_sets, true_class_scores, false_class_scores, all_class_scores
    
    def compute_class_nonconformity_score(self, probabilities: torch.Tensor, class_idx: int) -> float:
        """
        Compute nonconformity score for a specific class based on probabilities.
        
        This is a utility method that may be used by create_prediction_sets.
        
        Args:
            probabilities: Probabilities for a single sample of shape (num_classes)
            class_idx: Class index to compute score for
            
        Returns:
            Nonconformity score for the specified class
        """
        raise NotImplementedError("Subclasses must implement this method based on their scoring function")

# ==================== Scoring Function Implementations ====================

@register_scorer("1-p")
class OneMinus_P_Scorer(BaseScorer):
    """
    Implementation of the 1-p scoring function for conformal prediction.
    
    The 1-p scoring function is defined as:
    s(x, y) = 1 - p_y
    
    Where p_y is the probability assigned to the true class y.
    """
    
    def compute_nonconformity_score(self, probabilities: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Compute nonconformity scores based on model outputs and true targets.
        
        For 1-p, the nonconformity score is 1 minus the probability of the true class.
        
        Args:
            probabilities: Softmax probabilities of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            List of nonconformity scores, one for each sample in the batch
        """
        batch_size = targets.size(0)
        nonconformity_scores = []
        
        for i in range(batch_size):
            true_class_prob = probabilities[i, targets[i]].item()
            nonconformity_score = 1 - true_class_prob
            nonconformity_scores.append(nonconformity_score)
        
        return nonconformity_scores
    
    def compute_class_nonconformity_score(self, probabilities: torch.Tensor, class_idx: int) -> float:
        """
        Compute 1-p nonconformity score for a specific class.
        
        Args:
            probabilities: Probabilities for a single sample of shape (num_classes)
            class_idx: Class index to compute score for
            
        Returns:
            1-p nonconformity score for the specified class
        """
        class_prob = probabilities[class_idx].item()
        return 1 - class_prob


@register_scorer("APS")
class APS_Scorer(BaseScorer):
    """
    Implementation of the Adaptive Prediction Sets (APS) scoring function for conformal prediction.
    
    The APS scoring function is defined based on the cumulative sum of probabilities:
    For a class y with rank k, the score is the sum of probabilities of all classes with higher probabilities.
    """
    
    def compute_nonconformity_score(self, probabilities: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Compute nonconformity scores based on model outputs and true targets.
        
        For APS, we compute the cumulative probability before the true class
        for each calibration sample.
        
        Args:
            probabilities: Softmax probabilities of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            List of nonconformity scores, one for each sample in the batch
        """
        batch_size = targets.size(0)
        nonconformity_scores = []
        
        for i in range(batch_size):
            # Get true class
            true_class = targets[i].item()
            
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probabilities[i], descending=True)
            
            # Compute cumulative sums
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Find position of true class in sorted indices
            true_class_pos = (sorted_indices == true_class).nonzero(as_tuple=True)[0].item()
            
            # Compute score: cumulative probability before true class
            if true_class_pos > 0:
                score = cum_probs[true_class_pos - 1].item()
            else:
                score = 0.0  # True class is most probable
            
            nonconformity_scores.append(score)
        
        return nonconformity_scores
    
    def compute_class_nonconformity_score(self, probabilities: torch.Tensor, class_idx: int) -> float:
        """
        Compute APS nonconformity score for a specific class.
        
        Args:
            probabilities: Probabilities for a single sample of shape (num_classes)
            class_idx: Class index to compute score for
            
        Returns:
            APS nonconformity score for the specified class
        """
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        
        # Compute cumulative sums
        cum_probs = torch.cumsum(sorted_probs, dim=0)
        
        # Find position of class in sorted indices
        class_pos = (sorted_indices == class_idx).nonzero(as_tuple=True)[0].item()
        
        # Compute score: cumulative probability before class
        if class_pos > 0:
            score = cum_probs[class_pos - 1].item()
        else:
            score = 0.0  # Class is most probable
        
        return score
    
    def create_prediction_sets(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[List[List[int]], List[float], List[float], List[List[float]]]:
        """
        Create prediction sets based on model outputs and the calibrated threshold.
        
        For APS, the prediction set includes all classes needed to exceed the threshold.
        
        Args:
            outputs: Model output logits of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            Tuple of (prediction_sets, true_class_scores, false_class_scores, all_class_scores)
        """
        # Check if outputs are already probabilities (sum to 1)
        if torch.allclose(outputs.sum(dim=1), torch.ones(outputs.size(0)).to(outputs.device), atol=1e-3):
            probabilities = outputs
        else:
            probabilities = F.softmax(outputs, dim=1)
        batch_size = targets.size(0)
        
        prediction_sets = []
        true_class_scores = []
        false_class_scores = []
        all_class_scores = []  # Store scores for all classes
        
        for i in range(batch_size):
            # Get true class
            true_class = targets[i].item()
            
            # Sort probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probabilities[i], descending=True)
            
            # Compute cumulative sums
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Find the smallest k such that cum_probs[k] >= tau
            # We use right=False to find the first index where cum_probs >= tau
            k = torch.searchsorted(cum_probs, self.tau, right=False).item()
            
            # Create prediction set - include classes 0 to k
            # Note: k is now the index where cum_probs[k] first exceeds tau
            # We want to include all classes up to and including k
            prediction_set = sorted_indices[:k+1].cpu().numpy().tolist()
            
            # Handle empty prediction sets based on configuration
            if len(prediction_set) == 0:
                empty_set_handling = self.config.get('empty_set_handling', 'keep_empty')
                
                if empty_set_handling == 'keep_empty':
                    logging.debug(f"Empty prediction set in APS for sample {i}")
                elif empty_set_handling == 'add_most_likely':
                    prediction_set = [sorted_indices[0].item()]
                    logging.warning(f"Empty APS set, added most likely class")
                elif empty_set_handling == 'add_top_k':
                    k = self.config.get('empty_set_top_k', 3)
                    prediction_set = sorted_indices[:k].tolist()
                    logging.warning(f"Empty APS set, added top-{k} classes")
            
            prediction_sets.append(prediction_set)
            
            # Store all class scores
            sample_all_scores = []
            
            # Collect scores for true and false classes
            for j, class_idx in enumerate(sorted_indices.cpu().numpy()):
                # For APS, the score is the cumulative probability before the class
                if j > 0:
                    score = cum_probs[j-1].item()
                else:
                    score = 0.0
                    
                # Store the score for this class
                # We need to map back from sorted order to original class index
                while len(sample_all_scores) <= class_idx:
                    sample_all_scores.append(0.0)  # Fill with placeholders
                sample_all_scores[class_idx] = score
                    
                if class_idx == true_class:
                    true_class_scores.append(score)
                else:
                    false_class_scores.append(score)
            
            all_class_scores.append(sample_all_scores)
        
        return prediction_sets, true_class_scores, false_class_scores, all_class_scores


@register_scorer("LogMargin")
class LogMargin_Scorer(BaseScorer):
    """
    Implementation of the Log-margin scoring function for conformal prediction.
    
    The Log-margin scoring function is defined as:
    s(x, y) = log(p_1/p_y)
    
    Where:
    - p_1 is the probability of the most likely class
    - p_y is the probability of the true class y
    """
    
    def compute_nonconformity_score(self, probabilities: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Compute nonconformity scores based on model outputs and true targets.
        
        For Log-margin, the nonconformity score is the log ratio between the highest probability
        and the true class probability.
        
        Args:
            probabilities: Softmax probabilities of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            List of nonconformity scores, one for each sample in the batch
        """
        batch_size = targets.size(0)
        nonconformity_scores = []
        
        for i in range(batch_size):
            # Get true class probability
            true_class = targets[i].item()
            true_class_prob = probabilities[i, true_class].item()
            
            # Get highest probability
            max_prob, _ = torch.max(probabilities[i], dim=0)
            max_prob = max_prob.item()
            
            # Compute log-margin score: log(p_1/p_y)
            if true_class_prob > 0:  # Avoid log(0)
                score = np.log(max_prob) - np.log(true_class_prob)
            else:
                score = float('inf')  # If true class has zero probability
            
            nonconformity_scores.append(score)
        
        return nonconformity_scores
    
    def compute_class_nonconformity_score(self, probabilities: torch.Tensor, class_idx: int) -> float:
        """
        Compute Log-margin nonconformity score for a specific class.
        
        Args:
            probabilities: Probabilities for a single sample of shape (num_classes)
            class_idx: Class index to compute score for
            
        Returns:
            Log-margin nonconformity score for the specified class
        """
        class_prob = probabilities[class_idx].item()
        max_prob, max_idx = torch.max(probabilities, dim=0)
        max_prob = max_prob.item()
        
        # If this class is the max probability class, score should be 0
        if class_idx == max_idx.item():
            return 0.0
        
        if class_prob > 0:  # Avoid log(0)
            return np.log(max_prob) - np.log(class_prob)
        else:
            return float('inf')  # If class has zero probability


@register_scorer("Sparsemax")
class Sparsemax_Scorer(BaseScorer):
    """
    Implementation of the Sparsemax scoring function for conformal prediction.
    
    The Sparsemax scoring function is defined as:
    s(x, y) = max(z_max - z_y - δ, 0)
    
    Where:
    - z_max is the highest sparsemax score
    - z_y is the sparsemax score for the true class
    - δ is a small threshold parameter that controls sparsity
    """
    
    def __init__(self, config: Dict[str, Any], dataset=None):
        """
        Initialize the Sparsemax scorer.
        
        Args:
            config: Configuration dictionary
            dataset: Optional pre-initialized dataset
        """
        super().__init__(config, dataset)
        
        # Set default delta parameter if not specified in config
        self.delta = config.get('delta', 0.05)
        if 'scoring_functions' in config and 'Sparsemax' in config['scoring_functions']:
            if 'delta' in config['scoring_functions']['Sparsemax']:
                self.delta = config['scoring_functions']['Sparsemax']['delta']
        
        logging.info(f"Sparsemax initialized with delta: {self.delta}")
    
    @staticmethod
    def sparsemax(logits):
        """
        Sparsemax function implementation
        """
        # Handle NaN values in input
        logits = torch.nan_to_num(logits, nan=0.0)
        
        # For numerical stability
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        
        # Sort logits in descending order
        z_sorted, _ = torch.sort(logits, dim=1, descending=True)
        
        # Calculate cumulative sum
        z_cumsum = torch.cumsum(z_sorted, dim=1)
        
        # Find the optimal k
        k = torch.arange(1, logits.size(1) + 1, dtype=logits.dtype, device=logits.device)
        k = k.view(1, -1)
        z_sorted = z_sorted.to(k.dtype)  # Ensure same dtype
        
        # Threshold
        threshold = 1 + k * z_sorted
        cumsum_threshold = z_cumsum < threshold
        
        # Find the largest k that satisfies the condition
        k_threshold = torch.sum(cumsum_threshold, dim=1, keepdim=True)
        
        # Edge case: if k_threshold is 0, set it to 1 to avoid division by zero
        k_threshold = torch.clamp(k_threshold, min=1)
        
        # Calculate threshold value
        tau = (z_cumsum.gather(1, k_threshold - 1) - 1) / k_threshold.to(z_cumsum.dtype)
        
        # Calculate sparsemax values
        sparsemax_val = torch.clamp(logits - tau, min=0)
        
        # Normalize to ensure sum to 1
        sum_vals = torch.sum(sparsemax_val, dim=1, keepdim=True)
        # Handle case where sum is 0
        sum_vals = torch.where(sum_vals == 0, torch.ones_like(sum_vals), sum_vals)
        sparsemax_val = sparsemax_val / sum_vals
        
        # Ensure no NaN values in output
        sparsemax_val = torch.nan_to_num(sparsemax_val, nan=1.0/logits.size(1))
        
        return sparsemax_val
    
    def compute_nonconformity_score(self, probabilities: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """
        Compute nonconformity scores based on model outputs and true targets.
        
        For Sparsemax, we need to first transform the probabilities, then compute
        the nonconformity scores.
        
        Args:
            probabilities: Softmax probabilities of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            List of nonconformity scores, one for each sample in the batch
        """
        # Safety check to avoid NaN or inf values in input probabilities
        probabilities = torch.nan_to_num(probabilities, nan=1.0/probabilities.size(1))
        
        # We need to convert probabilities back to logits for sparsemax
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        logits = torch.log(probabilities + epsilon)
        
        # Apply sparsemax transformation
        sparsemax_probs = self.sparsemax(logits)
        
        batch_size = targets.size(0)
        nonconformity_scores = []
        
        for i in range(batch_size):
            # Get true class
            true_class = targets[i].item()
            
            # Get sparsemax score for true class
            true_class_score = sparsemax_probs[i, true_class].item()
            
            # Get highest sparsemax score
            max_score, _ = torch.max(sparsemax_probs[i], dim=0)
            max_score = max_score.item()
            
            # For sparsemax, handle zero probabilities with high penalty
            if true_class_score < 1e-10:  # Effectively zero
                score = 1000.0  # High penalty for zero probabilities
            else:
                # Compute nonconformity score: max(z_max - z_y - delta, 0)
                score = max(max_score - true_class_score - self.delta, 0)
            
            nonconformity_scores.append(score)
        
        return nonconformity_scores
    
    def compute_class_nonconformity_score(self, probabilities: torch.Tensor, class_idx: int) -> float:
        """
        Compute nonconformity score for a specific class based on sparsemax of logits.
        
        Args:
            probabilities: Softmax probabilities for a single sample of shape (num_classes)
            class_idx: The index of the class to compute the score for
            
        Returns:
            The nonconformity score for the specified class
        """
        try:
            # Add a small epsilon to avoid log(0)
            epsilon = 1e-10
            logits = torch.log(probabilities + epsilon)
            
            # Handle NaN values
            logits = torch.nan_to_num(logits, nan=0.0)
            
            # Apply sparsemax to get probability distribution
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # Add batch dimension if missing
                
            sparsemax_probs = self.sparsemax(logits)
            
            # Get the class probability
            if sparsemax_probs.dim() > 1:
                sparsemax_probs = sparsemax_probs.squeeze(0)  # Remove batch dimension
                
            # Check if class_idx is valid
            if class_idx >= len(sparsemax_probs):
                logging.warning(f"Invalid class index {class_idx} for tensor of size {sparsemax_probs.size()}")
                return 1.0  # Return worst nonconformity score
                
            class_prob = sparsemax_probs[class_idx].item()
            max_prob = torch.max(sparsemax_probs).item()
            
            # For sparsemax, classes with zero probability should have infinite score
            if class_prob < 1e-10:  # Effectively zero
                return float('inf')
            
            # Calculate nonconformity score: max(max_prob - class_prob - delta, 0)
            score = max(max_prob - class_prob - self.delta, 0)
            
            return float(score)
        except Exception as e:
            logging.warning(f"Error in compute_class_nonconformity_score: {e}")
            return 1.0  # Return worst nonconformity score
    
    def create_prediction_sets(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[List[List[int]], List[float], List[float], List[List[float]]]:
        """
        Create prediction sets based on model outputs and the calibrated threshold.
        
        For Sparsemax, the prediction set includes all classes with score <= tau.
        
        Args:
            outputs: Model output logits of shape (batch_size, num_classes)
            targets: True class labels of shape (batch_size)
            
        Returns:
            Tuple of (prediction_sets, true_class_scores, false_class_scores, all_class_scores)
        """
        # Apply softmax and then convert to logits for sparsemax
        # Check if outputs are already probabilities (sum to 1)
        if torch.allclose(outputs.sum(dim=1), torch.ones(outputs.size(0)).to(outputs.device), atol=1e-3):
            probabilities = outputs
        else:
            probabilities = F.softmax(outputs, dim=1)
        epsilon = 1e-10
        probabilities = torch.nan_to_num(probabilities, nan=1.0/probabilities.size(1))
        logits = torch.log(probabilities + epsilon)
        
        # Apply sparsemax transformation
        sparsemax_probs = self.sparsemax(logits)
        
        batch_size = targets.size(0)
        num_classes = sparsemax_probs.size(1)
        
        prediction_sets = []
        true_class_scores = []
        false_class_scores = []
        all_class_scores = []
        
        for i in range(batch_size):
            # Get true class
            true_class = targets[i].item()
            
            # Get highest sparsemax score
            max_score, max_idx = torch.max(sparsemax_probs[i], dim=0)
            max_score = max_score.item()
            
            prediction_set = []
            sample_all_scores = []
            
            # Compute scores for all classes
            for class_idx in range(num_classes):
                class_score = sparsemax_probs[i, class_idx].item()
                
                # For sparsemax, classes with zero probability should have infinite score
                if class_score < 1e-10:  # Effectively zero
                    score = float('inf')  # Infinite score for zero probabilities
                else:
                    # Compute nonconformity score: max(z_max - z_class - delta, 0)
                    score = max(max_score - class_score - self.delta, 0)
                sample_all_scores.append(score)
                
                # Collect scores for true and false classes
                if class_idx == true_class:
                    true_class_scores.append(score)
                else:
                    false_class_scores.append(score)
                
                # Add to prediction set if score <= tau
                if score <= self.tau:
                    prediction_set.append(class_idx)
            
            # Handle empty prediction sets based on configuration
            if len(prediction_set) == 0:
                empty_set_handling = self.config.get('empty_set_handling', 'keep_empty')
                
                if empty_set_handling == 'keep_empty':
                    logging.debug(f"Empty prediction set in Sparsemax for sample {i}")
                elif empty_set_handling == 'add_most_likely':
                    most_likely_class = torch.argmax(probabilities[i]).item()
                    prediction_set.append(most_likely_class)
                    logging.warning(f"Empty Sparsemax set, added most likely class: {most_likely_class}")
                elif empty_set_handling == 'add_top_k':
                    k = self.config.get('empty_set_top_k', 3)
                    top_k = torch.topk(probabilities[i], k).indices.tolist()
                    prediction_set.extend(top_k)
                    logging.warning(f"Empty Sparsemax set, added top-{k} classes")
            
            prediction_sets.append(prediction_set)
            all_class_scores.append(sample_all_scores)
        
        return prediction_sets, true_class_scores, false_class_scores, all_class_scores

# ==================== Runner Functions ====================

def create_unified_csv_summary(dataset_results: Dict[str, Dict[str, Any]], dataset_name: str, config: Dict[str, Any]) -> None:
    """
    Create a unified CSV summary for a single dataset with all scoring functions.
    
    Args:
        dataset_results: Results for all scoring functions for this dataset
        dataset_name: Name of the dataset
        config: Configuration dictionary
    """
    # Setup output directory
    base_output_dir = os.path.join(config.get('base_dir', '.'), 'results_static_conformal', dataset_name.replace('vlm_', '').split('_')[0])
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    
    for scoring_function, results in dataset_results.items():
        if isinstance(results, dict) and "error" not in results:
            row = {
                'Dataset': dataset_name,
                'Scoring_Function': scoring_function,
                'Target_Coverage': results.get('target_coverage', 'N/A'),
                'Empirical_Coverage': results.get('empirical_coverage', 'N/A'),
                'Average_Set_Size_Excluding_Empty': results.get('average_set_size', 'N/A'),
                'Average_Set_Size_Including_Empty': results.get('average_set_size_with_empty', 'N/A'),
                'Median_Set_Size': results.get('median_set_size', 'N/A'),
                'Min_Set_Size': results.get('set_size_min', 'N/A'),
                'Max_Set_Size': results.get('set_size_max', 'N/A'),
                'Set_Size_Std': results.get('set_size_std', 'N/A'),
                'Empty_Sets_Count': results.get('empty_sets', 'N/A'),
                'Empty_Sets_Percentage': results.get('empty_set_percentage', 'N/A'),
                'AUROC_Scoring_Function': results.get('score_auroc', 'N/A'),
                'AUROC_Base_Model': results.get('base_auroc', 'N/A'),
                'Tau_Threshold': results.get('tau', 'N/A')
            }
            csv_data.append(row)
        elif "error" in results:
            row = {
                'Dataset': dataset_name,
                'Scoring_Function': scoring_function,
                'Target_Coverage': 'ERROR',
                'Empirical_Coverage': 'ERROR',
                'Average_Set_Size_Excluding_Empty': 'ERROR',
                'Average_Set_Size_Including_Empty': 'ERROR',
                'Median_Set_Size': 'ERROR',
                'Min_Set_Size': 'ERROR',
                'Max_Set_Size': 'ERROR',
                'Set_Size_Std': 'ERROR',
                'Empty_Sets_Count': 'ERROR',
                'Empty_Sets_Percentage': 'ERROR',
                'AUROC_Scoring_Function': 'ERROR',
                'AUROC_Base_Model': 'ERROR',
                'Tau_Threshold': 'ERROR'
            }
            csv_data.append(row)
    
    if csv_data:
        # Create DataFrame and save as CSV
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(base_output_dir, f'{dataset_name}_static_conformal_results.csv')
        df.to_csv(csv_file, index=False)
        
        # Create a readable JSON summary (formatted like a table)
        json_summary = {
            "dataset": dataset_name,
            "target_coverage": csv_data[0].get('Target_Coverage', 'N/A') if csv_data else 'N/A',
            "summary": "Static Conformal Prediction Results",
            "scoring_functions": {}
        }
        
        for row in csv_data:
            scoring_func = row['Scoring_Function']
            json_summary["scoring_functions"][scoring_func] = {
                "empirical_coverage": row['Empirical_Coverage'],
                "average_set_size_excluding_empty": row['Average_Set_Size_Excluding_Empty'],
                "average_set_size_including_empty": row['Average_Set_Size_Including_Empty'],
                "median_set_size": row['Median_Set_Size'],
                "min_set_size": row['Min_Set_Size'],
                "max_set_size": row['Max_Set_Size'],
                "set_size_std": row['Set_Size_Std'],
                "empty_sets_count": row['Empty_Sets_Count'],
                "empty_sets_percentage": row['Empty_Sets_Percentage'],
                "auroc_scoring_function": row['AUROC_Scoring_Function'],
                "auroc_base_model": row['AUROC_Base_Model'],
                "tau_threshold": row['Tau_Threshold']
            }
        
        # Save readable JSON summary
        json_summary_file = os.path.join(base_output_dir, f'{dataset_name}_static_conformal_summary.json')
        with open(json_summary_file, 'w') as f:
            json.dump(json_summary, f, indent=4)
        
        # Also save individual JSON results for each scoring function
        for scoring_function, results in dataset_results.items():
            json_file = os.path.join(base_output_dir, f'{scoring_function}_{dataset_name}_results.json')
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=4)
        
        logging.info(f"Unified CSV summary saved to {csv_file}")
        logging.info(f"Readable JSON summary saved to {json_summary_file}")
        logging.info(f"Individual JSON results saved in {base_output_dir}")

def create_comparison_report(results: Dict[str, Dict[str, Dict[str, Any]]], output_dir: str, target_coverage: float = 0.9) -> Dict[str, str]:
    """
    Create a comprehensive comparison report with multiple visualizations.
    
    Args:
        results: Nested dictionary of results (dataset -> scorer -> metrics)
        output_dir: Directory to save the report
        target_coverage: Target coverage value
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot comparative metrics
    plots = {}
    
    # Convert results to a dataframe for easier plotting
    rows = []
    
    for dataset, scorers in results.items():
        # Handle special case for VLM with multiple models and datasets
        if dataset == 'vlm' and isinstance(scorers, dict):
            for model_name, model_results in scorers.items():
                # Check if model_results is organized by dataset
                if any(isinstance(v, dict) and not 'error' in v for v in model_results.values()):
                    # New structure: model -> dataset -> scorer
                    for dataset_name, dataset_results in model_results.items():
                        for scorer, metrics in dataset_results.items():
                            if isinstance(metrics, dict) and "error" not in metrics:
                                rows.append({
                                    'Dataset': f'vlm/{model_name}/{dataset_name}',
                                    'Model': model_name,
                                    'VLM Dataset': dataset_name,
                                    'Scorer': scorer,
                                    'Empirical Coverage': metrics.get('empirical_coverage', 'N/A'),
                                    'Average Set Size': metrics.get('average_set_size', 'N/A'),
                                    'Median Set Size': metrics.get('median_set_size', 'N/A'),
                                    'Min Set Size': metrics.get('set_size_min', 'N/A'),
                                    'Max Set Size': metrics.get('set_size_max', 'N/A'),
                                    'AUROC': metrics.get('auroc', 'N/A')
                                })
                else:
                    # Old structure: model -> scorer (for backward compatibility)
                    for scorer, metrics in model_results.items():
                        if isinstance(metrics, dict) and "error" not in metrics:
                            rows.append({
                                'Dataset': f'vlm/{model_name}',
                                'Model': model_name,
                                'VLM Dataset': 'ai2d',  # Assume ai2d for old structure
                                'Scorer': scorer,
                                'Empirical Coverage': metrics.get('empirical_coverage', 'N/A'),
                                'Average Set Size': metrics.get('average_set_size', 'N/A'),
                                'Median Set Size': metrics.get('median_set_size', 'N/A'),
                                'Min Set Size': metrics.get('set_size_min', 'N/A'),
                                'Max Set Size': metrics.get('set_size_max', 'N/A'),
                                'AUROC': metrics.get('auroc', 'N/A')
                            })
        else:
            # Standard datasets
            for scorer, metrics in scorers.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    rows.append({
                        'Dataset': dataset,
                        'Model': 'N/A',
                        'VLM Dataset': 'N/A',
                        'Scorer': scorer,
                        'Empirical Coverage': metrics.get('empirical_coverage', 'N/A'),
                        'Average Set Size': metrics.get('average_set_size', 'N/A'),
                        'Median Set Size': metrics.get('median_set_size', 'N/A'),
                        'Min Set Size': metrics.get('set_size_min', 'N/A'),
                        'Max Set Size': metrics.get('set_size_max', 'N/A'),
                        'AUROC': metrics.get('auroc', 'N/A')
                    })
    
    if not rows:
        logging.warning("No valid results available for comparison report")
        return {}
    
    df = pd.DataFrame(rows)
    
    # Create the plots
    plt.figure(figsize=(12, 8))
    
    # Average set size comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Dataset', y='Average Set Size', hue='Scorer', data=df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    plt.title(f'Average Prediction Set Size Comparison (Target Coverage: {target_coverage:.2f})')
    plt.legend(title='Scoring Function', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()
    avg_size_path = os.path.join(output_dir, 'average_set_size_comparison.png')
    plt.savefig(avg_size_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots['average_set_size'] = avg_size_path
    
    # Empirical coverage comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Dataset', y='Empirical Coverage', hue='Scorer', data=df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    plt.title(f'Empirical Coverage Comparison (Target: {target_coverage:.2f})')
    plt.axhline(y=target_coverage, color='r', linestyle='--', label=f'Target ({target_coverage:.2f})')
    plt.legend(title='Scoring Function', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()
    coverage_path = os.path.join(output_dir, 'empirical_coverage_comparison.png')
    plt.savefig(coverage_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots['empirical_coverage'] = coverage_path
    
    # AUROC comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Dataset', y='AUROC', hue='Scorer', data=df)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f')
    plt.title('AUROC Comparison (Higher is Better)')
    plt.legend(title='Scoring Function', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.tight_layout()
    auroc_path = os.path.join(output_dir, 'auroc_comparison.png')
    plt.savefig(auroc_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots['auroc'] = auroc_path
    
    # Save summary as CSV
    summary_path = os.path.join(output_dir, 'metrics_summary.csv')
    df.to_csv(summary_path, index=False)
    plots['summary_table'] = summary_path
    
    # Also save as markdown for better readability
    try:
        markdown_path = os.path.join(output_dir, 'metrics_summary.md')
        with open(markdown_path, 'w') as f:
            f.write(f"# Conformal Prediction Metrics Summary\n\n")
            f.write(f"Target Coverage: {target_coverage:.2f}\n\n")
            f.write(df.to_markdown(index=False))
        plots['markdown_summary'] = markdown_path
    except Exception as e:
        logging.warning(f"Failed to create markdown summary: {str(e)}")
    
    # Create VLM-specific comparison if we have multiple VLM models
    try:
        # Extract VLM-specific data
        vlm_df = df[df['Dataset'].str.contains('vlm/')]
        
        if not vlm_df.empty:
            logging.info("Creating VLM-specific comparison plots")
            
            # Create model-specific comparisons across datasets and scoring functions
            for scorer in vlm_df['Scorer'].unique():
                scorer_df = vlm_df[vlm_df['Scorer'] == scorer]
                
                # Check if we have multiple datasets for any model
                if len(scorer_df['VLM Dataset'].unique()) > 1:
                    # Create dataset comparison plots for each model and scoring function
                    plt.figure(figsize=(14, 8))
                    ax = sns.barplot(x='Model', y='Average Set Size', hue='VLM Dataset', data=scorer_df)
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.3f')
                    plt.title(f'VLM Dataset Comparison - Average Set Size with {scorer} scoring function')
                    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    model_avg_path = os.path.join(output_dir, f'vlm_{scorer}_datasets_avg_size.png')
                    plt.savefig(model_avg_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots[f'vlm_{scorer}_datasets_avg_size'] = model_avg_path
                    
                    # Coverage comparison
                    plt.figure(figsize=(14, 8))
                    ax = sns.barplot(x='Model', y='Empirical Coverage', hue='VLM Dataset', data=scorer_df)
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.3f')
                    plt.axhline(y=target_coverage, color='r', linestyle='--', label=f'Target ({target_coverage:.2f})')
                    plt.title(f'VLM Dataset Comparison - Coverage with {scorer} scoring function')
                    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    model_cov_path = os.path.join(output_dir, f'vlm_{scorer}_datasets_coverage.png')
                    plt.savefig(model_cov_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots[f'vlm_{scorer}_datasets_coverage'] = model_cov_path
            
            # For each VLM dataset, create model comparison plots
            for dataset_name in vlm_df['VLM Dataset'].unique():
                dataset_df = vlm_df[vlm_df['VLM Dataset'] == dataset_name]
                
                # Model comparison for average set size
                plt.figure(figsize=(16, 10))
                ax = sns.barplot(x='Model', y='Average Set Size', hue='Scorer', data=dataset_df)
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f')
                plt.title(f'VLM Models Comparison - {dataset_name} Dataset - Average Set Size')
                plt.legend(title='Scoring Function', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                dataset_avg_path = os.path.join(output_dir, f'vlm_{dataset_name}_models_avg_size.png')
                plt.savefig(dataset_avg_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots[f'vlm_{dataset_name}_models_avg_size'] = dataset_avg_path
                
                # Model comparison for empirical coverage
                plt.figure(figsize=(16, 10))
                ax = sns.barplot(x='Model', y='Empirical Coverage', hue='Scorer', data=dataset_df)
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f')
                plt.axhline(y=target_coverage, color='r', linestyle='--', label=f'Target ({target_coverage:.2f})')
                plt.title(f'VLM Models Comparison - {dataset_name} Dataset - Empirical Coverage')
                plt.legend(title='Scoring Function', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
                plt.tight_layout()
                dataset_cov_path = os.path.join(output_dir, f'vlm_{dataset_name}_models_coverage.png')
                plt.savefig(dataset_cov_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots[f'vlm_{dataset_name}_models_coverage'] = dataset_cov_path
            
            # Save VLM-specific summary
            vlm_summary_path = os.path.join(output_dir, 'vlm_metrics_summary.csv')
            vlm_df.to_csv(vlm_summary_path, index=False)
            plots['vlm_summary_table'] = vlm_summary_path
            
            # Also save as markdown
            vlm_markdown_path = os.path.join(output_dir, 'vlm_metrics_summary.md')
            with open(vlm_markdown_path, 'w') as f:
                f.write(f"# VLM Models Conformal Prediction Metrics Summary\n\n")
                f.write(f"Target Coverage: {target_coverage:.2f}\n\n")
                f.write(vlm_df.to_markdown(index=False))
            plots['vlm_markdown_summary'] = vlm_markdown_path
    except Exception as e:
        logging.error(f"Failed to create VLM-specific comparison plots: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
    
    return plots

def run_dataset(config: Dict[str, Any], dataset_name: str, scoring_name: str) -> Dict[str, Any]:
    """
    Run the scoring function evaluation for a specific dataset.
    
    Args:
        config: Configuration dictionary
        dataset_name: Name of the dataset
        scoring_name: Name of the scoring function
        
    Returns:
        Results dictionary
    """
    # Update config with dataset name
    dataset_config = get_dataset_config(config, dataset_name)
    
    # Setup unified directory structure
    base_output_dir = os.path.join(config.get('base_dir', '.'), 'results_static_conformal', dataset_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Setup unified logging
    log_file = os.path.join(base_output_dir, f'{dataset_name}_static_conformal.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add the file handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    try:
        # Include model name in log message for VLM
        if dataset_name == 'vlm':
            model_name = dataset_config['dataset'].get('default_model', 'default')
            dataset_desc = f"{dataset_name} ({model_name})"
        else:
            dataset_desc = dataset_name
            
        logging.info(f"=== Starting evaluation for {dataset_desc} dataset using {scoring_name} scoring function ===")
        
        # For ImageNet, load the dataset-specific configuration
        if dataset_name == 'imagenet':
            # Load ImageNet-specific configuration if exists
            imagenet_config_path = os.path.join(config.get('base_dir', '.'), 'src', 'config', 'imagenet.yaml')
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
            scorer = get_scorer(scoring_name, dataset_config)
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
            
            logging.info(f"Results saved in unified directory: {base_output_dir}")
            logging.info(f"=== Completed evaluation for {dataset_desc} dataset ===\n")
            
            return json_results
        except ValueError as e:
            if "num_samples" in str(e):
                logging.error(f"Error running {scoring_name} evaluation for {dataset_desc}: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                return {"error": str(e)}
            else:
                raise
    except Exception as e:
        logging.error(f"Error running {scoring_name} evaluation for {dataset_desc}: {str(e)}")
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
    parser.add_argument('--config', type=str, default='src/config/base_conformal_scorers.yaml', 
                        help='Path to config file (default: src/config/base_conformal_scorers.yaml)')
    parser.add_argument('--dataset', type=str, default='all', 
                        help='Dataset to evaluate (default: all)')
    parser.add_argument('--scoring', type=str, default='all',
                        help='Scoring function to use (default: all)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run experiments in parallel (default: False)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: number of scorers * datasets)')
    parser.add_argument('--vlm-model', type=str, default=None,
                        help='Specific VLM model to evaluate (default: use config or all)')
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up directories
    for dir_key in ['log_dir', 'plot_dir', 'result_dir']:
        if dir_key in config:
            os.makedirs(config[dir_key], exist_ok=True)
    
    # Set up logging
    setup_logging(config)
    
    # Log configuration
    logging.info(f"Configuration loaded from {args.config}")
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Determine which datasets to run
    available_datasets = ['cifar10', 'cifar100', 'imagenet', 'vlm']  # Updated supported datasets
    if args.dataset == 'all':
        datasets = available_datasets
        logging.info(f"Running evaluation for all datasets: {', '.join(datasets)}")
    else:
        datasets = [args.dataset]
        logging.info(f"Running evaluation for {args.dataset} dataset")
    
    # Filter to only available datasets
    datasets = [d for d in datasets if d in available_datasets]
    if not datasets:
        logging.error(f"No valid datasets specified. Available datasets: {', '.join(available_datasets)}")
        sys.exit(1)
    
    # Determine which scoring functions to run
    available_scorers = list(SCORER_REGISTRY.keys())
    if args.scoring == 'all':
        scoring_functions = available_scorers
        logging.info(f"Running evaluation for all scoring functions: {', '.join(scoring_functions)}")
    else:
        if args.scoring in available_scorers:
            scoring_functions = [args.scoring]
            logging.info(f"Running evaluation for {args.scoring} scoring function")
        else:
            logging.error(f"Scoring function '{args.scoring}' not found. Available scorers: {', '.join(available_scorers)}")
            sys.exit(1)
    
    # Handle VLM models
    vlm_models = []
    if 'vlm' in datasets:
        # Check if we should run all models for VLM
        run_all_models = config.get('vlm', {}).get('run_all_models', False)
        
        if args.vlm_model:
            # User specified a specific VLM model
            vlm_models = [args.vlm_model]
            logging.info(f"Running evaluation for VLM model: {args.vlm_model}")
        elif run_all_models:
            # Run all available VLM models
            vlm_models = config.get('vlm', {}).get('models', ['cogagent-vqa-hf'])
            logging.info(f"Running evaluation for all VLM models: {', '.join(vlm_models)}")
        else:
            # Use default model from config
            default_model = config.get('vlm', {}).get('default_model', 'cogagent-vqa-hf')
            vlm_models = [default_model]
            logging.info(f"Running evaluation for default VLM model: {default_model}")
    
    # Run evaluation for each dataset and scorer combination
    all_results = {}
    
    # Special handling for VLM to iterate through models and datasets
    if 'vlm' in datasets and vlm_models:
        # Setup VLM dataset directory
        datasets.remove('vlm')  # Remove 'vlm' from standard processing
        all_results['vlm'] = {}
        
        # Get all available VLM datasets
        vlm_datasets = config.get('vlm', {}).get('datasets', ['ai2d'])
        logging.info(f"VLM datasets to process: {', '.join(vlm_datasets)}")
        
        for vlm_model in vlm_models:
            logging.info(f"=== Running evaluation for VLM model: {vlm_model} ===")
            
            # Initialize model results dictionary
            all_results['vlm'][vlm_model] = {}
            
            # Process each VLM dataset
            for vlm_dataset in vlm_datasets:
                logging.info(f"=== Processing dataset: {vlm_dataset} for model: {vlm_model} ===")
                
                # Create a copy of the config to modify for this model and dataset
                model_config = copy.deepcopy(config)
                # Update the model name
                model_config['vlm']['default_model'] = vlm_model
                # Set the current dataset
                model_config['vlm']['default_dataset'] = vlm_dataset
                
                # Run scorers for this model and dataset combination
                dataset_results = {}
                for scoring in scoring_functions:
                    try:
                        # Add model and dataset names to logging for clarity
                        logging.info(f"Running {scoring} evaluation for VLM model {vlm_model} on {vlm_dataset} dataset")
                        results = run_dataset(model_config, 'vlm', scoring)
                        dataset_results[scoring] = results
                    except Exception as e:
                        logging.error(f"Failed to run {scoring} evaluation for VLM model {vlm_model} on {vlm_dataset}: {str(e)}")
                        logging.error(f"Traceback: {traceback.format_exc()}")
                        dataset_results[scoring] = {"error": str(e)}
                
                # Store results for this dataset
                all_results['vlm'][vlm_model][vlm_dataset] = dataset_results
    
    if args.parallel and datasets:
        # Determine number of workers
        num_workers = args.num_workers or min(len(datasets) * len(scoring_functions), os.cpu_count() or 4)
        logging.info(f"Running experiments in parallel with {num_workers} workers")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_experiment = {
                executor.submit(run_dataset, config, dataset, scoring): 
                (dataset, scoring)
                for dataset in datasets
                for scoring in scoring_functions
            }
            
            for future in concurrent.futures.as_completed(future_to_experiment):
                dataset, scoring = future_to_experiment[future]
                try:
                    results = future.result()
                    if dataset not in all_results:
                        all_results[dataset] = {}
                    all_results[dataset][scoring] = results
                    logging.info(f"Completed {scoring} evaluation for {dataset}")
                except Exception as e:
                    logging.error(f"Failed to run {scoring} evaluation for {dataset}: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    if dataset not in all_results:
                        all_results[dataset] = {}
                    all_results[dataset][scoring] = {"error": str(e)}
    else:
        for dataset in datasets:
            all_results[dataset] = {}
            for scoring in scoring_functions:
                try:
                    results = run_dataset(config, dataset, scoring)
                    all_results[dataset][scoring] = results
                    logging.info(f"Completed {scoring} evaluation for {dataset}")
                except Exception as e:
                    logging.error(f"Failed to run {scoring} evaluation for {dataset}: {str(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    all_results[dataset][scoring] = {"error": str(e)}
    
    # Create unified CSV summary for each dataset
    for dataset in datasets:
        if dataset in all_results:
            create_unified_csv_summary(all_results[dataset], dataset, config)
    
    # Handle VLM results separately
    if 'vlm' in all_results:
        for model_name, model_results in all_results['vlm'].items():
            for dataset_name, dataset_results in model_results.items():
                create_unified_csv_summary(dataset_results, f'vlm_{model_name}_{dataset_name}', config)
    
    # Print summary table
    logging.info(f"\n=== Conformal Prediction Evaluation Summary ===")
    logging.info(f"{'Dataset':<10} | {'Scoring':<10} | {'Coverage':<10} | {'Set Size (w/empty)':<20} | {'Empty %':<8} | {'AUROC':<10} | {'Status':<10}")
    logging.info("-" * 100)
    
    # First, print non-VLM datasets
    for dataset in [d for d in all_results.keys() if d != 'vlm']:
        for scoring in scoring_functions:
            results = all_results.get(dataset, {}).get(scoring, {})
            
            if "error" in results:
                status = "ERROR"
                coverage = "N/A"
                avg_size = "N/A"
                auroc = "N/A"
            else:
                status = "SUCCESS"
                coverage = results.get("empirical_coverage", "N/A")
                avg_size = results.get("average_set_size", "N/A")
                avg_size_with_empty = results.get("average_set_size_with_empty", "N/A")
                empty_pct = results.get("empty_set_percentage", 0.0)
                auroc = results.get("auroc", "N/A")
                
                if isinstance(coverage, float):
                    coverage = f"{coverage:.4f}"
                if isinstance(avg_size, float):
                    avg_size = f"{avg_size:.4f}"
                if isinstance(avg_size_with_empty, float):
                    avg_size_with_empty = f"{avg_size_with_empty:.4f}"
                if isinstance(auroc, float):
                    auroc = f"{auroc:.4f}"
                    
            # Show both set size metrics
            set_size_str = f"{avg_size_with_empty} ({avg_size})"
            empty_str = f"{empty_pct:.1f}%" if isinstance(empty_pct, float) else "N/A"
            logging.info(f"{dataset:<10} | {scoring:<10} | {coverage:<10} | {set_size_str:<20} | {empty_str:<8} | {auroc:<10} | {status:<10}")
    
    # Then, print VLM models - updated to include datasets
    if 'vlm' in all_results:
        # Print header for VLM results
        logging.info("\n=== VLM Models Evaluation Summary ===")
        logging.info(f"{'Model/Dataset':<30} | {'Scoring':<10} | {'Coverage':<10} | {'Avg Set Size':<15} | {'AUROC':<10} | {'Status':<10}")
        logging.info("-" * 95)
        
        for model_name, model_results in all_results['vlm'].items():
            for dataset_name, dataset_results in model_results.items():
                for scoring, results in dataset_results.items():
                    if "error" in results:
                        status = "ERROR"
                        coverage = "N/A"
                        avg_size = "N/A"
                        auroc = "N/A"
                    else:
                        status = "SUCCESS"
                        coverage = results.get("empirical_coverage", "N/A")
                        avg_size = results.get("average_set_size", "N/A")
                        auroc = results.get("auroc", "N/A")
                        
                        if isinstance(coverage, float):
                            coverage = f"{coverage:.4f}"
                        if isinstance(avg_size, float):
                            avg_size = f"{avg_size:.4f}"
                        if isinstance(auroc, float):
                            auroc = f"{auroc:.4f}"
                            
                    logging.info(f"{model_name}/{dataset_name:<30} | {scoring:<10} | {coverage:<10} | {avg_size:<15} | {auroc:<10} | {status:<10}")
    
    logging.info("All experiments completed!")

if __name__ == "__main__":
    main() 