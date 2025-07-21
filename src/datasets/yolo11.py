"""
YOLO11 dataset module for loading cached YOLO11 object detection outputs
and preparing them for the scoring function.
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from torch.utils.data import Dataset as TorchDataset, DataLoader
import random
from src.datasets.base import BaseDataset

class YOLO11SampleDataset(TorchDataset):
    """
    PyTorch Dataset for YOLO11 detection logits to be used with DataLoader
    """
    def __init__(self, class_probs, targets):
        """
        Initialize dataset with class probabilities and targets
        
        Args:
            class_probs: Tensor of shape [num_samples, num_classes] containing class probabilities
            targets: Tensor of shape [num_samples] containing target class indices (0 to num_classes-1)
        """
        self.class_probs = class_probs
        self.targets = targets
        self.num_classes = class_probs.shape[1]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Return class probabilities and target class
        probs = self.class_probs[idx]  # Shape: [num_classes]
        target = self.targets[idx].item()  # Get scalar value
        
        # Ensure we have the right shape for the probabilities
        # This should be redundant with our other fixes, but add as extra safety
        if len(probs) != self.num_classes:
            logging.warning(f"Fixing probability shape mismatch in dataset: {len(probs)} != {self.num_classes}")
            if len(probs) > self.num_classes:
                # Trim extra columns
                probs = probs[:self.num_classes]
            else:
                # Pad with zeros and renormalize
                import torch
                import numpy as np
                if isinstance(probs, torch.Tensor):
                    padded_probs = torch.zeros(self.num_classes)
                    padded_probs[:len(probs)] = probs
                    probs = padded_probs / padded_probs.sum() if padded_probs.sum() > 0 else torch.ones(self.num_classes) / self.num_classes
                else:
                    padded_probs = np.zeros(self.num_classes, dtype=np.float32)
                    padded_probs[:len(probs)] = probs
                    probs = padded_probs / np.sum(padded_probs) if np.sum(padded_probs) > 0 else np.ones(self.num_classes) / self.num_classes
        
        return probs, target

class Dataset(BaseDataset):
    """
    YOLO11 dataset class for loading cached object detection outputs
    and preparing them for the scoring function.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLO11 dataset
        
        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        
        # Extract YOLO11-specific configurations
        self.yolo_dir = self.config.get('data_dir', '/mnt/ssd1/divake/learnable_scoring_funtion_01/cache/yolo11_cache')
        self.dataset_name = self.config['dataset'].get('default_dataset', 'coco')
        
        # Get split paths from config
        self.splits = self.config['dataset'].get('splits', {
            'train': 'training',
            'validation': 'validation',
            'calibration': 'calibration'
        })
        
        # Store the scoring function input dimension for our model
        self._num_classes = self.config['dataset'].get('num_classes', 79)  # Updated default to 79 classes
        
        # Load the data
        self.setup()
    
    def setup(self):
        """
        Load and prepare YOLO11 cached data
        """
        logging.info(f"Loading YOLO11 cached data for dataset {self.dataset_name}")
        
        # Process data for each split
        train_probs, train_targets = self._load_split(self.splits['train'])
        val_probs, val_targets = self._load_split(self.splits['validation'])
        cal_probs, cal_targets = self._load_split(self.splits['calibration'])
        
        # Convert to tensors
        self.train_probs = torch.tensor(train_probs, dtype=torch.float32)
        self.train_targets = torch.tensor(train_targets, dtype=torch.long)
        
        self.val_probs = torch.tensor(val_probs, dtype=torch.float32)
        self.val_targets = torch.tensor(val_targets, dtype=torch.long)
        
        self.cal_probs = torch.tensor(cal_probs, dtype=torch.float32)
        self.cal_targets = torch.tensor(cal_targets, dtype=torch.long)
        
        # Log the number of unique classes found
        train_unique = len(torch.unique(self.train_targets))
        val_unique = len(torch.unique(self.val_targets))
        cal_unique = len(torch.unique(self.cal_targets))
        
        logging.info(f"Unique classes found - train: {train_unique}, val: {val_unique}, cal: {cal_unique}")
        
        # Create dataloaders
        self._create_dataloaders()
        
        logging.info(f"YOLO11 dataset prepared with {len(self.train_probs)} training samples, "
                    f"{len(self.cal_probs)} calibration samples, and "
                    f"{len(self.val_probs)} validation samples")
    
    def _get_bbox_area(self, bbox):
        """
        Calculate the area of a bounding box
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Area of the bounding box
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
    
    def _load_split(self, split_name):
        """
        Load and process data for a specific split
        
        Args:
            split_name: Name of the split to load ('training', 'validation', or 'calibration')
            
        Returns:
            Tuple of (class_probs, targets) arrays
        """
        logging.info(f"Loading {split_name} split")
        
        # Set up file paths
        predictions_path = os.path.join(self.yolo_dir, split_name, 'yolo11_predictions.pt')
        probs_path = os.path.join(self.yolo_dir, split_name, 'yolo11_probs.pt')
        gt_path = os.path.join(self.yolo_dir, split_name, 'yolo11_ground_truth.pt')
        
        # Check if files exist
        for path in [predictions_path, probs_path, gt_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load cached data
        predictions = torch.load(predictions_path)
        probs = torch.load(probs_path)
        ground_truth = torch.load(gt_path)
        
        # Process the data to create class probabilities and targets
        all_class_probs = []
        all_targets = []
        
        # Process each image
        for img_name in predictions.keys():
            if img_name not in probs or img_name not in ground_truth:
                continue
                
            # Get predictions for this image
            img_preds = predictions[img_name]['boxes']
            img_gt = ground_truth[img_name]['ground_truth']
            
            # Skip images with no predictions or ground truth
            if not img_preds or not img_gt:
                continue
            
            # Find the largest bounding box among the predictions
            largest_bbox_idx = -1
            largest_area = -1
            
            for i, pred in enumerate(img_preds):
                bbox = pred.get('bbox', [0, 0, 0, 0])
                area = self._get_bbox_area(bbox)
                
                if area > largest_area:
                    largest_area = area
                    largest_bbox_idx = i
            
            # Skip if no valid bounding box found
            if largest_bbox_idx == -1:
                continue
            
            # Get the class ID and probability distribution for the largest bounding box
            largest_pred = img_preds[largest_bbox_idx]
            class_id = largest_pred.get('class_id', 0)
            
            # Create a probability distribution for the classes
            class_probs = np.zeros(self._num_classes, dtype=np.float32)
            
            if 'cls' in probs[img_name] and largest_bbox_idx < len(probs[img_name]['cls']):
                if isinstance(probs[img_name]['cls'], torch.Tensor) and probs[img_name]['cls'].dim() == 2:
                    # If we have full class probabilities for each detection
                    cls_probs = probs[img_name]['cls'][largest_bbox_idx]
                    
                    # Make sure we don't exceed the configured number of classes
                    if len(cls_probs) >= self._num_classes:
                        # Only use the first num_classes probabilities
                        class_probs = cls_probs[:self._num_classes].cpu().numpy()
                    else:
                        # If probabilities have fewer classes, pad with zeros or use a one-hot encoding
                        class_probs[class_id] = largest_pred.get('confidence', 1.0)
                else:
                    # If we only have class IDs, use the confidence score
                    if class_id < self._num_classes:
                        class_probs[class_id] = largest_pred.get('confidence', 1.0)
            else:
                # Fallback to one-hot encoding with high confidence
                if class_id < self._num_classes:
                    class_probs[class_id] = largest_pred.get('confidence', 1.0)
            
            # Normalize class probabilities to ensure they sum to 1.0
            if np.sum(class_probs) > 0:
                class_probs = class_probs / np.sum(class_probs)
            else:
                # Fallback to uniform distribution if all zeros
                class_probs = np.ones(self._num_classes, dtype=np.float32) / self._num_classes
            
            # Ensure class_probs has exactly num_classes elements
            if len(class_probs) != self._num_classes:
                logging.warning(f"Class probabilities shape {len(class_probs)} doesn't match num_classes {self._num_classes}, resizing")
                if len(class_probs) > self._num_classes:
                    # Trim extra columns
                    class_probs = class_probs[:self._num_classes]
                else:
                    # Pad with zeros and renormalize
                    padded_probs = np.zeros(self._num_classes, dtype=np.float32)
                    padded_probs[:len(class_probs)] = class_probs
                    class_probs = padded_probs / np.sum(padded_probs) if np.sum(padded_probs) > 0 else np.ones(self._num_classes) / self._num_classes
            
            # Find the ground truth class for the image
            # We'll use the first ground truth annotation or the one that best matches 
            # the largest detected box
            gt_class_id = 0  # Default to class 0
            
            if img_gt:
                # Find ground truth that best overlaps with the largest prediction
                best_match_idx = -1
                best_match_iou = -1
                
                for j, gt_box in enumerate(img_gt):
                    gt_bbox = gt_box.get('bbox', [0, 0, 0, 0])
                    iou = self._compute_iou(largest_pred.get('bbox', [0, 0, 0, 0]), gt_bbox)
                    
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_idx = j
                
                if best_match_idx >= 0 and best_match_iou >= 0.5:
                    # Use matched ground truth class
                    gt_class_id = img_gt[best_match_idx].get('category_id', 0)
                    
                    # Convert COCO category ID to YOLO class ID (0-based index)
                    # COCO IDs are not sequential, but YOLO IDs are 0-78
                    # Ensure the class ID is within valid range for our configured number of classes
                    gt_class_id = min(gt_class_id - 1, self._num_classes - 1)  # Adjust for 0-based index
                    gt_class_id = max(0, gt_class_id)  # Ensure valid class ID
                else:
                    # If no good match, use the predicted class as the target
                    # But ensure it's within valid range
                    gt_class_id = min(class_id, self._num_classes - 1)
            
            # Add to dataset
            all_class_probs.append(class_probs)
            all_targets.append(gt_class_id)
        
        return np.array(all_class_probs, dtype=np.float32), np.array(all_targets, dtype=np.int64)
    
    def _compute_iou(self, bbox1, bbox2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = self._get_bbox_area(bbox1)
        bbox2_area = self._get_bbox_area(bbox2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area
    
    def _create_dataloaders(self):
        """
        Create PyTorch DataLoaders for train/val/cal splits
        """
        # Create datasets
        train_dataset = YOLO11SampleDataset(self.train_probs, self.train_targets)
        val_dataset = YOLO11SampleDataset(self.val_probs, self.val_targets)
        cal_dataset = YOLO11SampleDataset(self.cal_probs, self.cal_targets)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.cal_loader = DataLoader(
            cal_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def get_model(self):
        """
        For YOLO11 datasets, we don't need a model as we're using precomputed outputs.
        Return a dummy model that reshapes inputs to be compatible with the scoring framework.
        
        Returns:
            torch.nn.Module: Dummy model that passes through input class probabilities
        """
        class DummyYOLO11Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                # The input x is already a probability tensor of shape [batch_size, num_classes]
                return x
        
        model = DummyYOLO11Model()
        logging.info(f"Using dummy model for YOLO11 with {self._num_classes} classes")
        return model
    
    def get_dataloaders(self):
        """
        Get train, calibration, and test data loaders
        
        Returns:
            Dict containing 'train', 'calibration', and 'test' dataloaders
        """
        return {
            'train': self.train_loader,
            'calibration': self.cal_loader,
            'test': self.test_loader
        }

def run_dataset(config, dataset_name):
    """
    Run evaluation for YOLO11 dataset using the learnable scoring function
    
    Args:
        config: Global configuration
        dataset_name: Name of the dataset (e.g., 'yolo11')
        
    Returns:
        Results dictionary
    """
    from src.core import ScoringFunction, ScoringFunctionTrainer
    from src.utils import set_seed
    import torch
    
    try:
        logging.info(f"=== Starting evaluation for YOLO11 dataset with learnable scoring function ===")
        
        # Get dataset-specific configuration
        dataset_config = config.copy()
        dataset_config['dataset']['name'] = dataset_name
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create dataset
        dataset = Dataset(dataset_config)
        dataloaders = dataset.get_dataloaders()
        
        # Get the dummy model
        model = dataset.get_model()
        
        # Initialize scoring function
        num_classes = dataset._num_classes
        scoring_fn = ScoringFunction(
            input_dim=1,  # The metrics.py expects 1 as input_dim
            hidden_dims=config['scoring_function']['hidden_dims'],
            output_dim=1,  # Output is a scalar score
            config=config
        ).to(config['device'])
        
        # Create a wrapper for the trainer's _compute_scores method to fix the shape mismatch issue
        original_compute_scores = ScoringFunctionTrainer._compute_scores
        
        def patched_compute_scores(self, inputs, targets=None):
            scores, target_scores, probs = original_compute_scores(self, inputs, targets)
            
            # Check if the number of columns in scores matches the configured number of classes
            if scores.shape[1] != num_classes:
                logging.warning(f"Fixing score shape mismatch: {scores.shape[1]} columns, expected {num_classes}")
                # Resize scores tensor to match the expected number of classes
                if scores.shape[1] > num_classes:
                    # Trim extra columns
                    scores = scores[:, :num_classes]
                    if probs is not None:
                        probs = probs[:, :num_classes]
                        # Renormalize probs
                        probs = probs / probs.sum(dim=1, keepdim=True)
                else:
                    # This case shouldn't happen with our fix, but handle it just in case
                    new_scores = torch.ones((scores.shape[0], num_classes), device=scores.device)
                    new_scores[:, :scores.shape[1]] = scores
                    scores = new_scores
                    
                    if probs is not None:
                        new_probs = torch.zeros((probs.shape[0], num_classes), device=probs.device)
                        new_probs[:, :probs.shape[1]] = probs
                        probs = new_probs / new_probs.sum(dim=1, keepdim=True)
                
                # Recalculate target scores if needed
                if targets is not None and target_scores is not None:
                    target_scores = scores[torch.arange(len(targets)), targets]
            
            return scores, target_scores, probs
        
        # Apply the patch for this dataset only
        ScoringFunctionTrainer._compute_scores = patched_compute_scores
        
        # Initialize trainer
        trainer = ScoringFunctionTrainer(
            base_model=model,
            scoring_fn=scoring_fn,
            train_loader=dataloaders['train'],
            cal_loader=dataloaders['calibration'],
            test_loader=dataloaders['test'],
            device=config['device'],
            config=config
        )
        
        # Training loop
        results = trainer.train(
            num_epochs=config.get('num_epochs', 50),
            target_coverage=config.get('target_coverage', 0.9),
            tau_config=config.get('tau', {'min': 0.1, 'max': 0.9, 'window_size': 5}),
            set_size_config=config.get('set_size', {'target': 1, 'max': 3.0, 'margin': 2}),
            save_dir=config.get('model_dir', 'models/yolo11'),
            plot_dir=config.get('plot_dir', 'plots/yolo11')
        )
        
        # Restore the original method after training
        ScoringFunctionTrainer._compute_scores = original_compute_scores
        
        return results
        
    except Exception as e:
        logging.error(f"Error in YOLO11 dataset evaluation: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            'error': str(e),
            'status': 'failed'
        } 