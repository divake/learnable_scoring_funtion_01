import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from .logger import Logger
from .exceptions import ConfigurationError
import numpy as np

class Experiment:
    """Experiment tracking and management."""
    
    def __init__(
        self,
        name: str,
        base_dir: str,
        config: Dict[str, Any],
        logger: Optional[Logger] = None
    ) -> None:
        """Initialize experiment.
        
        Args:
            name: Experiment name
            base_dir: Base directory for experiments
            config: Configuration dictionary
            logger: Logger instance
        """
        self.name = name
        self.base_dir = base_dir
        self.config = config
        self.logger = logger
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{name}_{self.timestamp}"
        
        # Setup experiment directory
        self.experiment_dir = os.path.join(base_dir, 'experiments', self.experiment_id)
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(self.experiment_dir, 'logs')
        self.plot_dir = os.path.join(self.experiment_dir, 'plots')
        
        # Create directories
        for directory in [self.experiment_dir, self.checkpoint_dir, self.log_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save configuration
        self.save_config()
        
        if self.logger:
            self.logger.info(f"Created experiment: {self.experiment_id}")
            self.logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize configuration, handling non-serializable objects.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            dict: Serializable configuration dictionary
        """
        serialized = {}
        for key, value in config.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_config(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [self._serialize_value(item) for item in value]
            else:
                serialized[key] = self._serialize_value(value)
        return serialized
    
    def _serialize_value(self, value: Any) -> Any:
        """Helper method to serialize individual values.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value
        """
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        elif isinstance(value, (np.float32, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif str(type(value).__name__) == 'device':
            return str(value)
        elif hasattr(value, '__dict__'):
            return str(value)
        return value
    
    def save_config(self) -> None:
        """Save experiment configuration to JSON file."""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        try:
            serialized_config = self._serialize_config(self.config)
            with open(config_path, 'w') as f:
                json.dump(serialized_config, f, indent=4)
            if self.logger:
                self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save configuration: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    def get_checkpoint_path(self, name: str) -> str:
        """Get path for model checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            str: Full path to checkpoint file
        """
        return os.path.join(self.checkpoint_dir, f"{name}.pth")
    
    def get_plot_path(self, name: str) -> str:
        """Get path for plot file.
        
        Args:
            name: Plot name
            
        Returns:
            str: Full path to plot file
        """
        return os.path.join(self.plot_dir, f"{name}.png")
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str = 'metrics.json') -> None:
        """Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filename: Name of metrics file
        """
        metrics_path = os.path.join(self.experiment_dir, filename)
        try:
            serialized_metrics = self._serialize_config(metrics)
            with open(metrics_path, 'w') as f:
                json.dump(serialized_metrics, f, indent=4)
            if self.logger:
                self.logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save metrics: {str(e)}")
            raise ConfigurationError(f"Failed to save metrics: {str(e)}")
    
    def load_metrics(self, filename: str = 'metrics.json') -> Dict[str, Any]:
        """Load metrics from JSON file.
        
        Args:
            filename: Name of metrics file
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics_path = os.path.join(self.experiment_dir, filename)
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load metrics: {str(e)}")
            raise ConfigurationError(f"Failed to load metrics: {str(e)}")
    
    def archive(self) -> None:
        """Archive experiment directory."""
        archive_dir = os.path.join(self.base_dir, 'archived_experiments')
        os.makedirs(archive_dir, exist_ok=True)
        
        try:
            shutil.make_archive(
                os.path.join(archive_dir, self.experiment_id),
                'zip',
                self.experiment_dir
            )
            if self.logger:
                self.logger.info(f"Archived experiment to {archive_dir}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to archive experiment: {str(e)}")
            raise ConfigurationError(f"Failed to archive experiment: {str(e)}")
    
    def clean(self) -> None:
        """Clean up experiment directory."""
        try:
            shutil.rmtree(self.experiment_dir)
            if self.logger:
                self.logger.info(f"Cleaned up experiment directory: {self.experiment_dir}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to clean experiment directory: {str(e)}")
            raise ConfigurationError(f"Failed to clean experiment directory: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of experiment."""
        return f"Experiment(id={self.experiment_id}, dir={self.experiment_dir})" 