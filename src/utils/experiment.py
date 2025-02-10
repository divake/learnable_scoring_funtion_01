import os
from typing import Dict, Any, Optional
from .logger import Logger
from .exceptions import ConfigurationError

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
        
        # Use fixed directories instead of creating new ones for each run
        self.model_dir = os.path.join(base_dir, 'models')
        self.plot_dir = os.path.join(base_dir, 'plots')
        self.log_dir = os.path.join(base_dir, 'logs')  # Keep reference but don't create
        
        # Create only model and plot directories
        for directory in [self.model_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Using fixed directories for experiment")
            self.logger.info(f"Model directory: {self.model_dir}")
            self.logger.info(f"Plot directory: {self.plot_dir}")
            self.logger.info(f"Log directory: {self.log_dir}")
    
    def get_checkpoint_path(self, name: str) -> str:
        """Get path for model checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            str: Full path to checkpoint file
        """
        return os.path.join(self.model_dir, f"{name}.pth")
    
    def get_plot_path(self, name: str) -> str:
        """Get path for plot file.
        
        Args:
            name: Plot name
            
        Returns:
            str: Full path to plot file
        """
        return os.path.join(self.plot_dir, f"{name}.png")
    
    def __str__(self) -> str:
        """String representation of experiment."""
        return f"Experiment(name={self.name}, base_dir={self.base_dir})" 