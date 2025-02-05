import logging
import os
from typing import Optional
from datetime import datetime

class Logger:
    """Structured logging utility for the project."""
    
    def __init__(
        self,
        log_file: str,
        name: str = "ConformalPrediction",
        level: int = logging.INFO,
        log_to_console: bool = True
    ) -> None:
        """Initialize logger with file and console handlers.
        
        Args:
            log_file: Path to the log file
            name: Logger name
            level: Logging level
            log_to_console: Whether to also log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        if log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def log_hyperparameters(self, config: dict) -> None:
        """Log hyperparameters at the start of training.
        
        Args:
            config: Dictionary containing hyperparameters
        """
        self.info("=== Hyperparameters ===")
        for section, params in config.items():
            self.info(f"\n[{section}]")
            if isinstance(params, dict):
                for key, value in params.items():
                    self.info(f"{key}: {value}")
            else:
                self.info(f"{section}: {params}")
    
    def log_metrics(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: Optional[dict] = None,
        prefix: str = ""
    ) -> None:
        """Log training and validation metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Optional dictionary of validation metrics
            prefix: Optional prefix for metric names
        """
        self.info(f"\n=== Epoch {epoch} ===")
        
        # Log training metrics
        self.info("Training Metrics:")
        for metric, value in train_metrics.items():
            self.info(f"{prefix}{metric}: {value:.4f}")
        
        # Log validation metrics
        if val_metrics:
            self.info("\nValidation Metrics:")
            for metric, value in val_metrics.items():
                self.info(f"{prefix}{metric}: {value:.4f}")
    
    def log_system_info(self, config) -> None:
        """Log system information at startup.
        
        Args:
            config: Configuration object containing system info
        """
        self.info("\n=== System Information ===")
        self.info(f"Device: {config.device}")
        self.info(f"Base Directory: {config.base_dir}")
        self.info(f"Data Directory: {config.data_dir}")
        self.info(f"Model Directory: {config.model_dir}")
        self.info(f"Plot Directory: {config.plot_dir}")
        self.info(f"Experiment ID: {self.experiment_id}")
    
    def log_model_summary(self, model) -> None:
        """Log model architecture summary.
        
        Args:
            model: PyTorch model
        """
        self.info("\n=== Model Architecture ===")
        self.info(str(model))
        self.info(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")
        self.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") 