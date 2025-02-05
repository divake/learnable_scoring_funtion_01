from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import torch
import numpy as np
from .logger import Logger

class BaseCallback(ABC):
    """Base class for callbacks."""
    
    @abstractmethod
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of an epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of an epoch."""
        pass

class ModelCheckpoint(BaseCallback):
    """Callback to save model checkpoints."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = True,
        mode: str = 'min',
        logger: Optional[Logger] = None
    ):
        """Initialize checkpoint callback.
        
        Args:
            filepath: Path to save the model file
            monitor: Quantity to monitor
            save_best_only: If True, only save when monitored quantity improves
            save_weights_only: If True, save only weights
            mode: One of {'min', 'max'}
            logger: Logger instance
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.logger = logger
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f"ModelCheckpoint mode {mode} is unknown")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.logger:
                self.logger.warning(f'Can save best model only with {self.monitor} available, skipping.')
            return
        
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.logger:
                    self.logger.info(
                        f'Epoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}, saving model to {self.filepath}'
                    )
                self.best = current
                if self.save_weights_only:
                    torch.save(logs['model'].state_dict(), self.filepath)
                else:
                    torch.save(logs['model'], self.filepath)
        else:
            if self.logger:
                self.logger.info(f'Epoch {epoch}: saving model to {self.filepath}')
            if self.save_weights_only:
                torch.save(logs['model'].state_dict(), self.filepath)
            else:
                torch.save(logs['model'], self.filepath) 