from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader

class BaseDataset(ABC):
    """Abstract base class for dataset implementations"""
    
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def setup(self):
        """Setup the dataset and create train/val/test/calibration splits"""
        pass
    
    @abstractmethod
    def get_model(self):
        """Return the pretrained base model for this dataset"""
        pass
        
    def get_dataloaders(self):
        """Return dictionary of dataloaders"""
        if not hasattr(self, 'train_loader'):
            self.setup()
            
        return {
            'train': self.train_loader,
            'val': self.val_loader,
            'test': self.test_loader,
            'calibration': self.cal_loader
        }
    
    @property
    def num_classes(self):
        """Return number of classes in the dataset"""
        return self._num_classes 