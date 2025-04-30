"""
Dataset implementations for different datasets
"""
from .cifar10 import Dataset as CIFAR10Dataset
from .yolo11 import Dataset as YOLO11Dataset

__all__ = ['CIFAR10Dataset', 'YOLO11Dataset'] 