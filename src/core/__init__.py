"""
Core functionality for learnable scoring functions
"""
from .scoring_function import ScoringFunction
from .trainer import ScoringFunctionTrainer
from .metrics import compute_tau, AverageMeter
from .config import ConfigManager
from .advanced_metrics import (
    calculate_auroc, 
    plot_roc_curve,
    plot_metrics_over_epochs,
    analyze_epoch_metrics,
    save_metrics_to_csv,
    calculate_ece
)

__all__ = [
    'ScoringFunction',
    'ScoringFunctionTrainer',
    'compute_tau',
    'AverageMeter',
    'ConfigManager',
    'calculate_auroc',
    'calculate_ece',
    'plot_roc_curve',
    'plot_metrics_over_epochs',
    'analyze_epoch_metrics',
    'save_metrics_to_csv'
] 