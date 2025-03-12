"""
Core functionality for learnable scoring functions
"""
from .scoring_function import ScoringFunction
from .trainer import ScoringFunctionTrainer
from .metrics import compute_tau, AverageMeter
from .config import ConfigManager
from .advanced_metrics import (
    calculate_auroc, 
    calculate_auarc,
    calculate_auarc_from_scores,
    plot_roc_curve,
    plot_auarc_curve,
    plot_metrics_over_epochs,
    analyze_epoch_metrics,
    save_metrics_to_csv
)

__all__ = [
    'ScoringFunction',
    'ScoringFunctionTrainer',
    'compute_tau',
    'AverageMeter',
    'ConfigManager',
    'calculate_auroc',
    'calculate_auarc',
    'calculate_auarc_from_scores',
    'plot_roc_curve',
    'plot_auarc_curve',
    'plot_metrics_over_epochs',
    'analyze_epoch_metrics',
    'save_metrics_to_csv'
] 