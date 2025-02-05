class ConformalPredictionError(Exception):
    """Base exception class for conformal prediction errors."""
    pass

class ConfigurationError(ConformalPredictionError):
    """Raised when there is an error in the configuration."""
    pass

class ModelError(ConformalPredictionError):
    """Raised when there is an error with the model."""
    pass

class DataError(ConformalPredictionError):
    """Raised when there is an error with the data."""
    pass

class TrainingError(ConformalPredictionError):
    """Raised when there is an error during training."""
    pass

class ValidationError(ConformalPredictionError):
    """Raised when there is an error during validation."""
    pass

class CalibrationError(ConformalPredictionError):
    """Raised when there is an error during calibration."""
    pass

class PredictionError(ConformalPredictionError):
    """Raised when there is an error during prediction."""
    pass

class ResourceError(ConformalPredictionError):
    """Raised when there is an error with system resources."""
    pass

class CheckpointError(ConformalPredictionError):
    """Raised when there is an error with model checkpoints."""
    pass

class VisualizationError(ConformalPredictionError):
    """Raised when there is an error during visualization."""
    pass 