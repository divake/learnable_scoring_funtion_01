"""
Main script for running conformal prediction evaluation.

This allows direct invocation of the base_conformal_scorers.py script with:
python -m src.datasets.base_conformal_scorers --config src/config/base_conformal_scorers.yaml
"""

from src.datasets.base_conformal_scorers import main

if __name__ == "__main__":
    main() 