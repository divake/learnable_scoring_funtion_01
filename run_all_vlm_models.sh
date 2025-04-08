#!/bin/bash

# Script to run conformal prediction analysis on all VLM models for all datasets

# Set up output directories
mkdir -p plots/conformal logs/conformal results/conformal

# Define scoring functions to run
SCORING_FUNCTIONS=("1-p" "APS" "LogMargin" "Sparsemax")

# Run using all scoring functions (combined) on all VLM datasets
echo "Running analysis for all VLM models with all scoring functions on ALL datasets..."
/home/divake/miniconda3/envs/env_cu121/bin/python -m src.datasets.base_conformal_scorers --dataset vlm --scoring all

# Run for each scoring function individually (optional)
# for SCORING in "${SCORING_FUNCTIONS[@]}"; do
#     echo "Running analysis for all VLM models with $SCORING scoring function..."
#     /home/divake/miniconda3/envs/env_cu121/bin/python -m src.datasets.base_conformal_scorers --dataset vlm --scoring "$SCORING"
# done

echo "Analysis complete!"
echo "Results saved to results/conformal/summary/"
echo "Plots saved to plots/conformal/"

# Print path to the summary files
echo ""
echo "Summary files:"
echo "- results/conformal/summary/conformal_summary.json"
echo "- plots/conformal/metrics_summary.csv"
echo "- plots/conformal/vlm_metrics_summary.csv (VLM models only)" 