#!/usr/bin/env python
"""
Script to train the MLP scoring function for all VLM models in parallel
"""

import os
import sys
import argparse
import logging
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Add src directory to Python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(src_dir))

from src.core import ConfigManager


def setup_logging(dataset_name):
    """Setup logging configuration"""
    log_dir = 'logs/vlm_all'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'vlm_all_models_{dataset_name}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def run_training_process(model_name, dataset_name, config_path, gpu_id=None):
    """Run a single training process for one model using subprocess"""
    # Use the specified Python executable
    python_executable = "/home/divake/miniconda3/envs/env_cu121/bin/python"
    
    cmd = [
        python_executable, "src/scripts/train_vlm_scoring.py",
        "--config", config_path,
        "--model", model_name,
        "--dataset", dataset_name
    ]
    
    # Only add GPU argument if explicitly provided
    if gpu_id is not None:
        cmd.extend(["--gpu", str(gpu_id)])
    
    # Create log directory
    log_dir = "logs/vlm_all"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use a more descriptive log file name that includes GPU ID
    process_log_file = f"{log_dir}/{model_name}_{dataset_name}_gpu{gpu_id}.log"
    
    logging.info(f"Starting training for {model_name} on dataset {dataset_name} using GPU {gpu_id}")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info(f"Log file: {process_log_file}")
    
    with open(process_log_file, 'w') as log_file:
        # Start process (don't set CUDA_VISIBLE_DEVICES as we directly specify GPU)
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        
    return {
        "model": model_name,
        "dataset": dataset_name,
        "process": process,
        "log_file": process_log_file,
        "gpu_id": gpu_id,
        "start_time": time.time()
    }


def monitor_processes(processes):
    """Monitor running processes and return when all complete"""
    if not processes:
        logging.warning("No processes to monitor")
        return {}
    
    active_processes = len(processes)
    process_status = {i: "running" for i in range(len(processes))}
    
    logging.info(f"Monitoring {active_processes} processes until completion...")
    
    # Show progress every 30 seconds
    last_progress_update = time.time()
    progress_interval = 30  # seconds
    
    while active_processes > 0:
        current_time = time.time()
        if current_time - last_progress_update > progress_interval:
            still_running = [p["model"] for i, p in enumerate(processes) if process_status[i] == "running"]
            logging.info(f"Still waiting on {active_processes} processes: {still_running}")
            last_progress_update = current_time
            
        for i, process_info in enumerate(processes):
            if process_status[i] == "running":
                returncode = process_info["process"].poll()
                if returncode is not None:
                    # Process finished
                    duration = time.time() - process_info["start_time"]
                    
                    if returncode == 0:
                        logging.info(f"✅ Process for {process_info['model']} completed successfully "
                                    f"(Duration: {duration:.1f}s)")
                        process_status[i] = "success"
                    else:
                        logging.error(f"❌ Process for {process_info['model']} failed with exit code {returncode} "
                                     f"(Duration: {duration:.1f}s)")
                        
                        # Log the error from the log file
                        try:
                            with open(process_info["log_file"], 'r') as f:
                                log_content = f.read()
                                # Extract the last few lines that might contain the error
                                last_lines = '\n'.join(log_content.splitlines()[-10:])
                                logging.error(f"Last lines from log for {process_info['model']}:\n{last_lines}")
                        except Exception as e:
                            logging.error(f"Could not read log file for {process_info['model']}: {str(e)}")
                            
                        process_status[i] = "failed"
                    
                    active_processes -= 1
        
        # Don't busy-wait
        if active_processes > 0:
            time.sleep(5)
    
    # Summarize results
    successes = sum(1 for status in process_status.values() if status == "success")
    failures = sum(1 for status in process_status.values() if status == "failed")
    logging.info(f"All processes completed: {successes} successful, {failures} failed")
    
    # Return success/failed status for each process
    return process_status


def collect_results(processes, process_status, output_dir):
    """Collect results from all training processes"""
    all_results = []
    
    for i, process_info in enumerate(processes):
        model_name = process_info["model"]
        dataset_name = process_info["dataset"]
        log_file_path = process_info["log_file"]
        
        logging.info(f"Checking results for {model_name} on {dataset_name}")
        
        # Check if the process completed successfully
        is_success = process_status[i] == "success"
        
        # Read the log file to check for errors or actual results
        log_content = ""
        try:
            with open(log_file_path, 'r') as f:
                log_content = f.read()
        except Exception as e:
            logging.error(f"Error reading log file for {model_name}: {str(e)}")
        
        # Check if the training actually ran (look for specific markers)
        training_ran = False
        error_message = None
        
        # Check for common error patterns
        if "CUDA out of memory" in log_content:
            error_message = "CUDA out of memory error"
            is_success = False
        elif "RuntimeError" in log_content:
            # Find the specific error message
            error_lines = [line for line in log_content.splitlines() if "RuntimeError" in line]
            error_message = error_lines[0] if error_lines else "Runtime error detected"
            is_success = False
        elif "Could not find file" in log_content or "FileNotFoundError" in log_content:
            error_message = "File not found error"
            is_success = False
        
        # Check for specific training completion markers
        if "Training completed!" in log_content and "Average Set Size:" in log_content:
            training_ran = True
        
        # If the process succeeded but didn't complete training, mark as failed
        if is_success and not training_ran and not "VLM dataset prepared with" in log_content:
            logging.warning(f"Process for {model_name} completed but training didn't run correctly")
            is_success = False
            error_message = "Training completed too quickly without results"
        
        # If training seemed to run, extract metrics from log
        result = {
            "model": model_name,
            "dataset": dataset_name,
            "success": is_success
        }
        
        if error_message:
            result["error"] = error_message
            
        if is_success:
            # Try to extract metrics
            metrics_found = False
            
            # Extract metrics from the log file - looking for the averaged metrics
            try:
                # Look for the metrics section we added
                avg_metrics_section = re.search(r"Metrics for epochs with coverage within ±2% of target:(.*?)(?:No epochs had coverage within|Training completed)", log_content, re.DOTALL)
                
                if avg_metrics_section:
                    avg_section_text = avg_metrics_section.group(1)
                    
                    # Extract the averaged metrics
                    metric_patterns = {
                        "avg_coverage": r"Average Coverage:\s+([\d\.]+)",
                        "avg_set_size": r"Average Set Size:\s+([\d\.]+)",
                        "avg_auroc": r"Average AUROC:\s+([\d\.]+)",
                        "avg_auarc": r"Average AUARC:\s+([\d\.]+)",
                        "avg_ece": r"Average ECE:\s+([\d\.]+)",
                        "avg_tau": r"Average Tau:\s+([\d\.]+)",
                        "avg_efficiency": r"Average Efficiency:\s+([\d\.]+)"
                    }
                    
                    for metric_name, pattern in metric_patterns.items():
                        match = re.search(pattern, avg_section_text)
                        if match:
                            value = float(match.group(1))
                            result[metric_name] = value
                            metrics_found = True
                
                # If no avg metrics section found, look for "Closest epoch to target coverage" section
                if not metrics_found:
                    closest_section = re.search(r"Closest epoch to target coverage:.*?Coverage:\s+([\d\.]+).*?Set Size:\s+([\d\.]+)", log_content, re.DOTALL)
                    if closest_section:
                        coverage = float(closest_section.group(1))
                        set_size = float(closest_section.group(2))
                        result["avg_coverage"] = coverage
                        result["avg_set_size"] = set_size
                        
                        # Calculate efficiency from these values
                        if set_size > 0:
                            result["avg_efficiency"] = coverage / set_size
                        
                        metrics_found = True
                
                # If still no metrics, fall back to the older method
                if not metrics_found:
                    logging.warning(f"No averaged metrics found for {model_name}. Looking for standard metrics.")
                    
                    # Legacy metric patterns
                    legacy_patterns = {
                        "coverage": r"Coverage:\s+([\d\.]+)",
                        "avg_set_size": r"Set Size:\s+([\d\.]+)",
                        "auroc": r"AUROC:\s+([\d\.]+)",
                        "auarc": r"AUARC:\s+([\d\.]+)",
                        "ece": r"ECE:\s+([\d\.]+)",
                        "tau": r"Tau:\s+([\d\.]+)"
                    }
                    
                    for metric_name, pattern in legacy_patterns.items():
                        match = re.search(pattern, log_content)
                        if match:
                            value = float(match.group(1))
                            # Convert the keys to match the new metric names
                            if metric_name == "coverage":
                                result["avg_coverage"] = value
                            elif metric_name == "auroc":
                                result["avg_auroc"] = value
                            elif metric_name == "auarc":
                                result["avg_auarc"] = value
                            elif metric_name == "ece":
                                result["avg_ece"] = value
                            elif metric_name == "tau":
                                result["avg_tau"] = value
                            else:
                                result[metric_name] = value
                            metrics_found = True
                    
                    # Calculate efficiency if we have coverage and set_size
                    if "avg_coverage" in result and "avg_set_size" in result and result["avg_set_size"] > 0:
                        result["avg_efficiency"] = result["avg_coverage"] / result["avg_set_size"]
                
                if not metrics_found:
                    logging.warning(f"No metrics found in log for {model_name}")
                    result["success"] = False
                    result["error"] = "No metrics found in log"
                else:
                    logging.info(f"Extracted metrics from log for {model_name}: " + 
                                f"Avg Coverage={result.get('avg_coverage', 'N/A'):.3f}, " +
                                f"Avg Set Size={result.get('avg_set_size', 'N/A'):.3f}, " +
                                f"Avg AUROC={result.get('avg_auroc', 'N/A'):.3f}")
            except Exception as e:
                logging.error(f"Error extracting metrics for {model_name}: {str(e)}")
                result["success"] = False
                result["error"] = f"Error extracting metrics: {str(e)}"
        
        all_results.append(result)
        
    # Save dataset-specific results to file
    dataset_name = processes[0]["dataset"] if processes else "unknown"
    results_file = os.path.join(output_dir, f"{dataset_name}_model_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logging.info(f"Saved results for {dataset_name} to {results_file}")
    
    # Check overall success rate
    successful = sum(1 for r in all_results if r.get('success', False))
    logging.info(f"Successfully extracted metrics for {successful}/{len(all_results)} models on {dataset_name}")
    
    return all_results


def plot_comparison(all_results, output_dir, dataset_name):
    """Generate comparison plots for all models"""
    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if not successful_results:
        logging.error("No successful results to plot!")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(successful_results)
    
    # Sort by set size for better visualization
    df_sorted = df.sort_values('avg_set_size')
    
    # Plot comparison metrics
    metrics = {
        'avg_set_size': 'Average Set Size',
        'avg_coverage': 'Average Coverage (target 90%)',
        'avg_auroc': 'Average AUROC Score',
        'avg_efficiency': 'Average Efficiency',
        'avg_auarc': 'Average AUARC Score',
        'avg_ece': 'Average ECE Score',
        'avg_tau': 'Average Tau Value',
    }
    
    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    for metric, title in metrics.items():
        if metric not in df_sorted.columns:
            continue
            
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_sorted['model'], df_sorted[metric])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f'{title} Comparison Across Models - {dataset_name}')
        plt.xlabel('VLM Model')
        plt.ylabel(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot to dataset-specific directory
        save_path = os.path.join(dataset_dir, f'comparison_{metric}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    # Also save to main directory for backward compatibility
    for metric, title in metrics.items():
        if metric not in df_sorted.columns:
            continue
            
        save_path = os.path.join(output_dir, f'comparison_{metric}_{dataset_name}.png')
        
        # Check if we already have data
        if os.path.exists(save_path):
            logging.info(f"Skipping duplicate plot {save_path} (already generated)")
            continue
            
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df_sorted['model'], df_sorted[metric])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', rotation=0)
        
        plt.title(f'{title} Comparison Across Models - {dataset_name}')
        plt.xlabel('VLM Model')
        plt.ylabel(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    # Return the sorted DataFrame for summary statistics
    return df_sorted


def generate_summary(df_sorted, output_dir, dataset_name):
    """Generate summary of best performing models"""
    if df_sorted is None or len(df_sorted) == 0:
        return "No successful training runs to summarize."
    
    # Find best models by different metrics
    target_coverage = 0.9
    
    # Skip calculations if columns don't exist
    metrics_summary = {}
    
    if 'avg_set_size' in df_sorted.columns:
        best_setsize = df_sorted.loc[df_sorted['avg_set_size'].idxmin()]
        metrics_summary['best_set_size'] = {
            'model': best_setsize['model'],
            'value': best_setsize['avg_set_size']
        }
    
    if 'avg_coverage' in df_sorted.columns:
        best_coverage = df_sorted.loc[(df_sorted['avg_coverage'] - target_coverage).abs().idxmin()]
        metrics_summary['best_coverage'] = {
            'model': best_coverage['model'],
            'value': best_coverage['avg_coverage']
        }
    
    if 'avg_auroc' in df_sorted.columns:
        best_auroc = df_sorted.loc[df_sorted['avg_auroc'].idxmax()]
        metrics_summary['best_auroc'] = {
            'model': best_auroc['model'],
            'value': best_auroc['avg_auroc']
        }
        
    if 'avg_efficiency' in df_sorted.columns:
        best_efficiency = df_sorted.loc[df_sorted['avg_efficiency'].idxmax()]
        metrics_summary['best_efficiency'] = {
            'model': best_efficiency['model'],
            'value': best_efficiency['avg_efficiency']
        }
    
    # Format summary text
    summary = f"SUMMARY FOR DATASET: {dataset_name}\n"
    summary += "=" * 50 + "\n\n"
    
    summary += "Best Models:\n"
    if 'best_set_size' in metrics_summary:
        model = metrics_summary['best_set_size']['model']
        size = metrics_summary['best_set_size']['value']
        coverage = df_sorted.loc[df_sorted['model'] == model, 'avg_coverage'].values[0] if 'avg_coverage' in df_sorted.columns else 'N/A'
        auroc = df_sorted.loc[df_sorted['model'] == model, 'avg_auroc'].values[0] if 'avg_auroc' in df_sorted.columns else 'N/A'
        summary += f"1. Minimum Set Size: {model} (Size: {size:.3f}, Coverage: {coverage if isinstance(coverage, str) else coverage:.3f}, AUROC: {auroc if isinstance(auroc, str) else auroc:.3f})\n"
    
    if 'best_coverage' in metrics_summary:
        model = metrics_summary['best_coverage']['model']
        coverage = metrics_summary['best_coverage']['value']
        size = df_sorted.loc[df_sorted['model'] == model, 'avg_set_size'].values[0] if 'avg_set_size' in df_sorted.columns else 'N/A'
        auroc = df_sorted.loc[df_sorted['model'] == model, 'avg_auroc'].values[0] if 'avg_auroc' in df_sorted.columns else 'N/A'
        summary += f"2. Closest to Target Coverage (90%): {model} (Coverage: {coverage:.3f}, Size: {size if isinstance(size, str) else size:.3f}, AUROC: {auroc if isinstance(auroc, str) else auroc:.3f})\n"
    
    if 'best_auroc' in metrics_summary:
        model = metrics_summary['best_auroc']['model']
        auroc = metrics_summary['best_auroc']['value']
        size = df_sorted.loc[df_sorted['model'] == model, 'avg_set_size'].values[0] if 'avg_set_size' in df_sorted.columns else 'N/A'
        coverage = df_sorted.loc[df_sorted['model'] == model, 'avg_coverage'].values[0] if 'avg_coverage' in df_sorted.columns else 'N/A'
        summary += f"3. Best AUROC: {model} (AUROC: {auroc:.3f}, Size: {size if isinstance(size, str) else size:.3f}, Coverage: {coverage if isinstance(coverage, str) else coverage:.3f})\n"
    
    if 'best_efficiency' in metrics_summary:
        model = metrics_summary['best_efficiency']['model']
        efficiency = metrics_summary['best_efficiency']['value']
        size = df_sorted.loc[df_sorted['model'] == model, 'avg_set_size'].values[0] if 'avg_set_size' in df_sorted.columns else 'N/A'
        coverage = df_sorted.loc[df_sorted['model'] == model, 'avg_coverage'].values[0] if 'avg_coverage' in df_sorted.columns else 'N/A'
        summary += f"4. Best Efficiency: {model} (Efficiency: {efficiency:.3f}, Size: {size if isinstance(size, str) else size:.3f}, Coverage: {coverage if isinstance(coverage, str) else coverage:.3f})\n"
    
    summary += "\nAll Models Performance:\n"
    summary += df_sorted.to_string(index=False, float_format=lambda x: f"{x:.3f}")
    
    # Create dataset-specific directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save summary to dataset-specific directory
    summary_file = os.path.join(dataset_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Also save to main directory for backward compatibility
    main_summary_file = os.path.join(output_dir, f'summary_{dataset_name}.txt')
    with open(main_summary_file, 'w') as f:
        f.write(summary)
    
    logging.info(f"Saved summary to {summary_file} and {main_summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description='Train scoring function for all VLM models in parallel')
    parser.add_argument('--config', type=str, default='src/config/vlm.yaml',
                      help='Path to config file (default: src/config/vlm.yaml)')
    parser.add_argument('--dataset', type=str, default=None,
                      help='Dataset to use (default: all configured datasets)')
    parser.add_argument('--gpus', type=str, default='0',
                      help='Comma-separated list of GPU IDs to use (overrides config, default: 0)')
    parser.add_argument('--max_parallel', type=int, default=4,
                      help='Maximum number of parallel jobs (default: 4)')
    parser.add_argument('--output_dir', type=str, default='plots/vlm_all',
                      help='Directory to save comparison results (default: plots/vlm_all)')
    args = parser.parse_args()
    
    # Load config to get models list and default GPU
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Get available GPUs - either from command line or from config
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
        logging.info(f"Using GPUs from command line: {gpu_ids}")
    else:
        # Use the device from config
        config_gpu = config.get('device')
        if isinstance(config_gpu, int):
            gpu_ids = [config_gpu]
        elif isinstance(config_gpu, str) and config_gpu.startswith('cuda:'):
            gpu_ids = [int(config_gpu.split(':')[1])]
        else:
            gpu_ids = [0]  # Default to GPU 0 as requested
        logging.info(f"Using GPUs from config: {gpu_ids}")
    
    # Get list of models from config
    models = config['dataset']['models']
    logging.info(f"Found {len(models)} models to train: {models}")
    
    # Get list of datasets to use
    if args.dataset:
        datasets = [args.dataset]  # Use the specified dataset
    else:
        # Use all supported datasets from config
        datasets = config['dataset'].get('datasets', ['ai2d'])
        # Default to ai2d if no datasets are configured
        if not datasets:
            datasets = ['ai2d']
    
    logging.info(f"Running training for datasets: {datasets}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Train each dataset separately
    for dataset_name in datasets:
        logging.info(f"\n\n{'='*50}")
        logging.info(f"STARTING TRAINING FOR DATASET: {dataset_name}")
        logging.info(f"{'='*50}\n")
        
        # Setup logging for this dataset
        log_file = setup_logging(dataset_name)
        
        # Limit parallel processes based on the number of GPUs and max_parallel setting
        max_jobs = min(args.max_parallel, len(gpu_ids))
        logging.info(f"Running up to {max_jobs} parallel jobs with {len(gpu_ids)} GPUs")
        
        # Start processes for each model in batches
        all_processes = []
        active_processes = []
        
        for i, model_name in enumerate(models):
            # Use round-robin GPU assignment if multiple GPUs are available
            gpu_id = gpu_ids[i % len(gpu_ids)] if len(gpu_ids) > 0 else None
            
            # Start the process for this dataset
            process_info = run_training_process(
                model_name=model_name,
                dataset_name=dataset_name,
                config_path=args.config,
                gpu_id=gpu_id
            )
            
            all_processes.append(process_info)
            active_processes.append(process_info)
            
            # Short delay to avoid race conditions when creating files
            time.sleep(1)
            
            # If we've reached the maximum number of parallel jobs,
            # wait for some to complete before starting more
            if len(active_processes) >= max_jobs and i < len(models) - 1:
                logging.info(f"Reached max parallel jobs ({max_jobs}). Waiting for processes to complete...")
                
                # Wait for at least one process to complete
                while True:
                    completed = [p for p in active_processes if p["process"].poll() is not None]
                    if completed:
                        # At least one process completed
                        for p in completed:
                            active_processes.remove(p)
                            exit_code = p["process"].returncode
                            duration = time.time() - p["start_time"]
                            if exit_code == 0:
                                logging.info(f"✅ Process for {p['model']} completed successfully "
                                            f"(Duration: {duration:.1f}s)")
                            else:
                                logging.error(f"❌ Process for {p['model']} failed with exit code {exit_code} "
                                            f"(Duration: {duration:.1f}s)")
                        break
                    
                    time.sleep(5)  # Check every 5 seconds
        
        # Monitor remaining processes until completion
        process_status = monitor_processes(all_processes)
        
        # Collect results from all processes for this dataset
        all_results = collect_results(all_processes, process_status, output_dir)
        
        # Generate comparison plots for this dataset
        df_sorted = plot_comparison(all_results, output_dir, dataset_name)
        
        # Generate summary for this dataset
        summary = generate_summary(df_sorted, output_dir, dataset_name)
        logging.info("\n" + summary)
    
    # Complete
    logging.info(f"All training completed for all datasets!")
    logging.info(f"Check the logs directory for logs and {output_dir} for results.")


if __name__ == '__main__':
    main() 