#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import os

# Read the conformal_summary.json file
print("Reading conformal_summary.json...")
with open('/ssd_4TB/divake/learnable_scoring_funtion_01/results/conformal/summary/conformal_summary.json', 'r') as f:
    conformal_data = json.load(f)

# Initialize lists for storing data
data_rows = []

# Define the datasets and metrics we want to extract
datasets = ['ai2d', 'mmbench', 'oodcv', 'scienceqa', 'seedbench']
metrics = ['empirical_coverage', 'average_set_size', 'auroc']
metrics_names = ['Coverage', 'Set Size', 'AUROC']
scorers = ['1-p', 'APS', 'LogMargin', 'Sparsemax', 'Our']

# Load the data for 'Our' scoring function from the summary files
print("Reading our scoring function data...")
our_data = {}
model_name_mapping = {
    "Yi-VL-6B": "Yi-VL-6B",
    "Qwen-VL-Chat": "Qwen-VL-Chat",
    "mplug-owl2-llama2-7b": "mplug-owl2-llama2-7b",
    "Monkey-Chat": "Monkey-Chat",
    "Monkey": "Monkey",
    "MoE-LLaVA-Phi2-2.7B-4e": "MoE-LLaVA-Phi2-2.7B-4e",
    "llava-v1.6-vicuna-7b": "llava-v1.6-vicuna-7b",
    "llava-v1.6-vicuna-13b": "llava-v1.6-vicuna-13b",
    "llava-v1.6-34b": "llava-v1.6-34b",
    "internlm-xcomposer2-vl-7b": "internlm-xcomposer2-vl-7b",
    "cogagent-vqa-hf": "cogagent-vqa-hf"
}

for dataset in datasets:
    our_data[dataset] = {}
    summary_file = f'/ssd_4TB/divake/learnable_scoring_funtion_01/plots/vlm_all/summary_{dataset}.txt'
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        # Process file by looking for hard-coded model names and patterns
        for model_name in model_name_mapping.keys():
            # Look for each model in the file content
            if model_name in content:
                # Find the line containing this model
                lines = content.split('\n')
                for line in lines:
                    if model_name in line and dataset in line:
                        # Split the line into parts and identify the numeric values
                        # The line format is expected to be:
                        # modelnamehere    dataset   True   0.903   1.477   0.951   0.162   0.058   0.167   0.611
                        try:
                            # Remove leading/trailing whitespace and split by multiple spaces
                            parts = [p for p in line.strip().split() if p]
                            
                            # Verify we have enough parts
                            if len(parts) >= 10:  # model, dataset, success, coverage, size, auroc, ...
                                # Check if the dataset name is correct
                                if parts[1] == dataset:
                                    # Extract values from known positions
                                    # Position 3 is avg_coverage, 4 is avg_set_size, 5 is avg_auroc
                                    coverage = float(parts[3])
                                    set_size = float(parts[4])
                                    auroc = float(parts[5])
                                    
                                    our_data[dataset][model_name] = {
                                        'empirical_coverage': coverage,
                                        'average_set_size': set_size,
                                        'auroc': auroc
                                    }
                                    
                                    print(f"Added data for {model_name} in {dataset}: Coverage={coverage}, Size={set_size}, AUROC={auroc}")
                        except (ValueError, IndexError) as e:
                            print(f"Error processing line for {model_name} in {dataset}: {e}")
                            print(f"Line: {line}")
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")

# Process the conformal data
print("Processing conformal data...")
for model_name, model_data in conformal_data['vlm'].items():
    for dataset, dataset_data in model_data.items():
        for scorer, metrics_data in dataset_data.items():
            # Extract the key metrics
            coverage = metrics_data.get('empirical_coverage', np.nan)
            set_size = metrics_data.get('average_set_size', np.nan)
            auroc = metrics_data.get('auroc', np.nan)
            
            # Add row for this model/dataset/scorer combination
            data_rows.append({
                'Model': model_name,
                'Dataset': dataset,
                'Scorer': scorer,
                'Coverage': coverage,
                'Set Size': set_size,
                'AUROC': auroc
            })
            
# Now add our scoring function data
print("Adding our scoring function data...")
for model_name in conformal_data['vlm'].keys():
    for dataset in datasets:
        if dataset in our_data and model_name in our_data[dataset]:
            # Extract the key metrics for our scoring function
            coverage = our_data[dataset][model_name].get('empirical_coverage', np.nan)
            set_size = our_data[dataset][model_name].get('average_set_size', np.nan)
            auroc = our_data[dataset][model_name].get('auroc', np.nan)
            
            # Add row for our scoring function
            data_rows.append({
                'Model': model_name,
                'Dataset': dataset,
                'Scorer': 'Our',
                'Coverage': coverage,
                'Set Size': set_size,
                'AUROC': auroc
            })

# Create a DataFrame from the data
df = pd.DataFrame(data_rows)

# Pivot the data to create a comprehensive comparison table
# For each dataset-model combination, we'll have 5 scorers and 3 metrics
print("Creating pivot table...")
output_rows = []

for dataset in datasets:
    # Add a row for the dataset header
    output_rows.append({'Model': f'=== {dataset.upper()} ==='})
    
    dataset_df = df[df['Dataset'] == dataset]
    
    for model in dataset_df['Model'].unique():
        model_df = dataset_df[dataset_df['Model'] == model]
        
        # Create a row for this model
        model_row = {'Model': model}
        
        # Add data for each scorer and metric
        for scorer in scorers:
            scorer_data = model_df[model_df['Scorer'] == scorer]
            if not scorer_data.empty:
                for i, metric in enumerate(metrics):
                    metric_name = metrics_names[i]
                    model_row[f'{scorer}_{metric_name}'] = scorer_data[metric_name].values[0]
            else:
                for metric_name in metrics_names:
                    model_row[f'{scorer}_{metric_name}'] = np.nan
        
        output_rows.append(model_row)
    
    # Add a blank row after each dataset
    output_rows.append({'Model': ''})

# Create the final DataFrame
output_df = pd.DataFrame(output_rows)

# Save the table to CSV
print("Saving comparison table to CSV...")
output_file = '/ssd_4TB/divake/learnable_scoring_funtion_01/plots/vlm_all/comparison_table.csv'
output_df.to_csv(output_file, index=False)

# Create a more readable markdown table with proper formatting
print("Creating markdown tables...")
md_lines = []
md_lines.append('# Comparison of Scoring Functions Across VLM Models and Datasets\n')

# Add data rows to markdown
for _, row in output_df.iterrows():
    if row['Model'].startswith('==='):
        # This is a header row
        md_lines.append('\n## ' + row['Model'].replace('===', '').strip() + '\n')
        
        # Add table header after dataset header
        header_parts = []
        header_parts.append('| Model')
        
        # For each scoring function, add the three metrics as columns
        for scorer in scorers:
            for metric in metrics_names:
                header_parts.append(f'{scorer}-{metric}')
        
        header_line = ' | '.join(header_parts) + ' |'
        md_lines.append(header_line)
        
        # Add separator line
        separator = '| ' + ' | '.join(['---'] * (1 + len(scorers) * len(metrics_names))) + ' |'
        md_lines.append(separator)
        
    elif row['Model'] == '':
        # This is a separator row, do nothing
        continue
    else:
        # This is a data row
        row_parts = []
        row_parts.append(f"| {row['Model']}")
        
        for scorer in scorers:
            for i, metric in enumerate(metrics_names):
                col_name = f'{scorer}_{metric}'
                value = row.get(col_name, np.nan)
                
                if pd.isna(value):
                    row_parts.append('')
                elif metric == 'Coverage' or metric == 'AUROC':
                    row_parts.append(f'{value:.3f}')
                else:
                    row_parts.append(f'{value:.2f}')
        
        row_line = ' | '.join(row_parts) + ' |'
        md_lines.append(row_line)

# Save markdown table
md_output_file = '/ssd_4TB/divake/learnable_scoring_funtion_01/plots/vlm_all/comparison_table.md'
with open(md_output_file, 'w') as f:
    f.write('\n'.join(md_lines))

# Also create a summary table with averages across models for each dataset
print("Creating summary table...")
summary_rows = []

for dataset in datasets:
    # Add dataset header
    summary_rows.append({'Dataset': dataset, 'Metric': '---'})
    
    dataset_df = df[df['Dataset'] == dataset]
    
    for metric_name in metrics_names:
        # Calculate average across all models for each scorer
        avg_row = {'Dataset': '', 'Metric': metric_name}
        
        for scorer in scorers:
            scorer_data = dataset_df[dataset_df['Scorer'] == scorer]
            if not scorer_data.empty:
                avg_value = scorer_data[metric_name].mean()
                avg_row[scorer] = avg_value
            else:
                avg_row[scorer] = np.nan
        
        summary_rows.append(avg_row)
    
    # Add a blank row after each dataset
    summary_rows.append({'Dataset': '', 'Metric': ''})

# Create summary DataFrame
summary_df = pd.DataFrame(summary_rows)

# Save summary to CSV
summary_file = '/ssd_4TB/divake/learnable_scoring_funtion_01/plots/vlm_all/summary_comparison.csv'
summary_df.to_csv(summary_file, index=False)

# Create markdown for summary table
md_summary_lines = []
md_summary_lines.append('# Summary Comparison Across Datasets (Average of All Models)\n')

for i, row in summary_df.iterrows():
    if row['Dataset'] and row['Metric'] == '---':
        # This is a dataset header
        md_summary_lines.append(f'\n## {row["Dataset"]}\n')
        
        # Add table header
        header_parts = ['| Metric']
        for scorer in scorers:
            header_parts.append(scorer)
        md_summary_lines.append(' | '.join(header_parts) + ' |')
        
        # Add separator
        md_summary_lines.append('| ' + ' | '.join(['---'] * (1 + len(scorers))) + ' |')
    
    elif row['Dataset'] == '' and row['Metric'] == '':
        # Blank row, skip
        continue
    
    elif row['Metric'] in metrics_names:
        # This is a metric row
        row_parts = [f"| {row['Metric']}"]
        
        for scorer in scorers:
            value = row.get(scorer, np.nan)
            
            if pd.isna(value):
                row_parts.append('')
            elif row['Metric'] == 'Coverage' or row['Metric'] == 'AUROC':
                row_parts.append(f'{value:.3f}')
            else:
                row_parts.append(f'{value:.2f}')
        
        md_summary_lines.append(' | '.join(row_parts) + ' |')

# Save summary markdown
md_summary_file = '/ssd_4TB/divake/learnable_scoring_funtion_01/plots/vlm_all/summary_comparison.md'
with open(md_summary_file, 'w') as f:
    f.write('\n'.join(md_summary_lines))

print(f'Data processing complete. Tables saved to:')
print(f'1. Full comparison table: {output_file} and {md_output_file}')
print(f'2. Summary comparison: {summary_file} and {md_summary_file}') 