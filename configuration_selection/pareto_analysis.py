import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_cv_results(file_path):
    """
    Parse the alternating mean/std deviation CSV format from nested cross-validation results.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Parsed results with columns for each metric's mean and std
    """
    
    # Read the raw CSV file
    with open(file_path, 'r') as f:
        lines = [line.strip().replace('\r', '') for line in f.readlines() if line.strip()]
    
    results = []
    current_window = None
    
    # Skip header lines and parse data
    for i in range(2, len(lines)):
        parts = [p.strip(' "') for p in lines[i].split(',')]
        
        window_size = parts[0] if parts[0] else None
        model = parts[1] if parts[1] else None
        
        # Update current window size
        if window_size:
            current_window = window_size
            
        # If we have a model name, this is a mean values row
        if model:
            # Extract mean values
            accuracy_mean = float(parts[2]) if parts[2] else np.nan
            precision_mean = float(parts[3]) if parts[3] else np.nan
            recall_mean = float(parts[4]) if parts[4] else np.nan
            f1_mean = float(parts[5]) if parts[5] else np.nan
            
            # Check if next line contains standard deviations
            if i + 1 < len(lines):
                std_parts = [p.strip(' "') for p in lines[i + 1].split(',')]
                if not std_parts[0] and not std_parts[1]:  # Empty window and model fields
                    accuracy_std = float(std_parts[2]) if std_parts[2] else np.nan
                    precision_std = float(std_parts[3]) if std_parts[3] else np.nan
                    recall_std = float(std_parts[4]) if std_parts[4] else np.nan
                    f1_std = float(std_parts[5]) if std_parts[5] else np.nan
                    
                    results.append({
                        'window_size': current_window,
                        'model': model,
                        'accuracy_mean': accuracy_mean,
                        'accuracy_std': accuracy_std,
                        'precision_mean': precision_mean,
                        'precision_std': precision_std,
                        'recall_mean': recall_mean,
                        'recall_std': recall_std,
                        'f1_mean': f1_mean,
                        'f1_std': f1_std
                    })
    
    return pd.DataFrame(results)

def find_pareto_front(df, mean_col, std_col):
    """
    Find true Pareto optimal solutions (maximize mean, minimize std).
    
    This corrects the original implementation which had a flaw in the algorithm.
    A configuration is Pareto optimal if no other configuration dominates it.
    Configuration A dominates B if: A_mean >= B_mean AND A_std <= B_std, 
    with at least one strict inequality.
    """
    df = df.copy()
    pareto_indices = []
    
    for i, row_i in df.iterrows():
        is_dominated = False
        
        for j, row_j in df.iterrows():
            if i == j:
                continue
                
            # Check if j dominates i
            # j dominates i if j has higher/equal mean AND lower/equal std
            # with at least one strict inequality
            if (row_j[mean_col] >= row_i[mean_col] and 
                row_j[std_col] <= row_i[std_col] and 
                (row_j[mean_col] > row_i[mean_col] or row_j[std_col] < row_i[std_col])):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    pareto_front = df.loc[pareto_indices].copy()
    
    # Sort by mean performance (descending) for better presentation
    pareto_front = pareto_front.sort_values(mean_col, ascending=False)
    
    return pareto_front

def analyze_pareto_selection(df, metric='f1'):
    """
    Complete Pareto frontier analysis for model selection.
    
    Parameters:
    df (pd.DataFrame): Results dataframe with mean and std columns
    metric (str): Base metric name ('f1', 'accuracy', 'recall', 'precision')
    
    Returns:
    dict: Analysis results including Pareto configurations and selection strategies
    """
    
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    # Remove any rows with NaN values
    df_clean = df.dropna(subset=[mean_col, std_col]).copy()
    
    print(f"=== PARETO FRONTIER ANALYSIS FOR {metric.upper()} ===")
    print("=" * 60)
    print(f"Total configurations: {len(df_clean)}")
    
    # Find Pareto frontier
    pareto_configs = find_pareto_front(df_clean, mean_col, std_col)
    print("Using corrected Pareto algorithm")
    
    print(f"Pareto optimal configurations: {len(pareto_configs)}")
    print()
    
    # Add analysis metrics
    pareto_configs = add_selection_metrics(pareto_configs, metric)
    
    # Display Pareto configurations
    print("PARETO OPTIMAL CONFIGURATIONS:")
    print("-" * 40)
    
    for idx, (_, row) in enumerate(pareto_configs.iterrows(), 1):
        config_name = f"{row['window_size']} + {row['model']}"
        mean_val = row[mean_col]
        std_val = row[std_col]
        cv_val = row[f'{metric}_cv']
        
        print(f"{idx:2d}. {config_name}")
        print(f"    {metric.title()}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"    CV: {cv_val:.3f} | Distance from ideal: {row['distance_from_ideal']:.3f}")
        print()
    
    # Selection strategies
    strategies = ['closest_to_ideal', 'highest_mean', 'lowest_std', 'best_cv']
    selections = {}
    
    print("SELECTION STRATEGIES FROM PARETO FRONTIER:")
    print("-" * 50)
    
    for strategy in strategies:
        selected = select_from_pareto(pareto_configs, metric, strategy, top_n=3)
        selections[strategy] = selected
        
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        for i, (_, row) in enumerate(selected.iterrows(), 1):
            config_name = f"{row['window_size']} + {row['model']}"
            mean_val = row[mean_col]
            std_val = row[std_col]
            print(f"  {i}. {config_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    return {
        'pareto_configs': pareto_configs,
        'all_configs': df_clean,
        'selections': selections,
        'metric': metric
    }

def add_selection_metrics(pareto_configs, metric):
    """Add metrics to help with selection within Pareto frontier."""
    
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    # Coefficient of variation (relative stability measure)
    pareto_configs[f'{metric}_cv'] = pareto_configs[std_col] / pareto_configs[mean_col]
    
    # Normalize metrics for distance calculations
    if len(pareto_configs) > 1:
        mean_min, mean_max = pareto_configs[mean_col].min(), pareto_configs[mean_col].max()
        std_min, std_max = pareto_configs[std_col].min(), pareto_configs[std_col].max()
        
        # Avoid division by zero
        if mean_max > mean_min:
            mean_norm = (pareto_configs[mean_col] - mean_min) / (mean_max - mean_min)
        else:
            mean_norm = pd.Series(1.0, index=pareto_configs.index)
            
        if std_max > std_min:
            std_norm = (pareto_configs[std_col] - std_min) / (std_max - std_min)
        else:
            std_norm = pd.Series(0.0, index=pareto_configs.index)
        
        # Distance from ideal point (1, 0) in normalized space
        pareto_configs['distance_from_ideal'] = np.sqrt(
            (1 - mean_norm)**2 + std_norm**2
        )
    else:
        pareto_configs['distance_from_ideal'] = 0.0
    
    # Confidence intervals
    pareto_configs[f'{metric}_lower_95ci'] = (
        pareto_configs[mean_col] - 1.96 * pareto_configs[std_col]
    )
    pareto_configs[f'{metric}_upper_95ci'] = (
        pareto_configs[mean_col] + 1.96 * pareto_configs[std_col]
    )
    
    return pareto_configs

def select_from_pareto(pareto_configs, metric='f1', strategy='closest_to_ideal', top_n=3):
    """
    Select configurations from Pareto frontier using different strategies.
    
    Parameters:
    pareto_configs (pd.DataFrame): Pareto optimal configurations
    metric (str): Base metric name
    strategy (str): Selection strategy
    top_n (int): Number of configurations to select
    
    Returns:
    pd.DataFrame: Selected configurations
    """
    
    df = pareto_configs.copy()
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    if strategy == 'closest_to_ideal':
        df = df.sort_values('distance_from_ideal', ascending=True)
    elif strategy == 'highest_mean':
        df = df.sort_values(mean_col, ascending=False)
    elif strategy == 'lowest_std':
        df = df.sort_values(std_col, ascending=True)
    elif strategy == 'best_cv':
        df = df.sort_values(f'{metric}_cv', ascending=True)
    
    return df.head(top_n)

def visualize_pareto_frontier(results_dict):
    """
    Create visualization of Pareto frontier analysis.
    
    Parameters:
    results_dict (dict): Results from analyze_pareto_selection
    """
    
    pareto_configs = results_dict['pareto_configs']
    all_configs = results_dict['all_configs']
    metric = results_dict['metric']
    
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    plt.figure(figsize=(12, 8))
    
    # Plot all configurations by model type
    colors = {'DT': 'lightcoral', 'RF': 'lightblue', 'XGB': 'lightgreen'}
    
    for model in all_configs['model'].unique():
        model_data = all_configs[all_configs['model'] == model]
        plt.scatter(model_data[std_col], model_data[mean_col], 
                c=colors.get(model, 'gray'), label=f'{model} (All)', 
                alpha=0.6, s=50)
    
    # Highlight Pareto frontier
    pareto_sorted = pareto_configs.sort_values(std_col)
    plt.plot(pareto_sorted[std_col], pareto_sorted[mean_col], 
            'r-', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    # Plot Pareto optimal points
    plt.scatter(pareto_configs[std_col], pareto_configs[mean_col], 
            c='red', s=120, marker='*', alpha=0.9, 
            label=f'Pareto Optimal ({len(pareto_configs)})', 
            edgecolors='darkred', linewidth=1)
    
    # Add annotations for Pareto points
    for _, row in pareto_configs.iterrows():
        config_name = f"{row['window_size']}+{row['model']}"
        plt.annotate(config_name, 
                    (row[std_col], row[mean_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    plt.xlabel(f'{metric.title()} Standard Deviation')
    plt.ylabel(f'{metric.title()} Mean')
    plt.title(f'Pareto Frontier Analysis: {metric.title()} Performance vs Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()    
    plt.show()

# Example usage functions:

def run_pareto_analysis(file_path, metric='f1'):
    """
    Complete function to run Pareto analysis on your CV results.
    
    Parameters:
    file_path (str): Path to your CSV file
    metric (str): Metric to analyze
    
    Returns:
    dict: Complete analysis results
    """
    
    # Parse the data
    print("Parsing cross-validation results...")
    df = parse_cv_results(file_path)
    print(f"Successfully parsed {len(df)} configurations\n")
    
    # Compare algorithms (optional)
    print("Comparing Pareto algorithms...")
    print()
    
    # Run main analysis
    results = analyze_pareto_selection(df, metric)
    
    # Create visualization
    print("Creating visualization...")
    visualize_pareto_frontier(results)
    
    return results

