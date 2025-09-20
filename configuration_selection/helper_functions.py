import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def analyze_best_configurations(df, primary_metric='f1_mean', top_n=10):
    """
    Analyze and display the best performing configurations.
    
    Parameters:
    df (pd.DataFrame): Results dataframe from parse_cv_results
    primary_metric (str): Metric to optimize for ('f1_mean', 'accuracy_mean', 'recall_mean')
    top_n (int): Number of top configurations to display
    
    Returns:
    pd.DataFrame: Top configurations sorted by primary metric
    """
    
    # Calculate stability metrics
    df = df.copy()
    df['f1_lower_bound'] = df['f1_mean'] - 2 * df['f1_std']
    df['recall_lower_bound'] = df['recall_mean'] - 2 * df['recall_std']
    df['accuracy_lower_bound'] = df['accuracy_mean'] - 2 * df['accuracy_std']
    
    # Sort by primary metric
    top_configs = df.nlargest(top_n, primary_metric)
    
    print(f"=== TOP {top_n} CONFIGURATIONS BY {primary_metric.upper()} ===")
    print("-" * 80)
    
    for idx, (i, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{idx:2d}. {row['window_size']} + {row['model']}")
        print(f"    F1:       {row['f1_mean']:.4f} ± {row['f1_std']:.4f} (95% CI: >{row['f1_lower_bound']:.4f})")
        print(f"    Accuracy: {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}")
        print(f"    Recall:   {row['recall_mean']:.4f} ± {row['recall_std']:.4f}")
        print()
    
    return top_configs

def find_absolute_best(df):
    """Find the single best configuration for each metric."""
    
    best_configs = {}
    metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    
    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_configs[metric] = df.loc[best_idx]
    
    print("=== ABSOLUTE BEST PERFORMERS ===")
    print("-" * 50)
    
    for metric, config in best_configs.items():
        metric_name = metric.replace('_mean', '').title()
        print(f"Best {metric_name}: {config['window_size']} + {config['model']}")
        print(f"  Value: {config[metric]:.4f} ± {config[metric.replace('_mean', '_std')]:.4f}")
        print()
    
    return best_configs

def analyze_window_patterns(df):
    """Analyze performance patterns by window size and overlap."""
    
    # Extract base window and overlap information
    df = df.copy()
    df['base_window'] = df['window_size'].str.extract(r'(\d+s)')[0]
    df['overlap'] = df['window_size'].str.extract(r'_(\d+\.?\d*)')[0]
    df['overlap'] = df['overlap'].fillna('no_overlap')
    
    print("=== WINDOW SIZE ANALYSIS ===")
    print("-" * 40)
    
    # Best model for each window size
    window_analysis = df.groupby('window_size').agg({
        'f1_mean': ['mean', 'max', 'idxmax'],
        'f1_std': 'mean'
    }).round(4)
    
    for window_size in df['window_size'].unique():
        window_data = df[df['window_size'] == window_size]
        best_config = window_data.loc[window_data['f1_mean'].idxmax()]
        avg_f1 = window_data['f1_mean'].mean()
        
        print(f"{window_size}:")
        print(f"  Best: {best_config['model']} (F1: {best_config['f1_mean']:.4f} ± {best_config['f1_std']:.4f})")
        print(f"  Average F1 across models: {avg_f1:.4f}")
        print()
    
    print("=== OVERLAP EFFECT ANALYSIS ===")
    print("-" * 40)
    
    overlap_analysis = df.groupby('overlap').agg({
        'f1_mean': 'mean',
        'recall_mean': 'mean',
        'accuracy_mean': 'mean'
    }).round(4)
    
    for overlap in overlap_analysis.index:
        row = overlap_analysis.loc[overlap]
        overlap_name = 'No overlap' if overlap == 'no_overlap' else f'{overlap} overlap'
        count = len(df[df['overlap'] == overlap])
        print(f"{overlap_name}: Avg F1={row['f1_mean']:.4f}, Avg Recall={row['recall_mean']:.4f} ({count} configs)")

def create_visualizations(df):
    """Create visualizations for the results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Heatmap of F1 scores by window size and model
    pivot_f1 = df.pivot(index='window_size', columns='model', values='f1_mean')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('F1-Score by Window Size and Model')
    axes[0,0].set_ylabel('Window Size')
    
    # 2. Bar plot of top 10 configurations
    top_10 = df.nlargest(10, 'f1_mean')
    top_10['config'] = top_10['window_size'] + ' + ' + top_10['model']
    axes[0,1].barh(range(len(top_10)), top_10['f1_mean'], 
                   xerr=top_10['f1_std'], capsize=3)
    axes[0,1].set_yticks(range(len(top_10)))
    axes[0,1].set_yticklabels(top_10['config'], fontsize=8)
    axes[0,1].set_xlabel('F1-Score')
    axes[0,1].set_title('Top 10 Configurations with Error Bars')
    axes[0,1].invert_yaxis()
    
    # 3. Scatter plot: F1 mean vs std (stability analysis)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        axes[1,0].scatter(model_data['f1_std'], model_data['f1_mean'], 
                         label=model, alpha=0.7, s=60)
    axes[1,0].set_xlabel('F1 Standard Deviation')
    axes[1,0].set_ylabel('F1 Mean')
    axes[1,0].set_title('Performance vs Stability')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Box plot of F1 scores by model
    df_melted = df.melt(id_vars=['model'], value_vars=['f1_mean'], 
                        value_name='f1_score')
    sns.boxplot(data=df, x='model', y='f1_mean', ax=axes[1,1])
    axes[1,1].set_title('F1-Score Distribution by Model')
    axes[1,1].set_ylabel('F1-Score')
    
    plt.tight_layout()
    plt.show()

def comprehensive_analysis(file_path, primary_metric='f1_mean'):
    """
    Run complete analysis of cross-validation results.
    
    Parameters:
    file_path (str): Path to the CSV file
    primary_metric (str): Primary metric to optimize for
    
    Returns:
    pd.DataFrame: Parsed results dataframe
    """
    
    # Parse the data
    print("Parsing cross-validation results...")
    df = parse_cv_results(file_path)
    print(f"Successfully parsed {len(df)} configurations")
    print()
    
    # Find absolute best performers
    best_configs = find_absolute_best(df)
    
    # Analyze top configurations
    top_configs = analyze_best_configurations(df, primary_metric, top_n=8)
    
    # Analyze window patterns
    analyze_window_patterns(df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    return df, top_configs, best_configs

# Example usage:
# df, top_configs, best_configs = comprehensive_analysis('3class nested crossvalidation.csv')

# For more focused analysis:
# df = parse_cv_results('3class nested crossvalidation.csv')
# top_configs = analyze_best_configurations(df, primary_metric='recall_mean', top_n=5)