import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

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

def create_window_size_plots(df, file_path, task_name="2-Class", figsize=(20, 24)):
    """
    Create individual bar plots for each window size showing model performance across metrics.
    
    Parameters:
    df (pd.DataFrame): Parsed results dataframe
    task_name (str): Name of the classification task (e.g., "2-Class", "3-Class")
    figsize (tuple): Figure size for the overall plot
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    
    # Get unique window sizes and sort them logically
    window_sizes = df['window_size'].unique()
    
    # Custom sorting to handle numeric and overlap variants
    def sort_key(ws):
        if '_' in ws:
            base, overlap = ws.split('_')
            return (int(base[:-1]), float(overlap))
        else:
            return (int(ws[:-1]), 0)
    
    window_sizes = sorted(window_sizes, key=sort_key)
    
    # Set up the subplot grid (3 columns, rows as needed)
    n_windows = len(window_sizes)
    n_cols = 3
    n_rows = (n_windows + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define colors for metrics (matching your diagram style)
    colors = {
        'accuracy': '#3498db',    # Blue
        'precision': '#e74c3c',   # Red  
        'recall': '#f39c12',      # Orange
        'f1': '#27ae60'          # Green
    }
    
    # Define model order and colors
    models = ['DT', 'RF', 'XGB']
    model_colors = {
        'DT': '#2c3e50',     # Dark blue-gray
        'RF': '#8e44ad',     # Purple
        'XGB': '#16a085'     # Teal
    }
    
    for idx, window_size in enumerate(window_sizes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Filter data for current window size
        window_data = df[df['window_size'] == window_size].copy()
        
        # Ensure model order
        window_data['model'] = pd.Categorical(window_data['model'], categories=models, ordered=True)
        window_data = window_data.sort_values('model')
        
        # Prepare data for plotting
        metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        std_metrics = ['accuracy_std', 'precision_std', 'recall_std', 'f1_std']
        
        # Set up bar positions
        x = np.arange(len(models))
        width = 0.2
        multiplier = 0
        
        # Plot bars for each metric
        for i, (metric, std_metric, label) in enumerate(zip(metrics, std_metrics, metric_labels)):
            values = window_data[metric].values
            stds = window_data[std_metric].values
            
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, 
                        label=label, 
                        color=colors[metric.replace('_mean', '')],
                        alpha=0.8,
                        capsize=3)
            
            # Add error bars
            ax.errorbar(x + offset, values, yerr=stds, 
                    fmt='none', color='black', alpha=0.6, capsize=3)
            
            # Add value labels on bars
            for j, (bar, val, std) in enumerate(zip(bars, values, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
            
            multiplier += 1
        
        # Customize the subplot
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Performance Score', fontweight='bold')
        ax.set_title(f'Window Size: {window_size}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left', fontsize=9)
        
        # Add subtle background for better readability
        ax.set_facecolor('#fafafa')
    
    # Hide empty subplots
    for idx in range(n_windows, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Add overall title and layout adjustments
    fig.suptitle(f'{task_name} Classification: Model Performance Across Window Sizes', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_path)
    return fig

def create_comparison_summary_plot(df, task_name="2-Class", figsize=(15, 10)):
    """
    Create a summary comparison plot showing best performing configurations.
    
    Parameters:
    df (pd.DataFrame): Parsed results dataframe
    task_name (str): Name of the classification task
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    metrics = [('accuracy_mean', 'Accuracy'), ('precision_mean', 'Precision'), 
               ('recall_mean', 'Recall'), ('f1_mean', 'F1-Score')]
    axes = [ax1, ax2, ax3, ax4]
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
    
    for i, ((metric, title), ax, color) in enumerate(zip(metrics, axes, colors)):
        # Find best configuration for each metric
        best_configs = []
        window_sizes = df['window_size'].unique()
        
        for window_size in window_sizes:
            window_data = df[df['window_size'] == window_size]
            best_idx = window_data[metric].idxmax()
            best_config = window_data.loc[best_idx]
            best_configs.append({
                'window_size': window_size,
                'model': best_config['model'],
                'value': best_config[metric],
                'std': best_config[metric.replace('_mean', '_std')],
                'config_name': f"{window_size}+{best_config['model']}"
            })
        
        best_df = pd.DataFrame(best_configs)
        best_df = best_df.sort_values('value', ascending=True)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(best_df))
        bars = ax.barh(y_pos, best_df['value'], color=color, alpha=0.7)
        
        # Add error bars
        ax.errorbar(best_df['value'], y_pos, xerr=best_df['std'], 
                fmt='none', color='black', alpha=0.6, capsize=3)
        
        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(best_df['config_name'], fontsize=9)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(f'Best {title} by Window Size', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, best_df['value'])):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', ha='left', va='center', fontsize=8)
    
    fig.suptitle(f'{task_name} Classification: Best Configurations Summary', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig

def analyze_window_sizes(file_path_2class, file_path_3class=None):
    """
    Complete analysis function to create all window size visualizations.
    
    Parameters:
    file_path_2class (str): Path to 2-class results CSV
    file_path_3class (str, optional): Path to 3-class results CSV
    
    Returns:
    dict: Dictionary containing all created figures
    """
    
    print("Parsing 2-class classification results...")
    df_2class = parse_cv_results(file_path_2class)
    print(f"Successfully parsed {len(df_2class)} configurations for 2-class task\n")
    
    two_class_file_path = "/home/edumaba/Public/MPhil_Thesis/Code/wear_uropatch/configuration_selection/window_size_plots/all_two_class.png"
    three_class_file_path = "/home/edumaba/Public/MPhil_Thesis/Code/wear_uropatch/configuration_selection/window_size_plots/all_three_class.png"
    # Create 2-class plots
    print("Creating 2-class individual window plots...")
    fig_2class_individual = create_window_size_plots(df_2class, two_class_file_path, "2-Class")
    
    print("Creating 2-class summary plots...")
    # fig_2class_summary = create_comparison_summary_plot(df_2class, "2-Class")
    
    results = {
        '2class_individual': fig_2class_individual,
        # '2class_summary': fig_2class_summary,
        '2class_data': df_2class
    }
    
    # Create 3-class plots if provided
    if file_path_3class:
        print("Parsing 3-class classification results...")
        df_3class = parse_cv_results(file_path_3class)
        print(f"Successfully parsed {len(df_3class)} configurations for 3-class task\n")
        
        print("Creating 3-class individual window plots...")
        fig_3class_individual = create_window_size_plots(df_3class,three_class_file_path, "3-Class")
        
        print("Creating 3-class summary plots...")
        # fig_3class_summary = create_comparison_summary_plot(df_3class, "3-Class")
        
        results.update({
            '3class_individual': fig_3class_individual,
            # '3class_summary': fig_3class_summary,
            '3class_data': df_3class
        })
    
    return results

