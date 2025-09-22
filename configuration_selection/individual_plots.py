import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

def create_single_window_plot(df, window_size, task_name="2-Class", figsize=(10, 6)):
    """
    Create a single bar plot for one window size showing model performance across metrics.
    
    Parameters:
    df (pd.DataFrame): Parsed results dataframe
    window_size (str): The specific window size to plot (e.g., "1s", "3s_0.5")
    task_name (str): Name of the classification task (e.g., "2-Class", "3-Class")
    figsize (tuple): Figure size for the plot
    
    Returns:
    matplotlib.figure.Figure: The created figure
    """
    
    # Filter data for the specific window size
    window_data = df[df['window_size'] == window_size].copy()
    
    if window_data.empty:
        print(f"No data found for window size: {window_size}")
        return None
    
    # Define colors for metrics
    colors = {
        'accuracy': '#3498db',    # Blue
        'precision': '#e74c3c',   # Red  
        'recall': '#f39c12',      # Orange
        'f1': '#27ae60'          # Green
    }
    
    # Define model order
    models = ['DT', 'RF', 'XGB']
    window_data['model'] = pd.Categorical(window_data['model'], categories=models, ordered=True)
    window_data = window_data.sort_values('model')
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for plotting
    metrics = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    std_metrics = ['accuracy_std', 'precision_std', 'recall_std', 'f1_std']
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.2
    
    # Plot bars for each metric
    for i, (metric, std_metric, label) in enumerate(zip(metrics, std_metrics, metric_labels)):
        values = window_data[metric].values
        stds = window_data[std_metric].values
        
        offset = width * i
        bars = ax.bar(x + offset, values, width, 
                     label=label, 
                     color=colors[metric.replace('_mean', '')],
                     alpha=0.8,
                     capsize=3)
        
        # Add error bars
        ax.errorbar(x + offset, values, yerr=stds, 
                   fmt='none', color='black', alpha=0.6, capsize=3, linewidth=1)
        
        # Add value labels on bars
        for j, (bar, val, std) in enumerate(zip(bars, values, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, 
                   fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Performance Score', fontweight='bold', fontsize=12)
    ax.set_title(f'{task_name} Classification - Window Size: {window_size}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add subtle background for better readability
    ax.set_facecolor('#fafafa')
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def save_individual_window_plots(df, task_name="2-Class", output_dir="window_plots", 
                                dpi=300, format='png', show_plots=True):
    """
    Create and save individual plots for each window size.
    
    Parameters:
    df (pd.DataFrame): Parsed results dataframe
    task_name (str): Name of the classification task
    output_dir (str): Directory to save the plots
    dpi (int): Resolution for saved images
    format (str): Image format ('png', 'pdf', 'svg', etc.)
    show_plots (bool): Whether to display plots after creating them
    
    Returns:
    dict: Dictionary with window sizes as keys and file paths as values
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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
    
    saved_files = {}
    
    print(f"Creating individual plots for {len(window_sizes)} window sizes...")
    print("=" * 60)
    
    for i, window_size in enumerate(window_sizes, 1):
        print(f"[{i:2d}/{len(window_sizes)}] Creating plot for window size: {window_size}")
        
        # Create the plot
        fig = create_single_window_plot(df, window_size, task_name)
        
        if fig is not None:
            # Generate filename
            safe_window_name = window_size.replace('_', '-')
            safe_task_name = task_name.replace('-', '').replace(' ', '').lower()
            filename = f"{safe_task_name}_window_{safe_window_name}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            # Save the plot
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            saved_files[window_size] = filepath
            
            print(f"    ✓ Saved: {filepath}")
            
            # Show plot if requested
            if show_plots:
                plt.show()
            else:
                plt.close(fig)  # Close to save memory
        else:
            print(f"    ✗ Failed to create plot for {window_size}")
    
    print("=" * 60)
    print(f"✓ Successfully created {len(saved_files)} individual plots")
    print(f"✓ All plots saved to: {os.path.abspath(output_dir)}")
    
    return saved_files

def create_performance_summary_table(df, task_name="2-Class"):
    """
    Create a summary table showing the best configurations for each metric.
    
    Parameters:
    df (pd.DataFrame): Parsed results dataframe
    task_name (str): Name of the classification task
    
    Returns:
    pd.DataFrame: Summary table
    """
    
    metrics = [('accuracy_mean', 'Accuracy'), ('precision_mean', 'Precision'), 
               ('recall_mean', 'Recall'), ('f1_mean', 'F1-Score')]
    
    summary_data = []
    
    for metric_col, metric_name in metrics:
        # Find best overall configuration
        best_idx = df[metric_col].idxmax()
        best_config = df.loc[best_idx]
        
        summary_data.append({
            'Metric': metric_name,
            'Best_Score': f"{best_config[metric_col]:.4f}",
            'Std_Dev': f"{best_config[metric_col.replace('_mean', '_std')]:.4f}",
            'Window_Size': best_config['window_size'],
            'Model': best_config['model'],
            'Configuration': f"{best_config['window_size']}+{best_config['model']}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n=== {task_name} CLASSIFICATION - BEST CONFIGURATIONS ===")
    print(summary_df.to_string(index=False))
    print()
    
    return summary_df

def analyze_and_plot_window_sizes(file_path_2class, file_path_3class=None, 
                                 output_dir="window_plots", dpi=300, format='png',
                                 show_plots=False):
    """
    Complete analysis function to create and save all individual window size plots.
    
    Parameters:
    file_path_2class (str): Path to 2-class results CSV
    file_path_3class (str, optional): Path to 3-class results CSV
    output_dir (str): Directory to save plots
    dpi (int): Resolution for saved images
    format (str): Image format
    show_plots (bool): Whether to display plots
    
    Returns:
    dict: Dictionary containing saved file information and summary data
    """
    
    results = {}
    
    # Process 2-class data
    print("Parsing 2-class classification results...")
    df_2class = parse_cv_results(file_path_2class)
    print(f"Successfully parsed {len(df_2class)} configurations for 2-class task\n")
    
    # Create 2-class directory
    output_2class = os.path.join(output_dir, "2class")
    
    # Save individual 2-class plots
    saved_2class = save_individual_window_plots(
        df_2class, "2-Class", output_2class, dpi, format, show_plots
    )
    
    # Create summary
    summary_2class = create_performance_summary_table(df_2class, "2-Class")
    
    results['2class'] = {
        'data': df_2class,
        'saved_files': saved_2class,
        'summary': summary_2class
    }
    
    # Process 3-class data if provided
    if file_path_3class:
        print("\nParsing 3-class classification results...")
        df_3class = parse_cv_results(file_path_3class)
        print(f"Successfully parsed {len(df_3class)} configurations for 3-class task\n")
        
        # Create 3-class directory
        output_3class = os.path.join(output_dir, "3class")
        
        # Save individual 3-class plots
        saved_3class = save_individual_window_plots(
            df_3class, "3-Class", output_3class, dpi, format, show_plots
        )
        
        # Create summary
        summary_3class = create_performance_summary_table(df_3class, "3-Class")
        
        results['3class'] = {
            'data': df_3class,
            'saved_files': saved_3class,
            'summary': summary_3class
        }
    
    return results

