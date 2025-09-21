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

def rank_based_selection(df, metric='f1', top_n=10, ranking_method='combined_rank'):
    """
    Select best configurations based on both mean performance and stability (low std).
    
    Parameters:
    df (pd.DataFrame): Results dataframe from parse_cv_results
    metric (str): Base metric name ('f1', 'accuracy', 'recall', 'precision')
    top_n (int): Number of top configurations to return
    ranking_method (str): Method for combining rankings
        - 'combined_rank': Sum of mean rank + std rank
        - 'weighted_score': Weighted combination of normalized mean and inverted std
        - 'pareto_front': Pareto optimal solutions
        - 'top_intersection': Intersection of top performers in both criteria
    
    Returns:
    pd.DataFrame: Selected configurations with ranking information
    """
    
    df = df.copy()
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    # Remove any rows with NaN values in the target metric
    df = df.dropna(subset=[mean_col, std_col])
    
    if ranking_method == 'combined_rank':
        # Rank by mean (descending) and std (ascending)
        df[f'{metric}_mean_rank'] = df[mean_col].rank(ascending=False, method='min')
        df[f'{metric}_std_rank'] = df[std_col].rank(ascending=True, method='min')
        
        # Combined rank (lower is better)
        df['combined_rank'] = df[f'{metric}_mean_rank'] + df[f'{metric}_std_rank']
        df = df.sort_values('combined_rank')
        
        ranking_col = 'combined_rank'
        
    elif ranking_method == 'weighted_score':
        # Normalize mean to 0-1 (higher is better)
        df[f'{metric}_mean_norm'] = (df[mean_col] - df[mean_col].min()) / (df[mean_col].max() - df[mean_col].min())
        
        # Normalize std to 0-1 and invert (lower std = higher score)
        df[f'{metric}_std_norm'] = 1 - (df[std_col] - df[std_col].min()) / (df[std_col].max() - df[std_col].min())
        
        # Weighted combination (you can adjust these weights)
        mean_weight = 0.7
        std_weight = 0.3
        df['weighted_score'] = mean_weight * df[f'{metric}_mean_norm'] + std_weight * df[f'{metric}_std_norm']
        df = df.sort_values('weighted_score', ascending=False)
        
        ranking_col = 'weighted_score'
        
    elif ranking_method == 'top_intersection':
        # Find intersection of top performers in both criteria
        top_mean = df.nlargest(top_n * 2, mean_col).index
        top_std = df.nsmallest(top_n * 2, std_col).index
        
        # Intersection
        intersection_idx = list(set(top_mean) & set(top_std))
        df_intersection = df.loc[intersection_idx].copy()
        
        if len(df_intersection) < top_n:
            print(f"Warning: Only {len(df_intersection)} configurations in intersection. Expanding search...")
            # Expand search if intersection is too small
            top_mean = df.nlargest(top_n * 3, mean_col).index
            top_std = df.nsmallest(top_n * 3, std_col).index
            intersection_idx = list(set(top_mean) & set(top_std))
            df_intersection = df.loc[intersection_idx].copy()
        
        # Sort by mean performance within intersection
        df = df_intersection.sort_values(mean_col, ascending=False)
        ranking_col = mean_col
        
    elif ranking_method == 'pareto_front':
        # Find Pareto optimal solutions (maximize mean, minimize std)
        df_pareto = find_pareto_front(df, mean_col, std_col)
        df = df_pareto.sort_values(mean_col, ascending=False)
        ranking_col = mean_col
    
    # Select top N
    selected = df.head(top_n).copy()
    
    # Add confidence intervals
    selected[f'{metric}_lower_95ci'] = selected[mean_col] - 1.96 * selected[std_col]
    selected[f'{metric}_upper_95ci'] = selected[mean_col] + 1.96 * selected[std_col]
    
    return selected, ranking_col

def find_pareto_front(df, mean_col, std_col):
    """Find Pareto optimal solutions (maximize mean, minimize std)."""
    df_sorted = df.sort_values([mean_col, std_col], ascending=[False, True])
    
    pareto_front = []
    current_min_std = float('inf')
    
    for _, row in df_sorted.iterrows():
        if row[std_col] <= current_min_std:
            pareto_front.append(row.name)
            current_min_std = row[std_col]
    
    return df.loc[pareto_front]

def analyze_dual_criteria_selection(df, metric='f1', top_n=10):
    """
    Compare different selection methods for balancing mean performance and stability.
    
    Parameters:
    df (pd.DataFrame): Results dataframe
    metric (str): Metric to analyze
    top_n (int): Number of configurations to select
    
    Returns:
    dict: Results from different ranking methods
    """
    
    methods = ['combined_rank', 'weighted_score', 'top_intersection', 'pareto_front']
    results = {}
    
    print(f"=== DUAL-CRITERIA SELECTION ANALYSIS FOR {metric.upper()} ===")
    print("=" * 60)
    
    for method in methods:
        print(f"\n--- {method.upper().replace('_', ' ')} METHOD ---")
        print("-" * 40)
        
        try:
            selected, ranking_col = rank_based_selection(df, metric, top_n, method)
            results[method] = selected
            
            print(f"Selected {len(selected)} configurations:")
            
            for idx, (_, row) in enumerate(selected.head(8).iterrows(), 1):
                config_name = f"{row['window_size']} + {row['model']}"
                mean_val = row[f'{metric}_mean']
                std_val = row[f'{metric}_std']
                
                if method == 'combined_rank':
                    score_info = f"Rank: {row['combined_rank']:.0f}"
                elif method == 'weighted_score':
                    score_info = f"Score: {row['weighted_score']:.3f}"
                else:
                    score_info = f"{metric.title()}: {mean_val:.4f}"
                
                print(f"{idx:2d}. {config_name}")
                print(f"    {metric.title()}: {mean_val:.4f} ± {std_val:.4f} | {score_info}")
        
        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = None
    
    return results

def compare_selection_overlap(results_dict, metric='f1'):
    """Analyze overlap between different selection methods."""
    
    print(f"\n=== SELECTION METHOD OVERLAP ANALYSIS ===")
    print("-" * 50)
    
    # Extract configuration names from each method
    config_sets = {}
    for method, df_selected in results_dict.items():
        if df_selected is not None:
            configs = [f"{row['window_size']} + {row['model']}" 
                      for _, row in df_selected.iterrows()]
            config_sets[method] = set(configs)
    
    # Find intersections
    methods = list(config_sets.keys())
    
    if len(methods) >= 2:
        # Pairwise overlaps
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                overlap = config_sets[method1] & config_sets[method2]
                overlap_pct = len(overlap) / min(len(config_sets[method1]), len(config_sets[method2])) * 100
                
                print(f"{method1} ∩ {method2}: {len(overlap)} configs ({overlap_pct:.1f}% overlap)")
                
                if overlap:
                    print(f"  Common selections: {list(overlap)[:3]}{'...' if len(overlap) > 3 else ''}")
        
        # Find consensus (appears in all methods)
        if len(methods) > 2:
            consensus = set.intersection(*config_sets.values())
            print(f"\nConsensus picks (in all methods): {len(consensus)}")
            if consensus:
                for config in list(consensus)[:5]:
                    print(f"  - {config}")

def visualize_dual_criteria(df, metric='f1', highlight_top=10):
    """
    Create visualization showing the trade-off between mean performance and stability.
    
    Parameters:
    df (pd.DataFrame): Results dataframe
    metric (str): Metric to visualize
    highlight_top (int): Number of top configurations to highlight
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    
    # 1. Scatter plot: Mean vs Std with different selection methods
    ax = axes[0, 0]
    
    # Plot all points
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax.scatter(model_data[std_col], model_data[mean_col], 
                  label=model, alpha=0.6, s=50)
    
    # Highlight top configurations by combined rank
    top_combined, _ = rank_based_selection(df, metric, highlight_top, 'combined_rank')
    ax.scatter(top_combined[std_col], top_combined[mean_col], 
              color='red', s=100, alpha=0.8, marker='*', 
              label=f'Top {highlight_top} (Combined Rank)')
    
    ax.set_xlabel(f'{metric.title()} Standard Deviation')
    ax.set_ylabel(f'{metric.title()} Mean')
    ax.set_title(f'{metric.title()}: Performance vs Stability Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Ranking comparison heatmap
    ax = axes[0, 1]
    
    methods = ['combined_rank', 'weighted_score']
    rankings_df = pd.DataFrame()
    
    for method in methods:
        selected, _ = rank_based_selection(df, metric, 20, method)
        selected['config'] = selected['window_size'] + ' + ' + selected['model']
        rankings_df[method] = range(1, len(selected) + 1)
        if rankings_df.empty:
            rankings_df.index = selected['config']
        else:
            # Align with existing index
            for config in selected['config']:
                if config not in rankings_df.index:
                    new_row = pd.Series([np.nan] * len(rankings_df.columns), name=config)
                    rankings_df = pd.concat([rankings_df, new_row.to_frame().T])
    
    # Fill rankings for configurations that appear in one method but not another
    for method in methods:
        rankings_df[method] = rankings_df[method].fillna(25)  # Assign low rank if not in top list
    
    # Create heatmap of top 15 configs
    top_15_configs = rankings_df.head(15)
    sns.heatmap(top_15_configs.T, annot=True, fmt='.0f', cmap='RdYlBu_r', ax=ax)
    ax.set_title('Ranking Comparison Across Methods')
    ax.set_xlabel('Configuration')
    
    # 3. Distribution of selected configurations
    ax = axes[1, 0]
    
    top_configs, _ = rank_based_selection(df, metric, highlight_top, 'combined_rank')
    
    # Create combined score distribution
    ax.hist(df['f1_mean'], bins=20, alpha=0.5, label='All Configurations', color='lightblue')
    ax.hist(top_configs['f1_mean'], bins=10, alpha=0.8, label=f'Top {highlight_top}', color='red')
    
    ax.set_xlabel(f'{metric.title()} Mean')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {metric.title()} Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Stability vs Performance trade-off curve
    ax = axes[1, 1]
    
    # Sort by mean performance and show std trend
    df_sorted = df.sort_values(mean_col, ascending=False)
    top_configs_idx = df_sorted.head(20).index
    
    ax.plot(range(1, 21), df_sorted[mean_col].head(20), 'bo-', label=f'{metric.title()} Mean', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(range(1, 21), df_sorted[std_col].head(20), 'ro-', label=f'{metric.title()} Std', alpha=0.7)
    
    # Highlight dual-criteria selections
    for i, idx in enumerate(top_configs_idx):
        if idx in top_combined.index:
            ax.plot(i+1, df_sorted.loc[idx, mean_col], 'g*', markersize=10)
    
    ax.set_xlabel('Rank by Mean Performance')
    ax.set_ylabel(f'{metric.title()} Mean', color='blue')
    ax2.set_ylabel(f'{metric.title()} Std', color='red')
    ax.set_title('Top 20: Mean vs Stability Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def comprehensive_dual_criteria_analysis(file_path, metric='f1', top_n=10):
    """
    Run complete dual-criteria analysis balancing mean performance and stability.
    
    Parameters:
    file_path (str): Path to the CSV file
    metric (str): Metric to optimize ('f1', 'accuracy', 'recall', 'precision')
    top_n (int): Number of top configurations to select
    
    Returns:
    tuple: (dataframe, analysis_results, recommended_configs)
    """
    
    # Parse the data
    print("Parsing cross-validation results...")
    df = parse_cv_results(file_path)
    print(f"Successfully parsed {len(df)} configurations")
    
    # Run dual-criteria analysis
    analysis_results = analyze_dual_criteria_selection(df, metric, top_n)
    
    # Compare method overlap
    compare_selection_overlap(analysis_results, metric)
    
    # Get recommended configurations (using combined rank as primary recommendation)
    recommended, _ = rank_based_selection(df, metric, top_n, 'combined_rank')
    
    print(f"\n=== FINAL RECOMMENDATIONS (Top {top_n}) ===")
    print("=" * 50)
    print("Based on combined ranking (mean performance + stability):")
    
    for idx, (_, row) in enumerate(recommended.head(top_n).iterrows(), 1):
        config_name = f"{row['window_size']} + {row['model']}"
        mean_val = row[f'{metric}_mean']
        std_val = row[f'{metric}_std']
        ci_lower = row[f'{metric}_lower_95ci']
        ci_upper = row[f'{metric}_upper_95ci']
        
        print(f"{idx:2d}. {config_name}")
        print(f"    {metric.title()}: {mean_val:.4f} ± {std_val:.4f}")
        print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print()
    
    # Create visualizations
    print("Creating dual-criteria visualizations...")
    # visualize_dual_criteria(df, metric, top_n)
    
    return df, analysis_results, recommended

# Example usage:
# df, analysis_results, recommended = comprehensive_dual_criteria_analysis('3class nested crossvalidation.csv', metric='f1', top_n=8)

# For specific method:
# top_configs, ranking_col = rank_based_selection(df, metric='f1', top_n=10, ranking_method='combined_rank')