import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import resample, medfilt, butter, filtfilt, lfilter, welch
import copy
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def resample_data(data: dict, target_fs: int = 60):
    """Properly resamples time-series data to consistent frequency with time preservation.

    This function resamples IMU data to a target frequency using interpolation,
    properly handling time vectors and preserving all sensor channels.

    Args:
        data (dict): A dictionary where keys are unique identifiers and values
            are pandas DataFrames. Each DataFrame must contain a 'time' column
            and sensor columns.
        target_fs (int, optional): The target sampling frequency in Hz.
            Defaults to 60.

    Returns:
        dict: A new dictionary with the same keys as the input, containing
            the resampled DataFrames with consistent 60Hz sampling.
    """
    resampled_dict = {}
    
    for void_instance in tqdm(data.keys()):
        old_df = data[void_instance].copy()
        
        # Remove unwanted columns if they exist
        old_df.drop(columns=['Real time'], axis=1, inplace=True, errors='ignore')
        
        # Create uniform time vector
        start_time = old_df['time'].iloc[0]
        end_time = old_df['time'].iloc[-1]
        duration = end_time - start_time
        num_samples = int(duration * target_fs) + 1  # +1 to include end point
        
        # Generate new uniform time vector
        new_time = np.linspace(start_time, end_time, num_samples)
        
        # Initialize resampled dataframe with new time
        resampled_df = pd.DataFrame({'time': new_time})
        
        # Interpolate each sensor channel
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        for col in sensor_cols:
            if col in old_df.columns:
                resampled_df[col] = np.interp(new_time, old_df['time'], old_df[col])
        
        # Preserve labels if they exist (using nearest neighbor)
        if 'label' in old_df.columns:
            label_indices = np.searchsorted(old_df['time'], new_time)
            label_indices = np.clip(label_indices, 0, len(old_df) - 1)
            resampled_df['label'] = old_df['label'].iloc[label_indices].values
        
        resampled_dict[void_instance] = resampled_df
    
    return resampled_dict


def three_class_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'label' column to an IMU DataFrame based on a ground truth time range.

    This function identifies the "void" period using the start and end timestamps 
    from the ground truth DataFrame. It then classifies each row in the IMU DataFrame 
    into one of three categories: 'pre-void', 'void', or 'post-void', based on whether 
    the IMU timestamp occurs before, during, or after this event period.

    Args:
        df: The DataFrame containing IMU (Inertial Measurement Unit) data.
                It must include a 'time' column with timestamp values.
        gt: The ground truth DataFrame that provides the reference time range.
            It must have a 'Time' column, where the first element marks the
            start and the last element marks the end of the "void" event.

    Returns:
        The `df` DataFrame, modified in place to include a new 'label'
        column that contains the classification for each row.
    """
    ue = [gt['Time'].iloc[0], gt['Time'].iloc[-1]]
    
    # Create boolean masks for efficient labeling
    pre_void_mask = df['time'] < ue[0]
    void_mask = (df['time'] >= ue[0]) & (df['time'] <= ue[1])
    post_void_mask = df['time'] > ue[1]
    
    # Initialize labels array
    labels = np.empty(len(df), dtype=object)
    labels[pre_void_mask] = 'pre-void'
    labels[void_mask] = 'void'
    labels[post_void_mask] = 'post-void'
    
    df['label'] = labels
    return df


def two_class_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'label' column to an IMU DataFrame based on a ground truth time range.

    This function identifies the "void" period using the start and end timestamps 
    from the ground truth DataFrame. It then classifies each row in the IMU DataFrame 
    into one of two categories: 'void', or 'non-void', based on whether the IMU 
    timestamp occurs during or outside this event period.

    Args:
        df: The DataFrame containing IMU (Inertial Measurement Unit) data.
                It must include a 'time' column with timestamp values.
        gt: The ground truth DataFrame that provides the reference time range.
            It must have a 'Time' column, where the first element marks the
            start and the last element marks the end of the "void" event.

    Returns:
        The `df` DataFrame, modified in place to include a new 'label'
        column that contains the classification for each row.
    """
    ue = [gt['Time'].iloc[0], gt['Time'].iloc[-1]]
    
    # Create boolean masks for efficient labeling
    void_mask = (df['time'] >= ue[0]) & (df['time'] <= ue[1])
    
    # Initialize all as non-void, then set void labels
    labels = np.full(len(df), 'non-void', dtype=object)
    labels[void_mask] = 'void'
    
    df['label'] = labels
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize all sensor channels.
    
    Args:
        df: DataFrame containing sensor data
        
    Returns:
        DataFrame with normalized sensor values
    """
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    
    for col in sensor_cols:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

def normalization(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit normalizer on training data and apply to both train and test data.
    
    Args:
        train_df: Training DataFrame containing sensor data
        test_df: Test DataFrame containing sensor data
        
    Returns:
        Tuple of (normalized training DataFrame, normalized test DataFrame, fitted scaler)
    """
    
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    available_cols = [col for col in sensor_cols if col in train_df.columns]
    
    scaler = StandardScaler()
    scaler.fit(train_df[available_cols])
    
    train_df_normalized = train_df.copy()
    test_df_normalized = test_df.copy()
    train_df_normalized[available_cols] = scaler.transform(train_df_normalized[available_cols])
    test_df_normalized[available_cols] = scaler.transform(test_df_normalized[available_cols])
    return train_df_normalized, test_df_normalized


def complete_preprocessing_pipeline(data_dict, ground_truth_dict, 
                                target_fs=60,normalize_data=True,
                                use_three_classes=False):
    """Complete preprocessing pipeline in correct order.
    
    Args:
        data_dict: Dictionary of DataFrames with IMU data
        ground_truth_dict: Dictionary of ground truth DataFrames
        target_fs: Target sampling frequency
        normalize_data: Whether to apply z-score normalization
        use_three_classes: If True, use 3-class labels, else 2-class
        
    Returns:
        Dictionary of fully preprocessed DataFrames
    """
    print("Step 1: Resampling data to", target_fs, "Hz...")
    resampled_data = resample_data(data_dict, target_fs)
    
    processed_data = {}
    
    for key in tqdm(resampled_data.keys(), desc="Step 2: Processing each instance"):
        df = resampled_data[key].copy()
        
        # Normalize if requested
        if normalize_data:
            df = normalize(df)
        
        # Add labels
        if key in ground_truth_dict:
            if use_three_classes:
                df = three_class_labels(df, ground_truth_dict[key])
            else:
                df = two_class_labels(df, ground_truth_dict[key])

        processed_data[key] = df
    
    return processed_data