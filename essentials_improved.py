import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import resample, medfilt, butter, filtfilt, lfilter, welch
import copy
    

def resample_data(data: dict, target_fs: int = 50):
    """Properly resamples time-series data to consistent frequency with time preservation.

    This function resamples IMU data to a target frequency using interpolation,
    properly handling time vectors and preserving all sensor channels.

    Args:
        data (dict): A dictionary where keys are unique identifiers and values
            are pandas DataFrames. Each DataFrame must contain a 'time' column
            and sensor columns.
        target_fs (int, optional): The target sampling frequency in Hz.
            Defaults to 50.

    Returns:
        dict: A new dictionary with the same keys as the input, containing
            the resampled DataFrames with consistent 50Hz sampling.
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


def preprocess_data(data, target_fs=50):
    """Complete preprocessing pipeline without gravity separation.
    
    Args:
        data: DataFrame with IMU data
        target_fs: Target sampling frequency (should be 50Hz after resampling)
        
    Returns:
        Preprocessed DataFrame
    """
    data = data.copy()
    
    # 1. Apply median filter
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    for col in sensor_cols:
        if col in data.columns:
            data[col] = medfilt(data[col], kernel_size=3)
    
    # 2. Butterworth Low-Pass (20Hz cutoff) - fs should be 50Hz after resampling
    def butter_lowpass_filter(data_array, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
        y = filtfilt(b, a, data_array, axis=0)  # Zero-phase filtering
        return y
    
    for col in sensor_cols:
        if col in data.columns:
            data[col] = butter_lowpass_filter(data[col], 20, target_fs)
    
    return data


def preprocess_data_separate_acc_gravity(data, target_fs=50):
    """Complete preprocessing pipeline with gravity separation.
    
    Args:
        data: DataFrame with IMU data
        target_fs: Target sampling frequency (should be 50Hz after resampling)
        
    Returns:
        Preprocessed DataFrame with gravity removed from acceleration
    """
    data = data.copy()
    
    # 1. Apply median filter
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    for col in sensor_cols:
        if col in data.columns:
            data[col] = medfilt(data[col], kernel_size=3)
    
    # 2. Butterworth Low-Pass (20Hz cutoff) - fs should be 50Hz after resampling
    def butter_lowpass_filter(data_array, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
        y = filtfilt(b, a, data_array, axis=0)  # Zero-phase filtering
        return y
    
    # Apply 20Hz filter to all sensors first
    for col in sensor_cols:
        if col in data.columns:
            data[col] = butter_lowpass_filter(data[col], 20, target_fs)
    
    # 3. Separate gravity from acceleration
    # Store original acceleration after 20Hz filtering
    acc_x = data['acc_x'].copy()
    acc_y = data['acc_y'].copy()
    acc_z = data['acc_z'].copy()
    
    # Extract gravity component using 0.3Hz low-pass filter
    grav_x = butter_lowpass_filter(acc_x, cutoff=0.3, fs=target_fs)
    grav_y = butter_lowpass_filter(acc_y, cutoff=0.3, fs=target_fs)
    grav_z = butter_lowpass_filter(acc_z, cutoff=0.3, fs=target_fs)
    
    # Remove gravity to get body acceleration
    data['acc_x'] = acc_x - grav_x
    data['acc_y'] = acc_y - grav_y
    data['acc_z'] = acc_z - grav_z
    
    # Optional: Add magnitude features

    
    return data


def complete_preprocessing_pipeline(data_dict, ground_truth_dict, 
                                target_fs=50, remove_gravity=True, 
                                normalize_data=True, use_three_classes=False):
    """Complete preprocessing pipeline in correct order.
    
    Args:
        data_dict: Dictionary of DataFrames with IMU data
        ground_truth_dict: Dictionary of ground truth DataFrames
        target_fs: Target sampling frequency
        remove_gravity: Whether to remove gravity from acceleration
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
        
        # Add labels
        if key in ground_truth_dict:
            if use_three_classes:
                df = three_class_labels(df, ground_truth_dict[key])
            else:
                df = two_class_labels(df, ground_truth_dict[key])
        
        # Apply preprocessing
        if remove_gravity:
            df = preprocess_data_separate_acc_gravity(df, target_fs)
        else:
            df = preprocess_data(df, target_fs)
        
        # Normalize if requested
        if normalize_data:
            df = normalize(df)
        
        processed_data[key] = df
    
    return processed_data