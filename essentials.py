import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import resample, medfilt, butter, filtfilt, lfilter, welch
import copy
    
def resample_data(data: dict, target_fs: int = 50):
    """Resamples time-series data within a dictionary of pandas DataFrames.

    This function iterates through each DataFrame in the input dictionary,
    removes specified columns, and then resamples the remaining data to a
    target sampling frequency.

    Note:
        - The function modifies the input DataFrames in place by dropping columns.
        - It assumes the `resample` function is from `scipy.signal`.

    Args:
        data (dict): A dictionary where keys are unique identifiers and values
            are pandas DataFrames. Each DataFrame must contain a 'time' column.
        target_fs (int, optional): The target sampling frequency in Hz.
            Defaults to 50.

    Returns:
        dict: A new dictionary with the same keys as the input, containing
            the resampled DataFrames.
    """
    resampled_dict = {}
    for i, void_instance in tqdm(enumerate(data.keys())):
        old_df = data[void_instance]
        old_df.drop(columns=['Real time'], axis = 1, inplace=True)
        
        original_fs = 1 / old_df['time'].diff().median()
        num_samples = int(len(old_df) * target_fs / original_fs)
        
        resampled_data = resample(old_df, num_samples)
        
        resampled_dict[void_instance] = pd.DataFrame(resampled_data, columns=old_df.columns )
    return resampled_dict


def three_class_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'label' column to an IMU DataFrame based on a ground truth time range.

    This function identifies the "void" period using the start and end timestamps from the ground truth DataFrame. It then
    classifies each row in the IMU DataFrame into one of three categories:
    'pre-void', 'void', or 'post-void', based on whether the IMU timestamp
    occurs before, during, or after this event period.

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
    pre_void_labels = [f'pre-void' for i in (df[(df['time'] < ue[0])])['time']]
    void_labels = [f'void' for i in (df[(df['time'] >= ue[0]) & (df['time'] <= ue[1])])['time']]
    post_void_labels = [f'post-void' for i in (df[(df['time'] > ue[1])])['time']]
    
    labels = pre_void_labels + void_labels + post_void_labels
    df['label'] = labels
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df['acc_x'] = (df['acc_x'] - df['acc_x'].mean()) / df['acc_x'].std()
    df['acc_y'] = (df['acc_y'] - df['acc_y'].mean()) / df['acc_y'].std()
    df['acc_z'] = (df['acc_z'] - df['acc_z'].mean()) / df['acc_z'].std()
    df['gyr_x'] = (df['gyr_x'] - df['gyr_x'].mean()) / df['gyr_x'].std()
    df['gyr_y'] = (df['gyr_y'] - df['gyr_y'].mean()) / df['gyr_y'].std()
    df['gyr_z'] = (df['gyr_z'] - df['gyr_z'].mean()) / df['gyr_z'].std()
    
    return df


def two_class_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'label' column to an IMU DataFrame based on a ground truth time range.

    This function identifies the "void" period using the start and end timestamps from the ground truth DataFrame. It then
    classifies each row in the IMU DataFrame into one of three categories:
    'void', or 'non-void', based on whether the IMU timestamp
    occurs before, during, or after this event period.
    Any data point outside the void window is classified as 'non-void'.

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
    pre_void_labels = [f'non-void' for i in (df[(df['time'] < ue[0])])['time']]
    void_labels = [f'void' for i in (df[(df['time'] >= ue[0]) & (df['time'] <= ue[1])])['time']]
    post_void_labels = [f'non-void' for i in (df[(df['time'] > ue[1])])['time']]
    
    labels = pre_void_labels + void_labels + post_void_labels
    df['label'] = labels
    return df


# def preprocess_data(data):
#     """Complete preprocessing pipeline matching Anguita et al.'s methodology."""
#     # 1. Apply median filter
#     for col in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']:
#         data[col] = medfilt(data[col], kernel_size=3)
    
#     time = data['time']
    
#     fs = 1 / data['time'].diff().median() # sampling frequency
    
#     # 2. Butterworth Low-Pass (20Hz cutoff)
#     def butter_lowpass_filter(data, cutoff, fs, order=4):
#         nyq = 0.5 * fs
#         normal_cutoff = cutoff / nyq
#         b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
#         y = filtfilt(b, a, data, axis=0)  # Zero-phase filtering
#         return y
    
#     data['acc_x'] = butter_lowpass_filter(data['acc_x'], 20, fs)
#     data['acc_y'] = butter_lowpass_filter(data['acc_y'], 20, fs)
#     data['acc_z'] = butter_lowpass_filter(data['acc_z'], 20, fs)
#     data['gyr_x'] = butter_lowpass_filter(data['gyr_x'], 20, fs)
#     data['gyr_y'] = butter_lowpass_filter(data['gyr_y'], 20, fs)
#     data['gyr_z'] = butter_lowpass_filter(data['gyr_z'], 20, fs)
    
#     return data


def preprocess_data_separate_acc_gravity(data):
    """Complete preprocessing pipeline matching Anguita et al.'s methodology."""
    # 1. Apply median filter
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']:
        data[col] = medfilt(data[col], kernel_size=3)
    
    time = data['time']
    
    fs = 1 / data['time'].diff().median() # sampling frequency
    
    # 2. Butterworth Low-Pass (20Hz cutoff)
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
        y = filtfilt(b, a, data, axis=0)  # Zero-phase filtering
        return y
    
    data['acc_x'] = butter_lowpass_filter(data['acc_x'], 20, fs)
    data['acc_y'] = butter_lowpass_filter(data['acc_y'], 20, fs)
    data['acc_z'] = butter_lowpass_filter(data['acc_z'], 20, fs)
    data['gyr_x'] = butter_lowpass_filter(data['gyr_x'], 20, fs)
    data['gyr_y'] = butter_lowpass_filter(data['gyr_y'], 20, fs)
    data['gyr_z'] = butter_lowpass_filter(data['gyr_z'], 20, fs)
    
    # Creating a copy of the total acceleration, I don't want any wahala
    acc_x = copy.deepcopy(data['acc_x'])
    acc_y = copy.deepcopy(data['acc_y'])
    acc_z = copy.deepcopy(data['acc_z'])
    
    # This is the acceleration due to gravity
    grav_x = butter_lowpass_filter(acc_x, cutoff=0.3, fs=fs)
    grav_y = butter_lowpass_filter(acc_y, cutoff=0.3, fs=fs)
    grav_z = butter_lowpass_filter(acc_z, cutoff=0.3, fs=fs)
    
    # Subtracting this from the original acceleration gives the acceleration due motion
    data['acc_x'] = acc_x - grav_x
    data['acc_y'] = acc_y - grav_y
    data['acc_z'] = acc_z - grav_z
    
    return data