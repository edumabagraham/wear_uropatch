import pandas as pd
import numpy as np
from tqdm import tqdm

class GenerateStatisticalFeatures:
    def __init__(self, fs=60, window_duration=1.0, overlap=0.8, window_type='rectangular'):
        self.window_duration = window_duration
        self.overlap = overlap
        self.fs = fs
        self.window_type = window_type
        self.results = None
        self.features = None
        self.global_windows = None
        self.all_labels = None

        # Validation
        if self.window_duration <= 0:
            raise ValueError("Window duration must be positive")
        valid_windows = ['hann', 'hamming', 'blackman', 'bartlett', 'rectangular', 'tukey']
        if self.window_type not in valid_windows:
            raise ValueError(f"Window type must be one of {valid_windows}")

    def apply_window_function(self, signal_data):
        n_samples = len(signal_data)
        if self.window_type == 'hann':
            w = np.hanning(n_samples)
        elif self.window_type == 'hamming':
            w = np.hamming(n_samples)
        elif self.window_type == 'blackman':
            w = np.blackman(n_samples)
        elif self.window_type == 'bartlett':
            w = np.bartlett(n_samples)
        elif self.window_type == 'tukey':
            w = np.ones(n_samples) * 0.5  # Simplified tukey
        else:
            w = np.ones(n_samples)
        return signal_data * w

    def compute_global_windows(self, n_samples, labels):
        win_size = round(self.window_duration * self.fs)
        step = round(win_size * (1 - self.overlap))
        if step == 0:
            step = 1
        windows = []
        self.all_labels = []
        for start_idx in range(0, n_samples - win_size + 1, step):
            end_idx = start_idx + win_size
            center_idx = (start_idx + end_idx) // 2
            windows.append((start_idx, end_idx))
            self.all_labels.append(labels[center_idx])
        self.global_windows = windows

    def calculate_statistical_features(self, windowed_signal):
        """Calculate only statistical features"""
        return {
            'mean': np.mean(windowed_signal),
            'std': np.std(windowed_signal),
            'range': np.max(windowed_signal) - np.min(windowed_signal),
            'rms': np.sqrt(np.mean(windowed_signal**2)),
            'min': np.min(windowed_signal),
            'max': np.max(windowed_signal)
        }

    def analyze_signal(self, axis, axis_name="axis"):
        signal = np.array(axis)
        window_features = []

        if self.global_windows is None or self.all_labels is None:
            raise ValueError("Global windows and labels must be computed before analyzing signal.")

        for (start_idx, end_idx), label in zip(self.global_windows, self.all_labels):
            window_signal = signal[start_idx:end_idx]
            windowed_signal = self.apply_window_function(window_signal)
            
            # Calculate only statistical features
            stat_features = self.calculate_statistical_features(windowed_signal)
            
            feature_dict = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_idx / self.fs,
                'end_time': end_idx / self.fs,
                'center_time': (start_idx + end_idx) / 2 / self.fs,
                'signal_name': axis_name,
                'label': label
            }
            
            # Add statistical features
            feature_dict.update(stat_features)
            window_features.append(feature_dict)
            
        return pd.DataFrame(window_features)

    def analyze_multi_axis_imu(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'label']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if self.global_windows is None or self.all_labels is None:
            self.compute_global_windows(len(df), df['label'])

        signals = {
            'acc_x': df['acc_x'], 'acc_y': df['acc_y'], 'acc_z': df['acc_z'], 
            'gyr_x': df['gyr_x'], 'gyr_y': df['gyr_y'], 'gyr_z': df['gyr_z']
        }

        all_results = []
        for name, sig in tqdm(signals.items(), desc='Analyzing statistical features'):
            result_df = self.analyze_signal(sig, axis_name=name)
            all_results.append(result_df)

        self.features = pd.concat(all_results, ignore_index=True)
        self.results = self.create_summary_table()
        return self.features, self.results

    def create_summary_table(self):
        if self.features is None:
            return None
            
        # Only statistical measures
        statistical_measures = ['mean', 'std', 'range', 'rms', 'min', 'max']
        
        unique_times = sorted(self.features['center_time'].unique())
        summary_data = []
        
        for t in unique_times:
            time_data = self.features[self.features['center_time'] == t]
            row = {
                'center_time': t, 
                'start_time': time_data.iloc[0]['start_time'],
                'end_time': time_data.iloc[0]['end_time']
            }
            
            for _, sig_row in time_data.iterrows():
                signal_name = sig_row['signal_name']
                for measure in statistical_measures:
                    row[f"{signal_name}_{measure}"] = sig_row[measure]
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
        
        if self.all_labels is not None and len(self.all_labels) == len(df):
            df['label'] = self.all_labels
            
        return df