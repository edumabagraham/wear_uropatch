import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import scipy.signal as signal
from scipy.signal import welch
from scipy.stats import pearsonr
import neurokit2 as nk


# Optimized sliding window feature extraction for IMU activity recognition
class GenerateFeatures:
    def __init__(self, fs=70, window_duration=1.0, overlap=0.8, window_type='rectangular'):
        self.window_duration = window_duration
        self.overlap = overlap
        self.fs = fs
        self.window_type = window_type

        self.results = None
        self.features = None
        self.all_labels = None
        self.original_df = None   # store original df for later use

        # Validation
        if self.window_duration <= 0:
            raise ValueError("Window duration must be positive")
        
        valid_windows = ['hann', 'hamming', 'blackman', 'bartlett', 'rectangular', 'tukey']
        if self.window_type not in valid_windows:
            raise ValueError(f"Window type must be one of {valid_windows}")
            
        window_samples = round(self.window_duration * self.fs)
        if window_samples < 10:
            raise ValueError("Window too small - need at least 10 samples for reliable feature calculation")
        
    def apply_window_function(self, signal_data):
        """Apply window function to signal data"""
        n_samples = len(signal_data)
        
        if self.window_type == 'hann':
            window = np.hanning(n_samples)
        elif self.window_type == 'hamming':
            window = np.hamming(n_samples)
        elif self.window_type == 'blackman':
            window = np.blackman(n_samples)
        elif self.window_type == 'bartlett':
            window = np.bartlett(n_samples)
        elif self.window_type == 'tukey':
            window = signal.windows.tukey(n_samples)
        else:  # rectangular
            window = np.ones(n_samples)
            
        return signal_data * window
    
    def calculate_jerk(self, signal_data):
        """Jerk = mean absolute derivative of acceleration/gyro signal"""
        diff = np.diff(signal_data) * self.fs
        return np.mean(np.abs(diff)) if len(diff) > 0 else 0
    
    def calculate_energy(self, signal_data):
        """Energy of signal"""
        return np.sum(signal_data**2) / len(signal_data)
    
    def calculate_autocorrelation_features(self, signal_data):
        """Return peak value and lag of autocorrelation (excluding lag 0)"""
        if len(signal_data) < 4:
            return 0, 0
        
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # keep positive lags
        autocorr = autocorr / (autocorr[0] + 1e-10)
        
        if len(autocorr) > 1:
            lag = np.argmax(autocorr[1:]) + 1
            return np.max(autocorr[1:]), lag / self.fs
        return 0, 0
    
    def calculate_frequency_features(self, signal_data):
        """Calculate discriminative frequency domain features"""
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=min(128, len(signal_data)))
        
        if len(freqs) <= 1 or np.sum(psd) == 0:
            return {
                'dominant_frequency': 0,
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_entropy': 0,
                'spectral_rolloff': 0,
                'low_band_power': 0,
                'mid_band_power': 0,
                'high_band_power': 0
            }
        
        psd_norm = psd / np.sum(psd)
        total_power = np.sum(psd)
        
        # 1. Dominant frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        # 2. Spectral centroid & spread
        spectral_centroid = np.sum(freqs * psd_norm)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
        
        # 3. Spectral entropy
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        # 4. Spectral rolloff (85%)
        cumsum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd >= 0.85 * cumsum_psd[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # 5. Band powers (IMU-specific bands)
        low_band = (0.5, 3)
        mid_band = (3, 8)
        high_band = (8, 20)
        
        low_power = self.calculate_band_power(freqs, psd, low_band)
        mid_power = self.calculate_band_power(freqs, psd, mid_band)
        high_power = self.calculate_band_power(freqs, psd, high_band)
        
        return {
            'dominant_frequency': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_entropy': spectral_entropy,
            'spectral_rolloff': spectral_rolloff,
            'low_band_power': low_power,
            'mid_band_power': mid_power,
            'high_band_power': high_power
        }
    
    def calculate_band_power(self, freqs, psd, band):
        """Calculate power in a specific frequency band"""
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.sum(psd[mask]) if np.any(mask) else 0
    
    def analyze_signal(self, axis, labels, axis_name="axis"):
        """Sliding window feature analysis for a single IMU axis"""
        signal_array = np.array(axis)
        n_samples = len(signal_array)
        
        if n_samples == 0:
            raise ValueError("Signal is empty. Cannot perform analysis.")

        window_samples = round(self.window_duration * self.fs)
        step_samples = round(window_samples * (1 - self.overlap)) or 1
        
        window_features, window_labels = [], []
        
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_signal = signal_array[start_idx:end_idx] 
            
            windowed_signal = self.apply_window_function(window_signal)
            
            # Frequency domain
            freq_features = self.calculate_frequency_features(windowed_signal)
            
            # Label assignment
            if labels is not None:
                window_label_data = labels[start_idx:end_idx]
                label = pd.Series(window_label_data).mode()[0]
            else:
                label = np.nan
            window_labels.append(label)
            
            # Time + temporal features
            autocorr_peak, autocorr_lag = self.calculate_autocorrelation_features(windowed_signal)
            sample_entropy = nk.entropy_sample(windowed_signal)[0]
            
            feature_dict = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'center_time': (start_idx + end_idx) / 2 / self.fs,
                'signal_name': axis_name,
                
                'mean': np.mean(windowed_signal),
                'std': np.std(windowed_signal),
                'energy': self.calculate_energy(windowed_signal),
                'jerk': self.calculate_jerk(windowed_signal),
                
                'autocorr_peak': autocorr_peak,
                'autocorr_lag': autocorr_lag,
                'sample_entropy': sample_entropy,
                
                'label': label
            }
            
            feature_dict.update(freq_features)
            window_features.append(feature_dict)
        
        return pd.DataFrame(window_features), window_labels
    
    def calculate_cross_axis_features(self, df, window_start, window_end):
        """Cross-axis correlations + coordination features"""
        window_df = df.iloc[window_start:window_end]
        
        try:
            acc_xy = pearsonr(window_df['acc_x'], window_df['acc_y'])[0]
            acc_xz = pearsonr(window_df['acc_x'], window_df['acc_z'])[0]
            acc_yz = pearsonr(window_df['acc_y'], window_df['acc_z'])[0]
            acc_gyr = pearsonr(window_df['acc_mag'], window_df['gyr_mag'])[0]
        except:
            acc_xy = acc_xz = acc_yz = acc_gyr = 0
        
        tilt = np.arctan2(window_df['acc_z'], np.sqrt(window_df['acc_x']**2 + window_df['acc_y']**2))
        tilt_var = np.var(tilt)
        
        return {
            'acc_xy_corr': acc_xy,
            'acc_xz_corr': acc_xz,
            'acc_yz_corr': acc_yz,
            'acc_gyr_corr': acc_gyr,
            'tilt_var': tilt_var,
            'acc_magnitude_mean': np.mean(window_df['acc_mag']),
            'gyr_magnitude_mean': np.mean(window_df['gyr_mag'])
        }
    
    def analyze_multi_axis_imu(self, df):
        """Extract features from all IMU axes + cross-axis features"""
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'label']
        if any(col not in df.columns for col in required_cols):
            raise ValueError("Missing required IMU columns.")
        
        df = df.copy()
        self.original_df = df   # save original dataframe for later use

        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
        
        labels = df['label']
        signals = {
            'acc_x': df['acc_x'],
            'acc_y': df['acc_y'], 
            'acc_z': df['acc_z'],
            'acc_mag': df['acc_mag'],
            'gyr_mag': df['gyr_mag']
        }
            
        all_results, self.all_labels = [], None
            
        for signal_name, signal_data in tqdm(signals.items(), desc='Analyzing signals'):
            result_df, window_labels = self.analyze_signal(signal_data, labels, signal_name)
            all_results.append(result_df)
            
            if self.all_labels is None:
                self.all_labels = window_labels
            
        self.features = pd.concat(all_results, ignore_index=True)
        self.results = self.create_summary_table()
        
        return self.features, self.results
    
    def create_summary_table(self):
        """Aggregate per-window features + cross-axis features"""
        if self.features is None or self.original_df is None:
            return None
        
        measures = [
            'mean', 'std', 'energy', 'jerk',
            'autocorr_peak', 'autocorr_lag', 'sample_entropy',
            'dominant_frequency', 'spectral_centroid', 'spectral_spread', 
            'spectral_entropy', 'spectral_rolloff',
            'low_band_power', 'mid_band_power', 'high_band_power'
        ]
        
        unique_times = sorted(self.features['center_time'].unique())
        
        summary_data = []
        window_samples = round(self.window_duration * self.fs)
        step_samples = round(window_samples * (1 - self.overlap)) or 1
        
        for i, time_point in enumerate(unique_times):
            time_data = self.features[self.features['center_time'] == time_point]
                
            row = {'center_time': time_point}
            row.update({f"{r['signal_name']}_{m}": r[m] for _, r in time_data.iterrows() for m in measures})
            
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            if end_idx <= len(self.original_df):
                row.update(self.calculate_cross_axis_features(self.original_df, start_idx, end_idx))
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
        if self.all_labels is not None and len(self.all_labels) == len(df):
            df['label'] = self.all_labels
        return df