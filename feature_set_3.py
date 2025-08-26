import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
import os
from neurokit2 import entropy_sample, entropy_shannon, entropy_permutation, entropy_spectral
import statsmodels.tsa.stattools as stattools
import scipy.signal as signal
from scipy.signal import correlate, welch
from scipy.stats import skew, kurtosis, pearsonr

# Improved sliding window feature extraction for IMU activity recognition
class GenerateFeatures:
    def __init__(self, fs=70, window_duration=1.0, overlap=0.8, window_type='rectangular'):
        self.window_duration = window_duration  # Keep original 1.0 seconds
        self.overlap = overlap  # Keep original 0.8
        self.fs = fs
        self.window_type = window_type
        
        self.results = None
        self.features = None
        
        # Validation
        if self.window_duration <= 0:
            raise ValueError("Window duration must be positive")
        
        # Validate window type
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
        elif self.window_type == 'rectangular':
            window = np.ones(n_samples)
        else:
            window = np.ones(n_samples)
            
        return signal_data * window
    
    def calculate_zero_crossing_rate(self, signal_data):
        """Calculate zero crossing rate - important for periodic movements"""
        zero_crossings = np.where(np.diff(np.signbit(signal_data)))[0]
        return len(zero_crossings) / len(signal_data) * self.fs
    
    def calculate_autocorrelation_peak(self, signal_data):
        """Calculate peak autocorrelation (excluding lag 0) - measures periodicity"""
        if len(signal_data) < 4:
            return 0
        
        # Calculate autocorrelation
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Keep only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize by lag 0
        
        # Find peak excluding lag 0
        if len(autocorr) > 1:
            return np.max(autocorr[1:])
        return 0
    
    def calculate_frequency_features(self, signal_data):
        """Calculate discriminative frequency domain features"""
        # Get power spectral density using Welch's method
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=min(128, len(signal_data)))
        
        # Remove DC component
        if len(freqs) > 1:
            freqs = freqs[1:]
            psd = psd[1:]
        
        if len(psd) == 0 or np.sum(psd) == 0:
            return {
                'dominant_frequency': 0,
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_flatness': 0,
                'spectral_rolloff': 0,
                'alpha_band_power': 0,
                'beta_band_power': 0,
                'relative_alpha_power': 0,
                'relative_beta_power': 0
            }
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        total_power = np.sum(psd)
        
        # 1. Dominant Frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        # 2. Spectral Centroid - "center of mass" of spectrum
        spectral_centroid = np.sum(freqs * psd_norm)
        
        # 3. Spectral Spread (Bandwidth) - how spread out the spectrum is
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
        
        # 4. Spectral Flatness - distinguishes tonal vs noise-like signals
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # 5. Spectral Rolloff - frequency below which 85% of energy is contained
        cumsum_psd = np.cumsum(psd)
        total_energy = cumsum_psd[-1]
        rolloff_idx = np.where(cumsum_psd >= 0.85 * total_energy)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # 6. Band Power features - relevant for human movement
        alpha_band = (1, 4)   # Low frequency movements
        beta_band = (4, 15)   # Higher frequency movements (adjusted for IMU)
        
        alpha_power = self.calculate_band_power(freqs, psd, alpha_band)
        beta_power = self.calculate_band_power(freqs, psd, beta_band)
        
        # 7. Relative Band Power Ratios
        relative_alpha = alpha_power / (total_power + 1e-10)
        relative_beta = beta_power / (total_power + 1e-10)
        
        return {
            'dominant_frequency': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_flatness': spectral_flatness,
            'spectral_rolloff': spectral_rolloff,
            'alpha_band_power': alpha_power,
            'beta_band_power': beta_power,
            'relative_alpha_power': relative_alpha,
            'relative_beta_power': relative_beta
        }
    
    def calculate_band_power(self, freqs, psd, band):
        """Calculate power in a specific frequency band"""
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if np.any(band_mask):
            return np.sum(psd[band_mask])
        return 0
    
    def calculate_hjorth_complexity(self, signal_data):
        """Calculate Hjorth complexity parameter"""
        if len(signal_data) < 3:
            return 0
    
        # Calculate derivatives
        d1 = np.diff(signal_data) * self.fs
        d2 = np.diff(d1) * self.fs
        
        if len(d1) == 0 or len(d2) == 0:
            return 0
        
        # Hjorth parameters
        var_signal = np.var(signal_data)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        
        # Stability checks
        if var_d1 < 1e-10 or var_signal < 1e-10:
            return 0
            
        mobility = np.sqrt(var_d1 / var_signal)
        complexity = np.sqrt(var_d2 / var_d1) / mobility
        
        return complexity

    def analyze_signal(self, axis, labels, axis_name="axis"):
        """Perform sliding window feature analysis on IMU signal"""
        signal_array = np.array(axis)
        n_samples = len(signal_array)
        
        if n_samples == 0:
            raise ValueError("Signal is empty. Cannot perform analysis.")

        window_samples = round(self.window_duration * self.fs)
        step_samples = round(window_samples * (1 - self.overlap))
        
        if step_samples == 0:
            step_samples = 1
        
        # Storage for results
        window_features = []
        window_labels = []
        
        # Sliding window analysis
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_signal = signal_array[start_idx:end_idx] 
            
            # Apply window function
            windowed_signal = self.apply_window_function(window_signal)
            
            # Calculate frequency domain features
            freq_features = self.calculate_frequency_features(windowed_signal)
            
            # Assign the label with the highest occurrence to the window
            if labels is not None:
                window_label_data = labels[start_idx:end_idx]
                label = pd.Series(window_label_data).mode()[0]
                window_labels.append(label)
            else:
                label = np.nan
                window_labels.append(label)
            
            # Core discriminative features for activity recognition
            feature_dict = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_idx / self.fs,
                'end_time': end_idx / self.fs,
                'center_time': (start_idx + end_idx) / 2 / self.fs,
                'signal_name': axis_name,
                
                # Time domain features (keep only most informative)
                'mean': np.mean(windowed_signal),                    # DC component
                'std': np.std(windowed_signal),                      # Signal variability
                'rms': np.sqrt(np.mean(windowed_signal**2)),         # Signal magnitude
                'zero_crossing_rate': self.calculate_zero_crossing_rate(windowed_signal),
                
                # Temporal structure
                'autocorr_peak': self.calculate_autocorrelation_peak(windowed_signal),
                'hjorth_complexity': self.calculate_hjorth_complexity(windowed_signal),
                
                'label': label
            }
            
            # Add frequency domain features
            feature_dict.update(freq_features)
            
            window_features.append(feature_dict)
        
        return pd.DataFrame(window_features), window_labels
    
    def calculate_cross_axis_features(self, df, window_start, window_end):
        """Calculate cross-axis correlation features"""
        window_df = df.iloc[window_start:window_end]
        
        features = {}
        
        # Cross-correlations between accelerometer axes
        try:
            acc_xy_corr = pearsonr(window_df['acc_x'], window_df['acc_y'])[0]
            acc_xz_corr = pearsonr(window_df['acc_x'], window_df['acc_z'])[0]
            acc_yz_corr = pearsonr(window_df['acc_y'], window_df['acc_z'])[0]
            
            # Cross-correlations between gyroscope axes
            gyr_xy_corr = pearsonr(window_df['gyr_x'], window_df['gyr_y'])[0]
            gyr_xz_corr = pearsonr(window_df['gyr_x'], window_df['gyr_z'])[0]
            gyr_yz_corr = pearsonr(window_df['gyr_y'], window_df['gyr_z'])[0]
            
            # Accelerometer-Gyroscope coordination
            acc_gyr_corr = pearsonr(window_df['acc_mag'], window_df['gyr_mag'])[0]
            
        except:
            # Handle cases with constant signals
            acc_xy_corr = acc_xz_corr = acc_yz_corr = 0
            gyr_xy_corr = gyr_xz_corr = gyr_yz_corr = 0
            acc_gyr_corr = 0
        
        # Signal Magnitude Areas (SMA) - important for activity recognition
        acc_sma = np.mean(window_df['acc_mag'])
        gyr_sma = np.mean(window_df['gyr_mag'])
        
        features = {
            'acc_xy_corr': acc_xy_corr,
            'acc_xz_corr': acc_xz_corr,
            'acc_yz_corr': acc_yz_corr,
            'gyr_xy_corr': gyr_xy_corr,
            'gyr_xz_corr': gyr_xz_corr,
            'gyr_yz_corr': gyr_yz_corr,
            'acc_gyr_corr': acc_gyr_corr,
            'acc_magnitude_mean': acc_sma,
            'gyr_magnitude_mean': gyr_sma
        }
        
        return features
            
    def analyze_multi_axis_imu(self, df):
        """Analyze all IMU axes and combine results with cross-axis features"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df data must be a pandas DataFrame.")
        
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Store original dataframe for cross-axis feature calculation
        self.original_df = df.copy()
        self.original_df['acc_mag'] = np.sqrt(self.original_df['acc_x']**2 + self.original_df['acc_y']**2 + self.original_df['acc_z']**2)
        self.original_df['gyr_mag'] = np.sqrt(self.original_df['gyr_x']**2 + self.original_df['gyr_y']**2 + self.original_df['gyr_z']**2)
        
        labels = self.original_df['label']
        
        # Focus on most discriminative signals - reduce redundancy
        signals = {
            'acc_x': self.original_df['acc_x'],
            'acc_y': self.original_df['acc_y'], 
            'acc_z': self.original_df['acc_z'],
            'acc_mag': self.original_df['acc_mag'],
            'gyr_mag': self.original_df['gyr_mag']  # Only magnitude for gyroscope to reduce redundancy
        }
            
        all_results = []
        self.all_labels = None
            
        for signal_name, signal_data in tqdm(signals.items(), desc='Analyzing signals'):
            result_df, window_labels = self.analyze_signal(signal_data, labels, signal_name)
            all_results.append(result_df)
            
            if self.all_labels is None:
                self.all_labels = window_labels
            
        # Combine all results
        self.features = pd.concat(all_results, ignore_index=True)
            
        # Create summary pivot table with cross-axis features automatically
        self.results = self.create_summary_table()
        
        return self.features, self.results
    
    def create_summary_table(self, original_df=None):
        """Create a summary table with cross-axis features"""
        if self.features is None:
            return None
        
        # Use stored original_df if not provided
        if original_df is None:
            original_df = self.original_df
            
        # Core discriminative features
        measures = [
            # Time domain (reduced set)
            'mean', 'std', 'rms', 'zero_crossing_rate',
            # Temporal structure  
            'autocorr_peak', 'hjorth_complexity',
            # Frequency domain (discriminative)
            'dominant_frequency', 'spectral_centroid', 'spectral_spread', 
            'spectral_flatness', 'spectral_rolloff', 
            'alpha_band_power', 'beta_band_power',
            'relative_alpha_power', 'relative_beta_power'
        ]
        
        # Get unique time windows
        unique_times = sorted(self.features['center_time'].unique())
        
        summary_data = []
        window_samples = round(self.window_duration * self.fs)
        step_samples = round(window_samples * (1 - self.overlap))
        
        if step_samples == 0:
            step_samples = 1
        
        for i, time_point in enumerate(unique_times):
            time_data = self.features[self.features['center_time'] == time_point]
                
            row = {
                'center_time': time_point, 
                'start_time': time_data.iloc[0]['start_time'], 
                'end_time': time_data.iloc[0]['end_time']
            }
            
            # Add per-signal features
            for _, signal_row in time_data.iterrows():
                signal_name = signal_row['signal_name']
                for measure in measures:
                    col_name = f"{signal_name}_{measure}"
                    row[col_name] = signal_row[measure]
            
            # Add cross-axis features
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            if end_idx <= len(original_df):
                cross_features = self.calculate_cross_axis_features(original_df, start_idx, end_idx)
                row.update(cross_features)
                
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
        
        # Add labels
        if self.all_labels is not None and len(self.all_labels) == len(df):
            df['label'] = self.all_labels
        elif self.all_labels is not None:
            print(f"Warning: Label count mismatch. Expected {len(df)}, got {len(self.all_labels)}")
            
        return df