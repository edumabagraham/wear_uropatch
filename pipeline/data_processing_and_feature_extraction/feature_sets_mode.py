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
from scipy.stats import skew, kurtosis

# Sliding window feature extraction
class GenerateFeatures:
    def __init__(self, fs=50, window_duration=1.0, overlap=0.8, window_type='rectangular'):
        self.window_duration = window_duration
        self.overlap = overlap
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
        if window_samples < 3:
            raise ValueError("Window too small - need at least 3 samples for entropy calculation")
        
    def apply_window_function(self, signal_data):
        """
        Apply window function to signal data
        """
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
            window = np.ones(n_samples)  # No windowing
        else:
            window = np.ones(n_samples)  # Default to rectangular
            
        return signal_data * window
    
    def calculate_energy(self, signal_data):
        """
        Calculate signal energy in time domain
        """
        return np.sum(signal_data**2)
    
    def calculate_spectral_energy(self, signal_data):
        """
        Calculate spectral energy using FFT
        """
        fft_vals = np.fft.fft(signal_data)
        spectral_energy = np.sum(np.abs(fft_vals)**2)
        return spectral_energy

    def calculate_frequency_features(self, signal_data):
        """
        Calculate comprehensive frequency domain features
        """
        # Get power spectral density using Welch's method
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=min(256, len(signal_data)))
        
        # Remove DC component
        if len(freqs) > 1:
            freqs = freqs[1:]
            psd = psd[1:]
        
        if len(psd) == 0 or np.sum(psd) == 0:
            # Return default values if PSD is empty or all zeros
            return {
                # 'dominant_frequency': 0,
                'spectral_centroid': 0,
                'spectral_spread': 0,
                'spectral_flatness': 0,
                'peak_frequency_ratio': 0,
                'hjorth_frequency': 0,
                'spectral_skewness': 0,
                'spectral_kurtosis': 0,
                'snr': 0
            }
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # 1. Dominant Frequency
        dominant_freq = freqs[np.argmax(psd)]
        
        # 2. Spectral Centroid
        spectral_centroid = np.sum(freqs * psd_norm)
        
        # 3. Spectral Spread (Bandwidth)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
        
        # 4. Spectral Flatness
        # Geometric mean / Arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # 5. Peak Frequency Ratio
        peak_power = np.max(psd)
        total_power = np.sum(psd)
        peak_frequency_ratio = peak_power / (total_power + 1e-10)
        
        # 6. Hjorth Frequency (requires time domain calculation)
        hjorth_freq = self.calculate_hjorth_frequency(signal_data)
        
        # 7. Spectral Skewness
        spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * psd_norm) / (spectral_spread ** 3 + 1e-10)
        
        # 8. Spectral Kurtosis
        spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * psd_norm) / (spectral_spread ** 4 + 1e-10)
        
        # 9. Band Power features (define frequency bands)
        # For IMU data, we'll use general frequency bands
        alpha_band = (8, 13)  # Hz
        beta_band = (13, 30)  # Hz
        gamma_band = (30, 50)  # Hz (limited by Nyquist frequency)
        
        alpha_power = self.calculate_band_power(freqs, psd, alpha_band)
        beta_power = self.calculate_band_power(freqs, psd, beta_band)
        gamma_power = self.calculate_band_power(freqs, psd, gamma_band)
        
        # 10. Relative Band Power Ratios
        relative_alpha = alpha_power / (total_power + 1e-10)
        relative_beta = beta_power / (total_power + 1e-10)
        relative_gamma = gamma_power / (total_power + 1e-10)
        
        # 11. SNR (Signal to Noise Ratio)
        # Define signal as power in dominant frequency bin and noise as remaining power
        signal_power = peak_power
        noise_power = total_power - signal_power
        snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        
        return {
            # 'dominant_frequency': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_flatness': spectral_flatness,
            'peak_frequency_ratio': peak_frequency_ratio,
            'hjorth_frequency': hjorth_freq,
            'spectral_skewness': spectral_skewness,
            'spectral_kurtosis': spectral_kurtosis,
            'snr': snr
        }
    
    def calculate_band_power(self, freqs, psd, band):
        """
        Calculate power in a specific frequency band
        """
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if np.any(band_mask):
            return np.sum(psd[band_mask])
        return 0
    
    def calculate_hjorth_frequency(self, signal_data):
        """
        Calculate Hjorth frequency parameter
        """
        if len(signal_data) < 3:
            return float('nan')  # or raise ValueError
    
        # Calculate derivatives
        d1 = np.diff(signal_data)   * self.fs
        d2 = np.diff(d1)    * self.fs
        
        if len(d1) == 0 or len(d2) == 0:
            return 0
        
        # Hjorth parameters
        var_signal = np.var(signal_data)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        
        # Stability checks
        if var_d1 < 1e-10:  # More robust than exact equality to 0
            return 0.0
            
        hjorth_freq = np.sqrt(var_d2 / var_d1) / (2 * np.pi)
        
        return hjorth_freq

    def analyze_signal(self, axis, labels, axis_name="axis"):
        """
        Perform sliding window entropy analysis on IMU signal
        """
        signal = np.array(axis)
        n_samples = len(signal)
        
        if n_samples == 0:
            raise ValueError("Signal is empty. Cannot perform analysis.")

        window_samples = round(self.window_duration * self.fs)
        step_samples = round(window_samples * (1 - self.overlap))
        
        if step_samples == 0:
            step_samples = 1  # Prevent infinite loop
        
        # Storage for results
        window_features = []
        window_labels = []
        
        # Sliding window analysis
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_signal = signal[start_idx:end_idx] 
            
            # Apply window function
            windowed_signal = self.apply_window_function(window_signal)
            
            try:
                # Calculate different entropy measures on windowed signal
                perm_ent = entropy_permutation(windowed_signal)[0]
                spectral_ent = entropy_spectral(windowed_signal)[0]
            except Exception as e:
                print(f"Warning: Entropy calculation failed for window {start_idx}-{end_idx}: {e}")
                perm_ent = np.nan
                spectral_ent = np.nan
            
            # Calculate energy features on windowed signal
            time_energy = self.calculate_energy(windowed_signal)
            spectral_energy = self.calculate_spectral_energy(windowed_signal)
            
            # Calculate frequency domain features on windowed signal
            freq_features = self.calculate_frequency_features(windowed_signal)

            # Assign the label with the highest occurrence to the window
            if labels is not None:
                window_label_data = labels[start_idx:end_idx]
                label = pd.Series(window_label_data).mode()[0]
                window_labels.append(label)
            else:
                label = np.nan
                window_labels.append(label)
            
            # Store results
            feature_dict = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_idx / self.fs,
                'end_time': end_idx / self.fs,
                'center_time': (start_idx + end_idx) / 2 / self.fs,
                'signal_name': axis_name,
                'permutation_entropy': perm_ent,
                'spectral_entropy': spectral_ent,
                'mean': np.mean(windowed_signal),
                'std': np.std(windowed_signal),
                'range': np.max(windowed_signal) - np.min(windowed_signal),
                'rms': np.sqrt(np.mean(windowed_signal**2)),
                'var': np.var(windowed_signal),
                'min': np.min(windowed_signal),
                'max': np.max(windowed_signal),
                'time_energy': time_energy,
                # 'spectral_energy': spectral_energy,
                'label': label
            }
            
            # Add frequency domain features
            feature_dict.update(freq_features)
            
            window_features.append(feature_dict)
        
        return pd.DataFrame(window_features), window_labels
            
    def analyze_multi_axis_imu(self, df):
        """
        Analyze all IMU axes and combine results
        """
        # Standardize IMU data (z-score normalization)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df data must be a pandas DataFrame.")
        
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
        
        acc_x = df['acc_x'] 
        acc_y = df['acc_y']
        acc_z = df['acc_z']
        gyr_x = df['gyr_x'] 
        gyr_y = df['gyr_y']
        gyr_z = df['gyr_z']
        labels = df['label']
        
        signals = {
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'acc_mag': df['acc_mag'],
            'gyr_x': gyr_x,
            'gyr_y': gyr_y,
            'gyr_z': gyr_z,
            'gyr_mag': df['gyr_mag']
        }
            
        all_results = []
        # Store labels from first signal only to avoid duplication
        self.all_labels = None
            
        for signal_name, signal_data in tqdm(signals.items(), desc='Analyzing '):
            result_df, window_labels = self.analyze_signal(signal_data, labels, signal_name)
            all_results.append(result_df)
            
            # Store labels from first signal analysis
            if self.all_labels is None:
                self.all_labels = window_labels
            
        # Combine all results
        self.features = pd.concat(all_results, ignore_index=True)
            
        # Create summary pivot table for easier analysis
        self.results = self.create_summary_table()
        
        return self.features, self.results
    
    def create_summary_table(self):
        """
        Create a summary table with entropy measures as columns
        """
        if self.features is None:
            return None
            
        measures = ['permutation_entropy', 'spectral_entropy', 'mean', 'std', 
                    'range', 'rms', 'var', 'min', 'max'
                    'time_energy',
                    # , 'spectral_energy', 
                    'spectral_centroid', 'spectral_spread', 
                    'spectral_flatness', 'peak_frequency_ratio', 'hjorth_frequency',
                    'spectral_skewness', 'spectral_kurtosis', 'snr']
        
        # Get unique time windows
        unique_times = sorted(self.features['center_time'].unique())
        
        summary_data = []
        
        for time_point in unique_times:
            time_data = self.features[self.features['center_time'] == time_point]
                
            row = {
                'center_time': time_point, 
                'start_time': time_data.iloc[0]['start_time'], 
                'end_time': time_data.iloc[0]['end_time']
            }
                
            # Add features for each signal
            for _, signal_row in time_data.iterrows():
                signal_name = signal_row['signal_name']
                for measure in measures:
                    col_name = f"{signal_name}_{measure}"
                    row[col_name] = signal_row[measure]
                
            summary_data.append(row)
        
        # Create DataFrame outside the loop
        df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
        
        # Add labels if available
        if self.all_labels is not None and len(self.all_labels) == len(df):
            df['label'] = self.all_labels
        elif self.all_labels is not None:
            print(f"Warning: Label count mismatch. Expected {len(df)}, got {len(self.all_labels)}")
            
        return df