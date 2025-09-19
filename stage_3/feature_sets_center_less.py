import pandas as pd
import numpy as np
from tqdm import tqdm
from neurokit2 import entropy_permutation, entropy_spectral
from scipy.signal import welch, windows

class GenerateFeatures:
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
        if round(self.window_duration * self.fs) < 3:
            raise ValueError("Window too small - need at least 3 samples for entropy calculation")

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
            w = windows.tukey(n_samples)
        else:
            w = np.ones(n_samples)
        return signal_data * w

    def calculate_energy(self, signal_data):
        return np.sum(signal_data**2)

    def calculate_spectral_energy(self, signal_data):
        fft_vals = np.fft.fft(signal_data)
        return np.sum(np.abs(fft_vals)**2)

    # Frequency and Hjorth functions remain unchanged
    def calculate_frequency_features(self, signal_data):
        freqs, psd = welch(signal_data, fs=self.fs, nperseg=min(256, len(signal_data)))
        if len(freqs) > 1:
            freqs = freqs[1:]
            psd = psd[1:]
        if len(psd) == 0 or np.sum(psd) == 0:
            return {key: 0 for key in ['spectral_centroid', 'spectral_spread', 'spectral_flatness',
                                    'peak_frequency_ratio', 'hjorth_frequency', 'spectral_skewness',
                                    'spectral_kurtosis', 'snr']}
        psd_norm = psd / np.sum(psd)
        spectral_centroid = np.sum(freqs * psd_norm)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        peak_power = np.max(psd)
        total_power = np.sum(psd)
        peak_frequency_ratio = peak_power / (total_power + 1e-10)
        hjorth_freq = self.calculate_hjorth_frequency(signal_data)
        spectral_skewness = np.sum(((freqs - spectral_centroid) ** 3) * psd_norm) / (spectral_spread ** 3 + 1e-10)
        spectral_kurtosis = np.sum(((freqs - spectral_centroid) ** 4) * psd_norm) / (spectral_spread ** 4 + 1e-10)
        signal_power = peak_power
        noise_power = total_power - signal_power
        snr = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_flatness': spectral_flatness,
            'peak_frequency_ratio': peak_frequency_ratio,
            'hjorth_frequency': hjorth_freq,
            'spectral_skewness': spectral_skewness,
            'spectral_kurtosis': spectral_kurtosis,
            'snr': snr
        }

    def calculate_hjorth_frequency(self, signal_data):
        if len(signal_data) < 3:
            return float('nan')
        d1 = np.diff(signal_data) * self.fs
        d2 = np.diff(d1) * self.fs
        if len(d1) == 0 or len(d2) == 0:
            return 0
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        if var_d1 < 1e-10:
            return 0.0
        return np.sqrt(var_d2 / var_d1) / (2 * np.pi)

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

    def analyze_signal(self, axis, axis_name="axis"):
        signal = np.array(axis)
        window_features = []
        for (start_idx, end_idx), label in zip(self.global_windows, self.all_labels):
            window_signal = signal[start_idx:end_idx]
            windowed_signal = self.apply_window_function(window_signal)
            try:
                perm_ent = entropy_permutation(windowed_signal)[0]
                spectral_ent = entropy_spectral(windowed_signal)[0]
            except:
                perm_ent = np.nan
                spectral_ent = np.nan
            time_energy = self.calculate_energy(windowed_signal)
            freq_features = self.calculate_frequency_features(windowed_signal)
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
                'min': np.min(windowed_signal),
                'max': np.max(windowed_signal),
                'time_energy': time_energy,
                'label': label
            }
            feature_dict.update(freq_features)
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
        for name, sig in tqdm(signals.items(), desc='Analyzing'):
            result_df = self.analyze_signal(sig, axis_name=name)
            all_results.append(result_df)

        self.features = pd.concat(all_results, ignore_index=True)
        self.results = self.create_summary_table()
        return self.features, self.results

    def create_summary_table(self):
        if self.features is None:
            return None
        measures = ['permutation_entropy','spectral_entropy', 'mean', 'std', 'range', 'rms',
                    'min', 'max', 'time_energy', 'spectral_centroid', 'spectral_spread',
                    'spectral_flatness', 'peak_frequency_ratio', 'hjorth_frequency',
                    'spectral_skewness', 'spectral_kurtosis', 'snr']
        unique_times = sorted(self.features['center_time'].unique())
        summary_data = []
        for t in unique_times:
            time_data = self.features[self.features['center_time'] == t]
            row = {'center_time': t, 'start_time': time_data.iloc[0]['start_time'],
                'end_time': time_data.iloc[0]['end_time']}
            for _, sig_row in time_data.iterrows():
                signal_name = sig_row['signal_name']
                for measure in measures:
                    row[f"{signal_name}_{measure}"] = sig_row[measure]
            summary_data.append(row)
        df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
        if self.all_labels is not None and len(self.all_labels) == len(df):
            df['label'] = self.all_labels
        return df