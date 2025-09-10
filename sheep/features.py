import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import welch

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
        if round(self.window_duration * self.fs) < 3:
            raise ValueError("Window too small - need at least 3 samples for calculation")

    def apply_window_function(self, signal_data):
        """Apply windowing function - kept for compatibility but simplified"""
        if self.window_type == 'rectangular' or self.window_type not in ['hann', 'hamming', 'blackman', 'bartlett']:
            return signal_data
        else:
            n_samples = len(signal_data)
            if self.window_type == 'hann':
                w = np.hanning(n_samples)
            elif self.window_type == 'hamming':
                w = np.hamming(n_samples)
            elif self.window_type == 'blackman':
                w = np.blackman(n_samples)
            elif self.window_type == 'bartlett':
                w = np.bartlett(n_samples)
            else:
                w = np.ones(n_samples)
            return signal_data * w

    def calculate_static_acceleration(self, signal_data):
        """Calculate static acceleration as mean over window"""
        return np.mean(signal_data)

    def calculate_dynamic_acceleration(self, signal_data, static_acc):
        """Calculate dynamic acceleration: DyX = StX - RawX"""
        return static_acc - signal_data

    def calculate_vedba(self, dyn_x, dyn_y, dyn_z):
        """Calculate Vectorial Dynamic Body Acceleration"""
        return np.sqrt(dyn_x**2 + dyn_y**2 + dyn_z**2)

    def calculate_pitch(self, st_z):
        """Calculate pitch: Asin(StZ)"""
        # Clamp to valid range for arcsin
        st_z_clamped = np.clip(st_z, -1, 1)
        return np.arcsin(st_z_clamped)

    def calculate_sway(self, st_y):
        """Calculate sway: Asin(StY)"""
        # Clamp to valid range for arcsin
        st_y_clamped = np.clip(st_y, -1, 1)
        return np.arcsin(st_y_clamped)

    def calculate_power_spectrum_features(self, signal_data):
        """Calculate Power Spectrum Density features"""
        try:
            freqs, psd = welch(signal_data, fs=self.fs, nperseg=min(256, len(signal_data)))
            
            # Remove DC component
            if len(freqs) > 1:
                freqs = freqs[1:]
                psd = psd[1:]
            
            if len(psd) == 0 or np.sum(psd) == 0:
                return {'psd1': 0, 'freq1': 0, 'psd2': 0, 'freq2': 0}
            
            # Find first and second maximum PSD values and their frequencies
            sorted_indices = np.argsort(psd)[::-1]  # Sort in descending order
            
            psd1 = psd[sorted_indices[0]] if len(sorted_indices) > 0 else 0
            freq1 = freqs[sorted_indices[0]] if len(sorted_indices) > 0 else 0
            
            psd2 = psd[sorted_indices[1]] if len(sorted_indices) > 1 else 0
            freq2 = freqs[sorted_indices[1]] if len(sorted_indices) > 1 else 0
            
            return {
                'psd1': psd1,
                'freq1': freq1,
                'psd2': psd2,
                'freq2': freq2
            }
        except:
            return {'psd1': 0, 'freq1': 0, 'psd2': 0, 'freq2': 0}

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

    def analyze_signal_window(self, acc_x_window, acc_y_window, acc_z_window, 
                             gyr_x_window, gyr_y_window, gyr_z_window, 
                             start_idx, end_idx, label):
        """Analyze a single window and extract sheep paper features"""
        
        # Apply windowing function
        acc_x_windowed = self.apply_window_function(acc_x_window)
        acc_y_windowed = self.apply_window_function(acc_y_window)
        acc_z_windowed = self.apply_window_function(acc_z_window)
        gyr_x_windowed = self.apply_window_function(gyr_x_window)
        gyr_y_windowed = self.apply_window_function(gyr_y_window)
        gyr_z_windowed = self.apply_window_function(gyr_z_window)
        
        # Calculate static acceleration for each axis
        st_x = self.calculate_static_acceleration(acc_x_windowed)
        st_y = self.calculate_static_acceleration(acc_y_windowed)
        st_z = self.calculate_static_acceleration(acc_z_windowed)
        
        # Calculate dynamic acceleration for each axis
        dyn_x = acc_x_windowed - st_x
        dyn_y = acc_y_windowed - st_y
        dyn_z = acc_z_windowed - st_z
        
        # Calculate VeDBA and smoothed VeDBA
        vedba_values = self.calculate_vedba(dyn_x, dyn_y, dyn_z)
        vedba_smoothed = np.mean(vedba_values)  # Smoothed VeDBA as mean
        
        # Calculate pitch and sway
        pitch = self.calculate_pitch(st_z)
        sway = self.calculate_sway(st_y)
        
        # Calculate Power Spectrum Density features for each axis
        psd_x = self.calculate_power_spectrum_features(dyn_x)
        psd_y = self.calculate_power_spectrum_features(dyn_y)
        psd_z = self.calculate_power_spectrum_features(dyn_z)
        
        # Build feature dictionary following the paper's naming convention
        feature_dict = {
            # Window information
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_time': start_idx / self.fs,
            'end_time': end_idx / self.fs,
            'center_time': (start_idx + end_idx) / 2 / self.fs,
            'label': label,
            
            # Static acceleration (mean, std, min, max)
            'mean_st_x': st_x,
            'mean_st_y': st_y,
            'mean_st_z': st_z,
            'sd_st_x': np.std(acc_x_windowed),
            'sd_st_y': np.std(acc_y_windowed),
            'sd_st_z': np.std(acc_z_windowed),
            'min_st_x': np.min(acc_x_windowed),
            'min_st_y': np.min(acc_y_windowed),
            'min_st_z': np.min(acc_z_windowed),
            'max_st_x': np.max(acc_x_windowed),
            'max_st_y': np.max(acc_y_windowed),
            'max_st_z': np.max(acc_z_windowed),
            
            # Pitch and Sway (mean, std, min, max)
            'mean_pitch': pitch,
            'mean_sway': sway,
            'sd_pitch': np.std([pitch]),  # Single value, so std is 0
            'sd_sway': np.std([sway]),
            'min_pitch': pitch,
            'min_sway': sway,
            'max_pitch': pitch,
            'max_sway': sway,
            
            # Smoothed VeDBA (mean, std, min, max)
            'mean_vedba_s': vedba_smoothed,
            'sd_vedba_s': np.std(vedba_values),
            'min_vedba_s': np.min(vedba_values),
            'max_vedba_s': np.max(vedba_values),
            
            # Power Spectrum Density features for each axis
            'psd1_x': psd_x['psd1'],
            'psd2_x': psd_x['psd2'],
            'psd1_y': psd_y['psd1'],
            'psd2_y': psd_y['psd2'],
            'psd1_z': psd_z['psd1'],
            'psd2_z': psd_z['psd2'],
            
            # Gyroscope features (basic statistics)
            'mean_gyr_x': np.mean(gyr_x_windowed),
            'mean_gyr_y': np.mean(gyr_y_windowed),
            'mean_gyr_z': np.mean(gyr_z_windowed),
            'sd_gyr_x': np.std(gyr_x_windowed),
            'sd_gyr_y': np.std(gyr_y_windowed),
            'sd_gyr_z': np.std(gyr_z_windowed),
            'min_gyr_x': np.min(gyr_x_windowed),
            'min_gyr_y': np.min(gyr_y_windowed),
            'min_gyr_z': np.min(gyr_z_windowed),
            'max_gyr_x': np.max(gyr_x_windowed),
            'max_gyr_y': np.max(gyr_y_windowed),
            'max_gyr_z': np.max(gyr_z_windowed),
        }
        
        return feature_dict

    def analyze_multi_axis_imu(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'label']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        if self.global_windows is None or self.all_labels is None:
            self.compute_global_windows(len(df), df['label'])

        # Extract signals
        acc_x = np.array(df['acc_x'])
        acc_y = np.array(df['acc_y'])
        acc_z = np.array(df['acc_z'])
        gyr_x = np.array(df['gyr_x'])
        gyr_y = np.array(df['gyr_y'])
        gyr_z = np.array(df['gyr_z'])

        all_features = []
        
        for (start_idx, end_idx), label in tqdm(zip(self.global_windows, self.all_labels), 
                                               desc='Extracting sheep paper features',
                                               total=len(self.global_windows)):
            
            # Extract window data
            acc_x_window = acc_x[start_idx:end_idx]
            acc_y_window = acc_y[start_idx:end_idx]
            acc_z_window = acc_z[start_idx:end_idx]
            gyr_x_window = gyr_x[start_idx:end_idx]
            gyr_y_window = gyr_y[start_idx:end_idx]
            gyr_z_window = gyr_z[start_idx:end_idx]
            
            # Extract features for this window
            window_features = self.analyze_signal_window(
                acc_x_window, acc_y_window, acc_z_window,
                gyr_x_window, gyr_y_window, gyr_z_window,
                start_idx, end_idx, label
            )
            
            all_features.append(window_features)

        self.features = pd.DataFrame(all_features)
        self.results = self.features.copy()  # For compatibility
        
        return self.features, self.results

    def create_summary_table(self):
        """Return the features dataframe as summary"""
        return self.features if self.features is not None else None