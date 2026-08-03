"""
feature_extraction.py - Step 2: Extract EEG Features

Loads the processed EEG windows (from Step 1) and extracts:
- Band powers (Delta, Theta, Alpha, Beta, Gamma)
- Band power ratios
- Statistical features (mean, std, skewness, kurtosis)

Applies various feature selection methods (ANOVA, Feature Importance,
Linear Correlation, PCA) and saves the resulting feature matrices.

Usage: python feature_extraction.py
"""

import os
import numpy as np
from scipy import signal, stats
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import time

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = 'processed_data'
OUTPUT_DIR = 'processed_data'

# EEG Frequency Bands (Hz)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


# ============================================================
# Feature Extraction Functions
# ============================================================

def calculate_band_powers(data, fs):
    """
    Calculate absolute band powers for each frequency band using Welch's method.
    
    Args:
        data: shape (n_samples, n_channels) - a single 1s window
        fs: sampling rate (125 Hz)
    Returns:
        band_powers: shape (n_channels, 5) - power for each band per channel
    """
    n_channels = data.shape[1]
    band_powers = np.zeros((n_channels, len(BANDS)))
    
    for ch in range(n_channels):
        # Welch's method to estimate PSD
        # nperseg=fs means 1s window (which is the whole data segment here)
        freqs, psd = signal.welch(data[:, ch], fs, nperseg=fs, scaling='density')
        
        # Calculate power in each band
        for i, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
            # Find indices of frequencies in this band
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            # Integrate PSD over the frequency band (Simpson's rule could be used, 
            # but simple sum * df is fine for this approximation)
            bp = np.sum(psd[idx_band]) * (freqs[1] - freqs[0])
            band_powers[ch, i] = bp
            
    return band_powers


def extract_window_features(window, fs):
    """
    Extract all features for a single 1-second EEG window.
    
    Args:
        window: shape (125, 7) - 1s of data at 125 Hz for 7 channels
        fs: sampling rate
    Returns:
        1D feature vector for this window (size 84)
    """
    n_channels = window.shape[1]
    features = []
    
    # 1. Band Powers (5 per channel)
    bps = calculate_band_powers(window, fs)
    features.extend(bps.flatten())
    
    # 2. Band Ratios (3 per channel)
    # bps columns: 0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma
    for ch in range(n_channels):
        theta = bps[ch, 1]
        alpha = bps[ch, 2]
        beta = bps[ch, 3]
        
        # Add small epsilon to prevent division by zero
        eps = 1e-10
        alpha_beta = alpha / (beta + eps)
        theta_beta = theta / (beta + eps)
        theta_alpha = theta / (alpha + eps)
        
        features.extend([alpha_beta, theta_beta, theta_alpha])
    
    # 3. Statistical Features (4 per channel)
    for ch in range(n_channels):
        ch_data = window[:, ch]
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        skew_val = stats.skew(ch_data)
        kurt_val = stats.kurtosis(ch_data)
        
        features.extend([mean_val, std_val, skew_val, kurt_val])
        
    return np.array(features)


def process_dataset_features(X, fs):
    """
    Extract features for all windows in a dataset.
    
    Args:
        X: shape (n_windows, window_size, n_channels)
        fs: sampling rate
    Returns:
        feature matrix: shape (n_windows, n_features)
    """
    n_windows = X.shape[0]
    # Calculate feature vector size for first window to preallocate
    first_feat = extract_window_features(X[0], fs)
    n_features = len(first_feat)
    
    X_feat = np.zeros((n_windows, n_features))
    X_feat[0] = first_feat
    
    for i in range(1, n_windows):
        X_feat[i] = extract_window_features(X[i], fs)
        
    return X_feat


# ============================================================
# Feature Selection Methods
# ============================================================

def select_features(X_train, y_train, X_test, method='none'):
    """
    Apply feature selection method.
    
    Args:
        X_train, y_train: training data
        X_test: test data
        method: 'none', 'anova', 'fi', 'lcc', 'pca'
    Returns:
        X_train_sel, X_test_sel
    """
    if method == 'none':
        return X_train, X_test
        
    elif method == 'anova':
        # ANOVA F-test (p <= 0.05)
        # SelectKBest expects a fixed k, so we compute all p-values first
        f_vals, p_vals = f_classif(X_train, y_train)
        selected = p_vals <= 0.05
        
        # If no features selected, fallback to top 10 to prevent empty matrix
        if not np.any(selected):
            print("  ANOVA found 0 features p<=0.05, keeping top 10")
            idx = np.argsort(p_vals)[:10]
            selected = np.zeros(len(p_vals), dtype=bool)
            selected[idx] = True
            
        return X_train[:, selected], X_test[:, selected]
        
    elif method == 'fi':
        # Feature Importance (Random Forest)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        threshold = np.mean(importances)
        selected = importances >= threshold
        
        if not np.any(selected):
            # Fallback
            selected = importances > 0
            
        return X_train[:, selected], X_test[:, selected]
        
    elif method == 'lcc':
        # Linear Correlation Coefficient
        corrs = np.array([np.abs(np.corrcoef(X_train[:, i], y_train)[0, 1]) 
                          for i in range(X_train.shape[1])])
        # Handle NaNs if any feature is constant
        corrs = np.nan_to_num(corrs)
        threshold = np.mean(corrs)
        selected = corrs >= threshold
        
        if not np.any(selected):
            idx = np.argsort(corrs)[-10:] # Top 10
            selected = np.zeros(len(corrs), dtype=bool)
            selected[idx] = True
            
        return X_train[:, selected], X_test[:, selected]
        
    elif method == 'pca':
        # PCA with mle (automatic components)
        # PCA requires scaled data, but we'll scale inside the training pipeline.
        # However, for PCA transformation it's better to scale here just for the PCA step.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # 'mle' can sometimes fail if n_samples < n_features, use fixed variance if so
        try:
            pca = PCA(n_components='mle', random_state=42)
            X_train_pca = pca.fit_transform(X_train_sc)
        except Exception:
            # Fallback to retaining 95% variance
            pca = PCA(n_components=0.95, random_state=42)
            X_train_pca = pca.fit_transform(X_train_sc)
            
        X_test_pca = pca.transform(X_test_sc)
        return X_train_pca, X_test_pca
        
    else:
        raise ValueError(f"Unknown method {method}")


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("EEG Feature Extraction & Selection Pipeline")
    print("=" * 60)
    
    methods = ['none', 'anova', 'fi', 'lcc', 'pca']
    
    # Find all processed npz files
    npz_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.npz') and not '_features' in f])
    
    print(f"Found {len(npz_files)} processed subject files.")
    
    total_start = time.time()
    
    for i, npz_file in enumerate(npz_files):
        subj_name = npz_file.split('.')[0]
        print(f"\n[{i+1}/{len(npz_files)}] Extracting features for {subj_name}...")
        start = time.time()
        
        # Load processed windows
        data = np.load(os.path.join(INPUT_DIR, npz_file))
        X_train_raw = data['X_train']
        y_train = data['y_train']
        X_test_raw = data['X_test']
        y_test = data['y_test']
        fs = int(data['fs'])
        
        # 1. Extract base features
        print(f"  Extracting base features (84 total)...", end='', flush=True)
        t0 = time.time()
        X_train_feat = process_dataset_features(X_train_raw, fs)
        X_test_feat = process_dataset_features(X_test_raw, fs)
        print(f" done ({time.time()-t0:.1f}s)")
        
        # 2. Apply feature selection methods and save
        results = {'y_train': y_train, 'y_test': y_test}
        
        for method in methods:
            print(f"  Selecting features ({method})...", end='', flush=True)
            t0 = time.time()
            X_tr_sel, X_te_sel = select_features(X_train_feat, y_train, X_test_feat, method)
            
            results[f'X_train_{method}'] = X_tr_sel
            results[f'X_test_{method}'] = X_te_sel
            print(f" done ({X_tr_sel.shape[1]} features, {time.time()-t0:.1f}s)")
            
        # Save feature file
        out_path = os.path.join(OUTPUT_DIR, f"{subj_name}_features.npz")
        np.savez_compressed(out_path, **results)
        
        print(f"  Saved {subj_name} in {time.time()-start:.1f}s")
        
    print(f"\n{'=' * 60}")
    print(f"Feature extraction complete in {time.time()-total_start:.1f}s")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
