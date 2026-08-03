"""
data_processing.py - Step 1: Raw EEG Data Processing

Reads raw .txt EEG files from the MEMA For_graph directory,
selects 7 channels, downsamples to 125 Hz, applies bandpass
filtering, segments into 1-second windows, and saves processed
data as .npz files.

Usage: python data_processing.py
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import time

# ============================================================
# Configuration
# ============================================================

# Path to the For_graph directory containing raw .txt trial files
DATA_DIR = os.path.join('MEMA Dataset', 'MEMA', 'For_graph')

# Output directory for processed data
OUTPUT_DIR = 'processed_data'

# Original and target sampling rates
FS_ORIGINAL = 500   # Hz (MEMA dataset recorded at 500 Hz)
FS_TARGET = 125     # Hz (NeuroPawn headset rate)
DOWNSAMPLE_FACTOR = FS_ORIGINAL // FS_TARGET  # = 4

# 7 channels used in Papers 1 & 3 (Aci et al., Wang & Kim)
# Column indices in the 32-column .txt files
CHANNEL_COLS = [2, 3, 4, 12, 13, 14, 23]
CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz']

# Window size for segmentation (1 second at target rate)
WINDOW_SIZE = FS_TARGET  # 125 samples = 1 second

# Bandpass filter parameters
FILTER_LOW = 0.5    # Hz
FILTER_HIGH = 45.0  # Hz
FILTER_ORDER = 4

# Binary label mapping from filename
# _a = concentrating = focused (1)
# _n = neutral = unfocused (0)
# _r = relaxing = unfocused (0)
LABEL_MAP = {'a': 1, 'n': 0, 'r': 0}

# Train/test split: rounds 1-3 for training, round 4 for testing
TRAIN_ROUNDS = ['1', '2', '3']
TEST_ROUNDS = ['4']


# ============================================================
# Helper Functions
# ============================================================

def load_trial_txt(filepath):
    """
    Load a single trial .txt file and select the 7 target channels.
    
    The .txt files are comma-separated with no header, 32 columns,
    one row per time point at 500 Hz.
    
    Returns:
        numpy array of shape (n_samples, 7)
    """
    # Use usecols to only read the 7 channels we need (much faster)
    df = pd.read_csv(filepath, header=None, usecols=CHANNEL_COLS)
    
    # Reorder columns to match our CHANNEL_NAMES order
    # pd.read_csv with usecols preserves original column indices
    df.columns = CHANNEL_NAMES
    
    return df.values


def downsample(data, factor):
    """
    Downsample EEG data by the given factor with anti-aliasing.
    
    Uses scipy.signal.decimate which applies a low-pass filter
    before downsampling to prevent aliasing.
    
    Args:
        data: numpy array of shape (n_samples, n_channels)
        factor: integer downsample factor (e.g., 4 for 500->125 Hz)
    
    Returns:
        numpy array of shape (n_samples // factor, n_channels)
    """
    # decimate works along axis=0 by default, apply per channel
    downsampled = np.zeros((data.shape[0] // factor, data.shape[1]))
    for ch in range(data.shape[1]):
        downsampled[:, ch] = signal.decimate(data[:, ch], factor)
    return downsampled


def bandpass_filter(data, lowcut, highcut, fs, order):
    """
    Apply a Butterworth bandpass filter to EEG data.
    
    Removes DC drift (below lowcut) and high-frequency noise
    (above highcut). Uses filtfilt for zero-phase filtering.
    
    Args:
        data: numpy array of shape (n_samples, n_channels)
        lowcut: low frequency cutoff in Hz
        highcut: high frequency cutoff in Hz
        fs: sampling rate in Hz
        order: filter order
    
    Returns:
        filtered data, same shape as input
    """
    # Design Butterworth bandpass filter
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering to each channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
    
    return filtered


def segment_into_windows(data, window_size):
    """
    Split continuous EEG data into non-overlapping fixed-size windows.
    
    Any leftover samples that don't fill a complete window are dropped.
    
    Args:
        data: numpy array of shape (n_samples, n_channels)
        window_size: number of samples per window
    
    Returns:
        numpy array of shape (n_windows, window_size, n_channels)
    """
    n_samples = data.shape[0]
    n_windows = n_samples // window_size
    
    # Trim to exact multiple of window_size
    trimmed = data[:n_windows * window_size, :]
    
    # Reshape into windows
    windows = trimmed.reshape(n_windows, window_size, data.shape[1])
    
    return windows


def parse_trial_filename(filename):
    """
    Extract the attention state and round number from a trial filename.
    
    Filenames follow the pattern: SubjectN_X#.txt
    where X is the state letter (a/n/r) and # is the round (1-4).
    
    Examples:
        Subject1_a1.txt -> state='a', round='1'
        Subject3_r4.txt -> state='r', round='4'
    
    Returns:
        (state_letter, round_number) or (None, None) if not a trial file
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # The trial part is after the last underscore
    parts = name.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    
    trial_code = parts[1]
    
    # Trial code should be like 'a1', 'n3', 'r4'
    if len(trial_code) < 2:
        return None, None
    
    state = trial_code[0]
    round_num = trial_code[1:]
    
    if state not in LABEL_MAP:
        return None, None
    
    return state, round_num


def process_subject(subject_dir, subject_name):
    """
    Process all trial files for a single subject.
    
    Reads each trial .txt file, downsamples, filters, segments
    into 1-second windows, assigns binary labels, and splits
    into train/test sets.
    
    Args:
        subject_dir: path to the subject's directory
        subject_name: string like 'Subject1'
    
    Returns:
        dict with X_train, y_train, X_test, y_test arrays
    """
    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []
    
    # Find all .txt trial files
    txt_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.txt')])
    
    for txt_file in txt_files:
        state, round_num = parse_trial_filename(txt_file)
        if state is None:
            continue  # Skip non-trial files
        
        filepath = os.path.join(subject_dir, txt_file)
        
        # Step 1: Load raw data (7 channels only)
        raw_data = load_trial_txt(filepath)
        
        # Step 2: Downsample 500 Hz -> 125 Hz
        downsampled = downsample(raw_data, DOWNSAMPLE_FACTOR)
        
        # Step 3: Bandpass filter (0.5-45 Hz)
        filtered = bandpass_filter(downsampled, FILTER_LOW, FILTER_HIGH,
                                   FS_TARGET, FILTER_ORDER)
        
        # Step 4: Segment into 1-second windows
        windows = segment_into_windows(filtered, WINDOW_SIZE)
        
        # Step 5: Assign binary label
        label = LABEL_MAP[state]
        labels = np.full(windows.shape[0], label, dtype=np.int32)
        
        # Step 6: Split into train or test
        if round_num in TRAIN_ROUNDS:
            train_windows.append(windows)
            train_labels.append(labels)
        elif round_num in TEST_ROUNDS:
            test_windows.append(windows)
            test_labels.append(labels)
    
    # Combine all windows for this subject
    result = {
        'X_train': np.concatenate(train_windows, axis=0),
        'y_train': np.concatenate(train_labels, axis=0),
        'X_test': np.concatenate(test_windows, axis=0),
        'y_test': np.concatenate(test_labels, axis=0),
        'channel_names': np.array(CHANNEL_NAMES),
        'fs': FS_TARGET,
    }
    
    return result


# ============================================================
# Main Processing Pipeline
# ============================================================

def main():
    """Process all subjects and save as .npz files."""
    
    print("=" * 60)
    print("EEG Data Processing Pipeline")
    print(f"  Source: {DATA_DIR}")
    print(f"  Channels: {CHANNEL_NAMES} (7 of 32)")
    print(f"  Downsample: {FS_ORIGINAL} Hz -> {FS_TARGET} Hz")
    print(f"  Filter: {FILTER_LOW}-{FILTER_HIGH} Hz bandpass")
    print(f"  Window: {WINDOW_SIZE} samples = 1 second")
    print(f"  Labels: focused (1) vs unfocused (0)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all subject directories
    subject_dirs = sorted([
        d for d in os.listdir(DATA_DIR) 
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith('Subject')
    ])
    
    print(f"\nFound {len(subject_dirs)} subjects")
    
    total_start = time.time()
    
    for i, subject_name in enumerate(subject_dirs):
        subject_dir = os.path.join(DATA_DIR, subject_name)
        start = time.time()
        
        print(f"\n[{i+1}/{len(subject_dirs)}] Processing {subject_name}...", end=' ')
        
        try:
            result = process_subject(subject_dir, subject_name)
            
            # Save processed data
            output_path = os.path.join(OUTPUT_DIR, f'{subject_name}.npz')
            np.savez_compressed(output_path, **result)
            
            elapsed = time.time() - start
            
            # Print summary
            n_train = result['X_train'].shape[0]
            n_test = result['X_test'].shape[0]
            train_pos = result['y_train'].sum()
            test_pos = result['y_test'].sum()
            print(f"done ({elapsed:.1f}s)")
            print(f"    Train: {n_train} windows "
                  f"({train_pos} focused, {n_train - train_pos} unfocused)")
            print(f"    Test:  {n_test} windows "
                  f"({test_pos} focused, {n_test - test_pos} unfocused)")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"All subjects processed in {total_elapsed:.1f}s")
    print(f"Output saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
