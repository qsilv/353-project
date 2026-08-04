"""
data_processing.py - step 1: raw eeg data processing

this script reads the raw .txt files, picks the 7 channels i need, 
downsamples the data to 125 hz (so it matches the neuropawn headset), 
filters out noise, and cuts it into 1-second chunks.

i save the data into .npz files so it's easy to load for the next step.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import signal

# --- settings ---
DATA_FOLDER = 'MEMA Dataset/MEMA/For_graph'
OUTPUT_FOLDER = 'processed_data'

# i are dropping the sampling rate from 500 hz to 125 hz
ORIGINAL_HZ = 500
TARGET_HZ = 125
DOWNSAMPLE_FACTOR = 4

# the 7 channels i are using based on the research papers
CHANNEL_COLUMNS = [2, 3, 4, 12, 13, 14, 23]
CHANNEL_NAMES = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'Pz']

# a 1-second window at 125 hz means 125 samples
SAMPLES_PER_WINDOW = 125

# --- helper functions ---

def downsample_data(raw_eeg_data):
    """
    makes the data smaller (125 hz instead of 500 hz).
    i use scipy's decimate function because it automatically 
    applies a filter to stop aliasing (weird artifacts).
    """
    num_samples = raw_eeg_data.shape[0] #number of rows (time points)
    num_channels = raw_eeg_data.shape[1] #number of columns (channels)
    new_num_samples = num_samples // DOWNSAMPLE_FACTOR #how many rows new data will have
    
    #new empty array to store smaller data
    smaller_data = np.zeros((new_num_samples, num_channels))
    
    # process one channel at a time
    for channel_index in range(num_channels):
        channel_data = raw_eeg_data[:, channel_index] #all rows, selected channel column
        #applies a low pass filter to prevent aliasing then downsamples by the factor
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
        smaller_data[:, channel_index] = signal.decimate(channel_data, DOWNSAMPLE_FACTOR)
        
    return smaller_data


def filter_data(eeg_data):
    """
    removes low-frequency drift and high-frequency noise.
    i keep frequencies between 0.5 hz and 45.0 hz.

    why: less than 0.5hz is slow muscle movements (artifacts) and higher than 45hz is noise
    """
    # create a butterworth bandpass filter
    #nyquist theorem: the highest frequency i can detect is half the sampling rate (learned this in CMPT 365)
    nyquist_freq = TARGET_HZ / 2.0
    low_cutoff = 0.5 / nyquist_freq #filters want normalized frequencies between 0 and 1
    high_cutoff = 45.0 / nyquist_freq 
    
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    
    filtered_data = np.zeros_like(eeg_data)
    
    # filter one channel at a time
    for channel_index in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, channel_index]
        filtered_data[:, channel_index] = signal.filtfilt(b, a, channel_data)
        
    return filtered_data


def cut_into_windows(eeg_data):
    """
    cuts the continuous eeg data into 1 second chunks.
    drops any leftover data at the end that doesn't fit into a full second.
    """
    num_samples = eeg_data.shape[0]
    num_channels = eeg_data.shape[1]
    
    num_full_windows = num_samples // SAMPLES_PER_WINDOW
    
    # cut off the extra samples at the end
    total_samples_to_keep = num_full_windows * SAMPLES_PER_WINDOW
    trimmed_data = eeg_data[0:total_samples_to_keep, :]
    
    # reshape it so it is (windows, 125, 7)
    windows = trimmed_data.reshape(num_full_windows, SAMPLES_PER_WINDOW, num_channels)
    
    return windows


def get_label_from_filename(filename):
    """
    looks at the filename to figure out if the person was focused or not.
    _a means concentrating (focused = 1)
    _n means neutral (unfocused = 0)
    _r means relaxing (unfocused = 0)

    this is based on the dataset description on the repository
    """
    if '_a' in filename:
        return 1, 'focused'
    elif '_n' in filename or '_r' in filename:
        return 0, 'unfocused'
    else:
        return -1, 'unknown'


def get_round_from_filename(filename):
    """
    finds out which round of the experiment this file is from.
    rounds 1, 2, and 3 are for training. round 4 is for testing.
    """
    if '1.txt' in filename or '2.txt' in filename or '3.txt' in filename:
        return 'train'
    elif '4.txt' in filename:
        return 'test'
    else:
        return 'unknown'


# --- main process ---

def main():
    print("Starting data processing")
    
    # create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # get a list of all subjects (subject1, subject2, etc.)
    all_items_in_folder = os.listdir(DATA_FOLDER)
    subject_folders = []
    for item in all_items_in_folder:
        if 'Subject' in item:
            subject_folders.append(item)
            
    print(f"Found {len(subject_folders)} subjects to process.")
    
    start_time = time.time()
    
    # process each subject one by one
    for subject in subject_folders:
        print(f"Working on {subject}")
        
        subject_path = os.path.join(DATA_FOLDER, subject)
        txt_files = os.listdir(subject_path)
        
        # lists to hold the data for this subject
        train_windows_list = []
        train_labels_list = []
        test_windows_list = []
        test_labels_list = []
        
        for file in txt_files:
            # skip files that aren't .txt
            if not file.endswith('.txt'):
                continue
                
            file_path = os.path.join(subject_path, file)
            
            # figure out the label and if it's train or test
            label_number, label_name = get_label_from_filename(file)
            dataset_type = get_round_from_filename(file)
            
            if label_number == -1 or dataset_type == 'unknown':
                continue
                
            # 1. read the raw data using pandas (only the 7 columns i need)
            df = pd.read_csv(file_path, header=None, usecols=CHANNEL_COLUMNS)
            raw_data = df.values
            
            # 2. downsample
            smaller_data = downsample_data(raw_data)
            
            # 3. filter
            clean_data = filter_data(smaller_data)
            
            # 4. cut into windows
            windows = cut_into_windows(clean_data)
            
            # create a label for each window (focused/unfocused)
            num_windows = windows.shape[0]
            labels = np.zeros(num_windows, dtype=int)
            for i in range(num_windows):
                labels[i] = label_number
                
            # put them in the right list
            if dataset_type == 'train':
                train_windows_list.append(windows)
                train_labels_list.append(labels)
            elif dataset_type == 'test':
                test_windows_list.append(windows)
                test_labels_list.append(labels)
                
        # combine all the lists into big numpy arrays
        X_train = np.concatenate(train_windows_list, axis=0)
        y_train = np.concatenate(train_labels_list, axis=0)
        X_test = np.concatenate(test_windows_list, axis=0)
        y_test = np.concatenate(test_labels_list, axis=0)
        
        # save the arrays to a file so they can be used later
        output_file = os.path.join(OUTPUT_FOLDER, f"{subject}.npz")
        np.savez_compressed(
            output_file,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            channel_names=CHANNEL_NAMES,
            fs=TARGET_HZ
        )
        
    end_time = time.time()
    print(f"Finished processing everything in {end_time - start_time:.1f} seconds")

if __name__ == '__main__':
    main()
