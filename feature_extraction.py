"""
feature_extraction.py - step 2: extract eeg features

this script takes the processed 1-second chunks and calculates mathematical 
features from them (like band powers, averages, and standard deviations).
then, it uses 4 different methods to select the most important features, 
which helps my machine learning models run faster and sometimes better.
"""

import os
import time
import numpy as np
from scipy import signal, stats
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- settings ---
INPUT_FOLDER = 'processed_data'
OUTPUT_FOLDER = 'processed_data'

# eeg frequency bands (in hz)
# brain waves are categorized by these frequencies
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)
GAMMA_BAND = (30, 45)


# --- feature extraction functions ---

def calculate_power_for_band(frequencies, power_spectrum, min_freq, max_freq):
    """
    calculates how much power is in a specific frequency band (like alpha or beta).
    it looks at the power_spectrum array and adds up all the values that fall
    between the min_freq and max_freq limits. power_sprectrum is the relative power of each frequency
    where power_spectrum and frequencies are 1D arrays given by welch.
    """
    total_power = 0
    # add up the power for all frequencies in my range
    for i in range(len(frequencies)):
        if frequencies[i] >= min_freq and frequencies[i] <= max_freq:
            total_power = total_power + power_spectrum[i]
    return total_power


def extract_features_for_one_window(window_data, sampling_rate):
    """
    calculates all 84 features for a single 1-second chunk of eeg data.
    there are 7 channels, and i calculate 12 features per channel.
    
    reference (paper 1): this extraction of statistical and frequency band 
    power features is derived from aci et al. (2019) "svm-based eeg attention classification".
    """
    num_channels = window_data.shape[1]
    features_list = []
    
    for channel in range(num_channels):
        channel_data = window_data[:, channel]
        
        # 1. calculate the frequency power spectrum using welch's method
        # frequencies is a 1d array of the frequencies, power_spectrum is a 1d array of the power at each frequency
        frequencies, power_spectrum = signal.welch(channel_data, sampling_rate, nperseg=sampling_rate)
        
        # calculate power for each brain wave band
        #this extracts the relative power of each brain wave band using the 2 arrays from welch
        delta_power = calculate_power_for_band(frequencies, power_spectrum, DELTA_BAND[0], DELTA_BAND[1])
        theta_power = calculate_power_for_band(frequencies, power_spectrum, THETA_BAND[0], THETA_BAND[1])
        alpha_power = calculate_power_for_band(frequencies, power_spectrum, ALPHA_BAND[0], ALPHA_BAND[1])
        beta_power = calculate_power_for_band(frequencies, power_spectrum, BETA_BAND[0], BETA_BAND[1])
        gamma_power = calculate_power_for_band(frequencies, power_spectrum, GAMMA_BAND[0], GAMMA_BAND[1])
        
        # 2. calculate band ratios (these are used in a couple focus/ADHD studies as a predictor of attention)
        #add a tiny number (0.0001) so i never divide by zero and crash the program
        alpha_beta_ratio = alpha_power / (beta_power + 0.0001)
        theta_beta_ratio = theta_power / (beta_power + 0.0001)
        theta_alpha_ratio = theta_power / (alpha_power + 0.0001)
        
        # 3. calculate basic statistical features (mean, std, skewness, kurtosis)
        # reference: aci et al. (2019) proved that time-domain statistical properties 
        # (like kurtosis, which measures sudden extreme spikes in the brainwave) 
        # provide clues for svm attention classification alongside frequencies.
        mean_value = np.mean(channel_data)
        std_deviation = np.std(channel_data)
        skewness = stats.skew(channel_data)
        kurtosis = stats.kurtosis(channel_data)
        
        # add all 12 features for this channel to my list
        features_list.append(delta_power)
        features_list.append(theta_power)
        features_list.append(alpha_power)
        features_list.append(beta_power)
        features_list.append(gamma_power)
        features_list.append(alpha_beta_ratio)
        features_list.append(theta_beta_ratio)
        features_list.append(theta_alpha_ratio)
        features_list.append(mean_value)
        features_list.append(std_deviation)
        features_list.append(skewness)
        features_list.append(kurtosis)
        
    # convert my list of features into a numpy array
    return np.array(features_list)


def process_all_windows(windows_array, sampling_rate):
    """
    loops through entire array of 1-second chunks (windows) one by one.
    it calls extract_features_for_one_window on each chunk and stacks the 
    resulting features into a massive 2d table (windows x features).
    """
    num_windows = windows_array.shape[0]
    
    # do the first window just to see how many features i get back
    first_window_features = extract_features_for_one_window(windows_array[0], sampling_rate)
    num_features = len(first_window_features)
    
    # create an empty table to hold all my features
    all_features = np.zeros((num_windows, num_features))
    all_features[0] = first_window_features
    
    # loop through the rest
    for i in range(1, num_windows):
        all_features[i] = extract_features_for_one_window(windows_array[i], sampling_rate)
        
    return all_features


# --- feature selection methods ---

def select_best_features(train_features, train_labels, test_features, method_name):
    """
    applies a feature selection method (like anova or random forest) to reduce the 
    number of features. it identifies the most predictive features in the training 
    data and throws away the rest, keeping only the best ones for both train and test sets.
    this helps the models train faster and ignore noisy data.
    
    reference (paper 3): the 'fi' (random forest feature importance) methodology 
    is derived from wang & kim (2024) "knn with feature importance for brain attention detection".
    """
    if method_name == 'none':
        # don't do anything, just keep all features
        return train_features, test_features
        
    elif method_name == 'anova':
        # anova tries to find features that are statistically different between classes
        # it basically runs ANOVA for 2 groups (focused and unfocused) for all features and gives each feature an f-score
        # and a p-value. this is equivalent to running t-tests on all the features.
        f_scores, p_values = f_classif(train_features, train_labels)
        
        # i only keep features where the p-value is less than or equal to 0.05
        # (meaning there's less than a 5% chance the difference is random)
        # this is basically an updated version of the method from paper 2
        features_to_keep = p_values <= 0.05
        
        # just in case anova throws away everything, save the top 10 as a backup
        if sum(features_to_keep) == 0:
            print("  ANOVA kept 0 features! Falling back to the top 10.")
            best_10_indexes = np.argsort(p_values)[0:10]
            features_to_keep = np.zeros(len(p_values), dtype=bool)
            for index in best_10_indexes:
                features_to_keep[index] = True
                
        return train_features[:, features_to_keep], test_features[:, features_to_keep]
        
    elif method_name == 'fi':
        # feature importance uses a random forest to see which features it relies on most
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(train_features, train_labels)
        
        importances = model.feature_importances_
        average_importance = np.mean(importances)
        
        # keep features that are more important than average
        features_to_keep = importances >= average_importance
        
        if sum(features_to_keep) == 0:
            features_to_keep = importances > 0
            
        return train_features[:, features_to_keep], test_features[:, features_to_keep]
        
    elif method_name == 'lcc':
        # linear correlation checks how strongly each feature relates to the label
        num_features = train_features.shape[1]
        correlations = np.zeros(num_features)
        
        for i in range(num_features):
            # calculate correlation and take the absolute value (i care about strength, not direction)
            corr = np.corrcoef(train_features[:, i], train_labels)[0, 1]
            correlations[i] = abs(corr)
            
        # fix any errors if a feature was completely flat (nan)
        correlations = np.nan_to_num(correlations)
        average_correlation = np.mean(correlations)
        
        features_to_keep = correlations >= average_correlation
        
        if sum(features_to_keep) == 0:
            best_10_indexes = np.argsort(correlations)[-10:]
            features_to_keep = np.zeros(len(correlations), dtype=bool)
            for index in best_10_indexes:
                features_to_keep[index] = True
                
        return train_features[:, features_to_keep], test_features[:, features_to_keep]
        
    elif method_name == 'pca':
        # pca (principal component analysis) squishes the features together to save space
        # it needs the data to be scaled first (mean=0, std=1)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_features)
        scaled_test = scaler.transform(test_features)
        
        # i tell it to keep 95% of the original variance (information)
        pca = PCA(n_components=0.95, random_state=42)
        pca_train = pca.fit_transform(scaled_train)
        pca_test = pca.transform(scaled_test)
        
        return pca_train, pca_test
        
    else:
        print(f"Error: Unknown method {method_name}")
        return train_features, test_features


# --- main process ---

def main():
    print("Starting feature extraction")
    
    methods_to_run = ['none', 'anova', 'fi', 'lcc', 'pca']
    
    # find all the files i created in step 1
    all_files = os.listdir(INPUT_FOLDER)
    subject_files = []
    for file in all_files:
        if file.endswith('.npz') and '_features' not in file:
            subject_files.append(file)
            
    start_time = time.time()
    
    for file in subject_files:
        subject_name = file.replace('.npz', '')
        print(f"\nProcessing {subject_name}...")
        
        # load the data (stored as numpy arrays in the previous step)
        data = np.load(os.path.join(INPUT_FOLDER, file))
        X_train_raw = data['X_train']
        y_train = data['y_train']
        X_test_raw = data['X_test']
        y_test = data['y_test']
        sampling_rate = int(data['fs'])
        
        # 1. calculate the base 84 features
        print("  Calculating 84 base features...")
        train_features = process_all_windows(X_train_raw, sampling_rate)
        test_features = process_all_windows(X_test_raw, sampling_rate)
        
        # 2. run feature selection and save everything
        # store all different feature versions in a dictionary
        data_to_save = {
            'y_train': y_train,
            'y_test': y_test
        }
        
        for method in methods_to_run:
            print(f"  Running feature selection: {method}")
            selected_train, selected_test = select_best_features(train_features, y_train, test_features, method)
            
            # save these selected features into the dictionary
            data_to_save[f'X_train_{method}'] = selected_train
            data_to_save[f'X_test_{method}'] = selected_test
            
        # save the dictionary to a new file
        output_file = os.path.join(OUTPUT_FOLDER, f"{subject_name}_features.npz")
        #since array names are dynamic its easier to throw it all into dicitonary and unpack it
        np.savez_compressed(output_file, **data_to_save)
        
    end_time = time.time()
    print(f"\nFinished extracting features in {end_time - start_time:.1f} seconds!")

if __name__ == '__main__':
    main()
