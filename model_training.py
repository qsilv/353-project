"""
model_training.py - step 3: model training and evaluation

this script trains 5 different machine learning models to guess 
if the person was focused or unfocused based on their brainwaves.

i test two scenarios:
1. subject-dependent: i train on a person's first 3 rounds and test on their 4th.
2. cross-subject: i train on 19 people and test on the 1 left out.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- settings ---
INPUT_FOLDER = 'processed_data'
OUTPUT_FOLDER = 'results/metrics'

FEATURE_METHODS = ['none', 'anova', 'fi', 'lcc', 'pca']

def get_models():
    """dictionary of models so i can train from scratch."""
    return {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'RF': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, n_jobs=-1),
        'DT': DecisionTreeClassifier(criterion='entropy', random_state=42),
        'GBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

# --- main process ---

def main():
    print("Starting model training")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    all_files = os.listdir(INPUT_FOLDER)
    feature_files = [f for f in all_files if f.endswith('_features.npz')]
    
    print(f"Loading data for {len(feature_files)} subjects")
    
    # 1. load all data into a big dictionary
    all_subjects_data = {}
    for file in feature_files:
        #gets rid of file tail to just get the actual name
        subject_name = file.replace('_features.npz', '')
        data = np.load(os.path.join(INPUT_FOLDER, file))
        
        # save the labels
        subject_data = {
            'y_train': data['y_train'],
            'y_test': data['y_test'],
            # combine train and test labels for cross-subject testing
            'y_combined': np.concatenate([data['y_train'], data['y_test']])
        }
        
        # save the features for each method
        for method in FEATURE_METHODS:
            subject_data[f'X_train_{method}'] = data[f'X_train_{method}']
            subject_data[f'X_test_{method}'] = data[f'X_test_{method}']
            subject_data[f'X_combined_{method}'] = np.concatenate([data[f'X_train_{method}'], data[f'X_test_{method}']])
            
        all_subjects_data[subject_name] = subject_data
        
    print("Data loaded")
    
    # list to hold all my results so i can save it to a csv later
    results_list = []
    
    # dictionary to save the confusion matrices (so i can draw pictures of them later)
    saved_matrices = {}
    
    start_time = time.time()
    subject_names = list(all_subjects_data.keys())
    
    # 2. run evaluations
    for i in range(len(subject_names)):
        test_subject = subject_names[i]
        print(f"Evaluating models on test subject: {test_subject} ({i+1}/{len(subject_names)})")
        
        subject_data = all_subjects_data[test_subject]
        
        # calculate baseline accuracy (if i just guessed the most common label every time)
        # mode returns object containing 2 arrays, most common label and how many times, i need first array
        most_common_train = stats.mode(subject_data['y_train'], keepdims=True)[0][0]
        # finds mean of true and false (since they are stored as 0 and 1) to produce baseline
        baseline_acc_dependent = np.mean(subject_data['y_test'] == most_common_train)
        
        # set up the cross-subject labels (train on everyone else, test on this subject)
        y_test_cross = subject_data['y_combined']
        y_train_cross_list = []
        for other_subj in subject_names:
            if other_subj != test_subject:
                y_train_cross_list.append(all_subjects_data[other_subj]['y_combined'])
        y_train_cross = np.concatenate(y_train_cross_list)
        
        most_common_cross = stats.mode(y_train_cross, keepdims=True)[0][0]
        baseline_acc_cross = np.mean(y_test_cross == most_common_cross)
        
        # loop through each feature selection method
        for method in FEATURE_METHODS:
            print(f"  -> {method.upper()} features:")
            
            # prepare subject-dependent data
            X_train_dep = subject_data[f'X_train_{method}']
            X_test_dep = subject_data[f'X_test_{method}']
            y_train_dep = subject_data['y_train']
            y_test_dep = subject_data['y_test']
            
            # prepare cross-subject data
            X_test_cross = subject_data[f'X_combined_{method}']
            X_train_cross_list = []
            for other_subj in subject_names:
                if other_subj != test_subject:
                    X_train_cross_list.append(all_subjects_data[other_subj][f'X_combined_{method}'])
            X_train_cross = np.concatenate(X_train_cross_list)
            
            # loop through each machine learning model
            models = get_models()
            for model_name, model in models.items():
                print(f"       Training {model_name}...", end='', flush=True)
                
                # --- a. subject-dependent evaluation ---
                
                # step 1 & 2: scale the data and train the model using a pipeline
                clf_dep = make_pipeline(StandardScaler(), model)
                
                t0 = time.time()
                clf_dep.fit(X_train_dep, y_train_dep)
                train_time_dep = time.time() - t0
                
                # step 3: make guesses
                predictions_dep = clf_dep.predict(X_test_dep)
                
                # step 4: check how well it did
                acc_dep = accuracy_score(y_test_dep, predictions_dep)
                f1_dep = f1_score(y_test_dep, predictions_dep, average='weighted', zero_division=0)
                
                results_list.append({
                    'Scenario': 'Subject-Dependent',
                    'Subject': test_subject,
                    'Method': method,
                    'Model': model_name,
                    'Features': X_train_dep.shape[1],
                    'Accuracy': acc_dep,
                    'F1': f1_dep,
                    'TrainTime_s': train_time_dep,
                    'BaselineAcc': baseline_acc_dependent
                })
                
                # save matrix for visualization later
                if model_name == 'KNN' and method == 'fi':
                    matrix = confusion_matrix(y_test_dep, predictions_dep)
                    saved_matrices[f"{test_subject}_dep"] = matrix
                
                
                # --- b. cross-subject evaluation ---
                # i have to use a fresh copy of the model for cross-subject!
                fresh_model = get_models()[model_name]
                
                # step 1 & 2: scale and train
                clf_cross = make_pipeline(StandardScaler(), fresh_model)
                
                t0 = time.time()
                clf_cross.fit(X_train_cross, y_train_cross)
                train_time_cross = time.time() - t0
                
                # step 3: predict
                predictions_cross = clf_cross.predict(X_test_cross)
                
                # step 4: evaluate
                acc_cross = accuracy_score(y_test_cross, predictions_cross)
                f1_cross = f1_score(y_test_cross, predictions_cross, average='weighted', zero_division=0)
                
                results_list.append({
                    'Scenario': 'Cross-Subject',
                    'Subject': test_subject,
                    'Method': method,
                    'Model': model_name,
                    'Features': X_train_cross.shape[1],
                    'Accuracy': acc_cross,
                    'F1': f1_cross,
                    'TrainTime_s': train_time_cross,
                    'BaselineAcc': baseline_acc_cross
                })
                
                # save matrix for visualization later
                if model_name == 'SVM' and method == 'pca':
                    matrix = confusion_matrix(y_test_cross, predictions_cross)
                    saved_matrices[f"{test_subject}_cross"] = matrix
                    
                print(" done")

    # 3. save all results
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(OUTPUT_FOLDER, 'all_experiments_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # calculate average accuracy across all subjects
    summary_df = results_df.groupby(['Scenario', 'Model', 'Method']).agg(
        Mean_Accuracy=('Accuracy', 'mean'),
        Mean_F1=('F1', 'mean'),
        Mean_TrainTime=('TrainTime_s', 'mean')
    ).reset_index()
    
    # sort so the best models are at the top
    summary_df = summary_df.sort_values(by=['Scenario', 'Mean_Accuracy'], ascending=[True, False])
    summary_path = os.path.join(OUTPUT_FOLDER, 'aggregated_results.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # save the confusion matrices
    matrix_path = os.path.join(OUTPUT_FOLDER, 'confusion_matrices.npz')
    np.savez_compressed(matrix_path, **saved_matrices)
    
    end_time = time.time()
    print(f"\nFinished training all models in {end_time - start_time:.1f} seconds!")

if __name__ == '__main__':
    main()
