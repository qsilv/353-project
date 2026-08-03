"""
model_training.py - Step 3: Model Training and Evaluation

Trains 5 classical machine learning models on the extracted EEG features:
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Decision Tree (DT)
- Gradient Boosting (GBoosting)

Evaluates each model across all 5 feature selection methods 
(None, ANOVA, FI, LCC, PCA) per subject using the held-out round 4 test set.
Saves evaluation metrics to CSV.

Usage: python model_training.py
"""

import os
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # ignore undefined metric warnings

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = 'processed_data'
RESULTS_DIR = os.path.join('results', 'metrics')

# Feature selection methods to test
METHODS = ['none', 'anova', 'fi', 'lcc', 'pca']

# Define models
MODELS = {
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'RF': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42, n_jobs=-1),
    'DT': DecisionTreeClassifier(criterion='entropy', random_state=42),
    'GBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}


# ============================================================
# Helper Functions
# ============================================================

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    # Using weighted average in case of class imbalance
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, prec, rec, f1


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name, model):
    """Train a model and evaluate it on the test set."""
    # Pipeline with StandardScaler (crucial for SVM, KNN, PCA)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Train
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Evaluate
    acc, prec, rec, f1 = evaluate_model(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, prec, rec, f1, train_time, cm


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("EEG Model Training & Evaluation")
    print("=" * 60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    feature_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('_features.npz')])
    print(f"Found {len(feature_files)} feature files.")
    
    # Store all results
    all_results = []
    
    # Store confusion matrices for the best model (to visualize later)
    # We will pick a representative baseline: KNN + FI (Paper 3)
    best_model_cms = {} 
    
    total_start = time.time()
    
    for i, feat_file in enumerate(feature_files):
        subj_name = feat_file.replace('_features.npz', '')
        print(f"\n[{i+1}/{len(feature_files)}] Training models for {subj_name}...")
        
        # Load features
        data = np.load(os.path.join(INPUT_DIR, feat_file))
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Determine baseline accuracy (always predict majority class)
        majority_class = stats.mode(y_train, keepdims=True)[0][0] if hasattr(stats, 'mode') else np.argmax(np.bincount(y_train))
        baseline_acc = np.mean(y_test == majority_class)
        
        for method in METHODS:
            X_train = data[f'X_train_{method}']
            X_test = data[f'X_test_{method}']
            
            for model_name, model in MODELS.items():
                # Train & Evaluate
                acc, prec, rec, f1, t_time, cm = train_and_evaluate(
                    X_train, y_train, X_test, y_test, model_name, model
                )
                
                # Save result
                all_results.append({
                    'Subject': subj_name,
                    'Method': method,
                    'Model': model_name,
                    'Features': X_train.shape[1],
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'TrainTime_s': t_time,
                    'BaselineAcc': baseline_acc
                })
                
                # Save CM for KNN + FI (Paper 3 reproduction)
                if model_name == 'KNN' and method == 'fi':
                    best_model_cms[subj_name] = cm
    
    # Process results
    results_df = pd.DataFrame(all_results)
    
    # 1. Full detailed results
    csv_path = os.path.join(RESULTS_DIR, 'all_experiments_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to {csv_path}")
    
    # 2. Aggregated results across subjects
    agg_df = results_df.groupby(['Model', 'Method']).agg(
        Mean_Accuracy=('Accuracy', 'mean'),
        Std_Accuracy=('Accuracy', 'std'),
        Mean_F1=('F1', 'mean'),
        Mean_TrainTime=('TrainTime_s', 'mean'),
        Mean_Features=('Features', 'mean')
    ).reset_index()
    
    agg_df = agg_df.sort_values(by='Mean_Accuracy', ascending=False)
    agg_path = os.path.join(RESULTS_DIR, 'aggregated_results.csv')
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated results to {agg_path}")
    
    # 3. Save confusion matrices for visualization
    cm_path = os.path.join(RESULTS_DIR, 'knn_fi_confusion_matrices.npz')
    np.savez_compressed(cm_path, **best_model_cms)
    
    print(f"\n{'=' * 60}")
    print(f"Model training complete in {time.time()-total_start:.1f}s")
    
    print("\nTop 5 Model Configurations:")
    print(agg_df.head(5).to_string(index=False))
    print(f"{'=' * 60}")

if __name__ == '__main__':
    from scipy import stats  # Import here since it's used for baseline acc
    main()
