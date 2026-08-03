"""
visualization.py - Step 4: Generate Report Visualizations

Generates all figures needed for the project report:
1. Model accuracy comparison across feature selection methods
2. Confusion matrix heatmap for the best model
3. Raw EEG trace examples (Focused vs Unfocused)
4. Power Spectral Density (PSD) comparison

Usage: python visualization.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = os.path.join('results', 'metrics')
FIGURES_DIR = os.path.join('results', 'figures')
PROCESSED_DIR = 'processed_data'

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
COLORS = sns.color_palette("husl", 8)


# ============================================================
# Plotting Functions
# ============================================================

def plot_model_comparison():
    """Plot average accuracy across all models and feature methods."""
    csv_path = os.path.join(RESULTS_DIR, 'aggregated_results.csv')
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Run model_training.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    ax = sns.barplot(
        data=df, 
        x='Model', 
        y='Mean_Accuracy', 
        hue='Method',
        palette='viridis'
    )
    
    plt.title('Model Accuracy by Feature Selection Method', fontsize=14, pad=15)
    plt.ylabel('Mean Accuracy (across subjects)', fontsize=12)
    plt.xlabel('Machine Learning Model', fontsize=12)
    plt.ylim(0, 1.05) # Cap at 1.0, give space for legend
    
    # Move legend outside
    plt.legend(title='Feature Selection', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add baseline accuracy line (assume ~0.5 for balanced, or calculate mean from data)
    # We will just add a grid line at 0.5 as visual baseline
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance (0.5)')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_confusion_matrix():
    """Plot confusion matrix heatmap for the chosen baseline (KNN+FI)."""
    cm_path = os.path.join(RESULTS_DIR, 'knn_fi_confusion_matrices.npz')
    if not os.path.exists(cm_path):
        print(f"Cannot find {cm_path}. Run model_training.py first.")
        return
        
    data = np.load(cm_path)
    # Sum CMs across all subjects to get overall performance
    total_cm = None
    for subj in data.files:
        if total_cm is None:
            total_cm = data[subj]
        else:
            total_cm += data[subj]
            
    # Normalize by row (true label) to get percentages
    cm_norm = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.1%', 
        cmap='Blues',
        xticklabels=['Unfocused', 'Focused'],
        yticklabels=['Unfocused', 'Focused'],
        cbar_kws={'label': 'Percentage of True Class'}
    )
    
    plt.title('Confusion Matrix (KNN + Feature Importance)', pad=15)
    plt.ylabel('True Attention State')
    plt.xlabel('Predicted Attention State')
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_eeg_signals_and_psd():
    """Plot raw EEG traces and PSD for one subject to show differences."""
    # Find the first subject file
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npz') and not '_features' in f]
    if not files:
        print(f"No processed data found in {PROCESSED_DIR}")
        return
        
    subj_file = files[0]
    data = np.load(os.path.join(PROCESSED_DIR, subj_file))
    
    X = data['X_train']
    y = data['y_train']
    fs = int(data['fs'])
    ch_names = data['channel_names']
    
    # We'll plot Fz channel (index 1 if 7 channels: F3, Fz, F4, C3, Cz, C4, Pz)
    fz_idx = 1
    
    # Find a focused window (y=1) and unfocused window (y=0)
    idx_foc = np.where(y == 1)[0][0]
    idx_unf = np.where(y == 0)[0][0]
    
    win_foc = X[idx_foc, :, fz_idx]
    win_unf = X[idx_unf, :, fz_idx]
    
    t = np.arange(len(win_foc)) / fs
    
    # --- Plot 1: Raw Traces ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    
    ax1.plot(t, win_unf, color='tab:blue')
    ax1.set_title(f'Raw EEG Trace (Unfocused) - Channel {ch_names[fz_idx]}')
    ax1.set_ylabel('Amplitude')
    
    ax2.plot(t, win_foc, color='tab:orange')
    ax2.set_title(f'Raw EEG Trace (Focused) - Channel {ch_names[fz_idx]}')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    
    plt.tight_layout()
    out_path1 = os.path.join(FIGURES_DIR, 'raw_traces.png')
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved {out_path1}")
    
    # --- Plot 2: PSD Comparison (average over all windows) ---
    f_foc, psd_foc = signal.welch(X[y==1, :, fz_idx].flatten(), fs, nperseg=fs)
    f_unf, psd_unf = signal.welch(X[y==0, :, fz_idx].flatten(), fs, nperseg=fs)
    
    plt.figure(figsize=(8, 5))
    # Plot only up to 45 Hz (our filter cutoff)
    valid_idx = f_foc <= 45
    
    plt.plot(f_unf[valid_idx], psd_unf[valid_idx], color='tab:blue', label='Unfocused', linewidth=2)
    plt.plot(f_foc[valid_idx], psd_foc[valid_idx], color='tab:orange', label='Focused', linewidth=2)
    
    # Highlight bands
    plt.axvspan(8, 13, color='yellow', alpha=0.2, label='Alpha (8-13 Hz)')
    plt.axvspan(13, 30, color='red', alpha=0.1, label='Beta (13-30 Hz)')
    
    plt.title(f'Power Spectral Density Comparison - Channel {ch_names[fz_idx]}', pad=15)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    out_path2 = os.path.join(FIGURES_DIR, 'psd_comparison.png')
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved {out_path2}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Generating Report Visualizations")
    print("=" * 60)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    plot_model_comparison()
    plot_confusion_matrix()
    plot_eeg_signals_and_psd()
    
    print(f"\n{'=' * 60}")
    print("All visualizations generated.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
