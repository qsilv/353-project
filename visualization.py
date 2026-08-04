"""
visualization.py - step 4: generate report visualizations

this script takes the results from our model training and draws
easy-to-understand charts for the final report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal

# --- settings ---
RESULTS_FOLDER = 'results/metrics'
FIGURES_FOLDER = 'results/figures'
PROCESSED_FOLDER = 'processed_data'

# make the charts look nice
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


# --- plotting functions ---

def draw_accuracy_chart():
    """draws a bar chart showing which model was the most accurate."""
    csv_path = os.path.join(RESULTS_FOLDER, 'aggregated_results.csv')
    if not os.path.exists(csv_path):
        print("Couldn't find the results CSV. Did you run Step 3?")
        return
        
    results_df = pd.read_csv(csv_path)
    
    # we use a tool called 'catplot' to draw two charts side-by-side
    # one for subject-dependent, one for cross-subject
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x='Model', 
        y='Mean_Accuracy', 
        hue='Method',
        col='Scenario',
        palette='viridis',
        height=6,
        aspect=1.2
    )
    
    chart.fig.suptitle('Model Accuracy Comparison', fontsize=16, y=1.05)
    chart.set_axis_labels("Machine Learning Model", "Average Accuracy")
    
    # make the y-axis go from 0 to 100% (0.0 to 1.0)
    for axis in chart.axes.flat:
        axis.set_ylim(0, 1.05)
        # draw a red line at 50% to show what random guessing looks like
        axis.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Guessing (50%)')
        
    plt.tight_layout()
    output_path = os.path.join(FIGURES_FOLDER, 'accuracy_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved accuracy chart!")


def draw_confusion_matrices():
    """draws a heatmap showing where our models made mistakes."""
    matrix_path = os.path.join(RESULTS_FOLDER, 'confusion_matrices.npz')
    if not os.path.exists(matrix_path):
        return
        
    data = np.load(matrix_path)
    
    # add up all the matrices for all subjects
    total_dependent = None
    total_cross = None
    
    for key in data.files:
        if '_dep' in key:
            if total_dependent is None:
                total_dependent = data[key]
            else:
                total_dependent = total_dependent + data[key]
        elif '_cross' in key:
            if total_cross is None:
                total_cross = data[key]
            else:
                total_cross = total_cross + data[key]
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if total_dependent is not None:
        # convert raw numbers to percentages
        row_totals = total_dependent.sum(axis=1)
        percentages = total_dependent / row_totals[:, None]
        
        sns.heatmap(
            percentages, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=['Unfocused', 'Focused'], yticklabels=['Unfocused', 'Focused'],
            ax=ax1
        )
        ax1.set_title('Subject-Dependent (KNN + Feature Importance)', pad=15)
        ax1.set_ylabel('What the person was actually doing')
        ax1.set_xlabel('What the AI guessed')
        
    if total_cross is not None:
        row_totals = total_cross.sum(axis=1)
        percentages = total_cross / row_totals[:, None]
        
        sns.heatmap(
            percentages, annot=True, fmt='.1%', cmap='Oranges',
            xticklabels=['Unfocused', 'Focused'], yticklabels=['Unfocused', 'Focused'],
            ax=ax2
        )
        ax2.set_title('Cross-Subject (SVM + PCA)', pad=15)
        ax2.set_ylabel('What the person was actually doing')
        ax2.set_xlabel('What the AI guessed')
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_FOLDER, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved confusion matrix!")


def draw_brainwaves():
    """draws a picture of the raw brainwaves and their frequencies."""
    # find the first subject's file just as an example
    all_files = os.listdir(PROCESSED_FOLDER)
    subject_files = [f for f in all_files if f.endswith('.npz') and not 'features' in f]
    
    if len(subject_files) == 0:
        return
        
    example_file = subject_files[0]
    data = np.load(os.path.join(PROCESSED_FOLDER, example_file))
    
    windows = data['X_train']
    labels = data['y_train']
    sampling_rate = int(data['fs'])
    
    # we'll just look at the 'fz' channel (which is index 1 in our list of 7)
    channel_index = 1 
    
    # find one focused window and one unfocused window
    focused_window_index = -1
    unfocused_window_index = -1
    
    for i in range(len(labels)):
        if labels[i] == 1 and focused_window_index == -1:
            focused_window_index = i
        if labels[i] == 0 and unfocused_window_index == -1:
            unfocused_window_index = i
            
    focused_wave = windows[focused_window_index, :, channel_index]
    unfocused_wave = windows[unfocused_window_index, :, channel_index]
    
    time_axis = np.arange(len(focused_wave)) / sampling_rate
    
    # --- plot 1: draw the raw squiggly lines ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    
    ax1.plot(time_axis, unfocused_wave, color='blue')
    ax1.set_title('Raw Brainwaves (Unfocused) - Channel Fz')
    ax1.set_ylabel('Voltage')
    
    ax2.plot(time_axis, focused_wave, color='orange')
    ax2.set_title('Raw Brainwaves (Focused) - Channel Fz')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Voltage')
    
    plt.tight_layout()
    output_path1 = os.path.join(FIGURES_FOLDER, 'raw_brainwaves.png')
    plt.savefig(output_path1, dpi=300)
    plt.close()
    print("Saved raw brainwaves plot!")


# --- main process ---

def main():
    print("Drawing pictures for the report...")
    
    if not os.path.exists(FIGURES_FOLDER):
        os.makedirs(FIGURES_FOLDER)
        
    draw_accuracy_chart()
    draw_confusion_matrices()
    draw_brainwaves()
    
    print("All done!")

if __name__ == '__main__':
    main()
