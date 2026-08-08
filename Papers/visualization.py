"""
visualization.py - step 4: generate report visualizations

this script takes the results from my model training and draws
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
sns.set_context("paper", font_scale=1.4)


# --- plotting functions ---

def draw_accuracy_chart():
    """draws a bar chart showing which model was the most accurate."""
    csv_path = os.path.join(RESULTS_FOLDER, 'aggregated_results.csv')
    if not os.path.exists(csv_path):
        print("Couldn't find the results CSV. Did you run Step 3?")
        return
        
    results_df = pd.read_csv(csv_path)
    
    # i use a tool called 'catplot' to draw two charts side-by-side
    # one for subject-dependent, one for cross-subject
    chart = sns.catplot(
        data=results_df, 
        kind="bar",
        x='Model', 
        y='Mean_Accuracy', 
        hue='Method',
        col='Scenario',
        palette='viridis',
        height=7,
        aspect=1.4
    )
    
    chart.fig.suptitle('Model Accuracy Comparison', fontsize=18, y=1.02)
    chart.set_axis_labels("Machine Learning Model", "Average Accuracy")
    
    # make the y-axis start at 0.4 so the differences are visible
    for axis in chart.axes.flat:
        axis.set_ylim(0.4, 0.7)
        # draw a red line at 50% to show what random guessing looks like
        axis.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        axis.tick_params(axis='x', labelsize=11)
    
    # make the legend text bigger
    chart._legend.set_title('Feature Method')
    plt.setp(chart._legend.get_texts(), fontsize=11)
    plt.setp(chart._legend.get_title(), fontsize=12)
        
    output_path = os.path.join(FIGURES_FOLDER, 'accuracy_chart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved accuracy chart!")


def draw_confusion_matrices():
    """draws a heatmap showing where my models made mistakes."""
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
        ax1.set_title('Subject-Dependent (GBoosting + All Features)', pad=15)
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
        ax2.set_title('Cross-Subject (RF + LCC)', pad=15)
        ax2.set_ylabel('What the person was actually doing')
        ax2.set_xlabel('What the AI guessed')
    
    plt.tight_layout()
    output_path = os.path.join(FIGURES_FOLDER, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved confusion matrix!")



# --- main process ---

def main():
    print("Drawing pictures for the report...")
    
    if not os.path.exists(FIGURES_FOLDER):
        os.makedirs(FIGURES_FOLDER)
        
    draw_accuracy_chart()
    draw_confusion_matrices()
    
    print("All done!")

if __name__ == '__main__':
    main()
