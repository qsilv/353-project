# EEG Mental Attention State Classification

Investigating classical machine learning techniques for classifying human mental attention states (Focused vs. Unfocused) from EEG data, targeting real-time Brain-Computer Interface (BCI) applications using the NeuroPawn 125 Hz headset.

## Dataset

**XJTU-EEG MEMA** (Multi-label EEG for Mental Attention states)
- Source: https://github.com/XJTU-EEG/MEMA
- 20 subjects, 12 trials each, 1,060 minutes total
- Raw data from `For_graph/` directory (pre-segmented `.txt` files)
- 7 channels selected: F3, Fz, F4, C3, Cz, C4, Pz

## How to Run

Scripts should be run **in order** from the project root directory:

```bash
# Step 1: Process raw .txt files -> cleaned .npz
python data_processing.py

# Step 2: Extract features from processed EEG data
python feature_extraction.py

# Step 3: Train and evaluate ML models
python model_training.py

# Step 4: Generate visualizations for the report
python visualization.py
```

## Required Libraries

All are standard Anaconda packages:
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

Install with: `pip install -r requirements.txt`

## Project Structure

```
353 project/
data_processing.py       # Raw EEG → cleaned, downsampled windows
feature_extraction.py    # Windows → feature matrices
model_training.py        # Train SVM, KNN, RF, DT, GBoosting
visualization.py         # Generate report figures
results/
   figures/             # Output plots
   metrics/             # CSV result tables
processed_data/          # Intermediate .npz files (generated)
MEMA Dataset/            # Raw data (not included, see above)
Papers/                  # Reference papers
requirements.txt
README.md
```

## Methods

- **Preprocessing**: Bandpass filter (0.5-45 Hz), downsample 500→125 Hz, 1s windows
- **Features**: Band powers (delta/theta/alpha/beta/gamma), band ratios, statistical features (84 total)
- **Feature Selection**: ANOVA, Feature Importance, Linear Correlation, PCA
- **Models**: SVM, KNN, Random Forest, Decision Tree, Gradient Boosting
- **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, 5-fold CV

## References

1. Aci et al. (2019) - SVM-based EEG attention classification
2. Wang & Kim (2024) - KNN with feature importance for brain attention detection
3. Liu et al. (2025) - MEMA dataset paper
