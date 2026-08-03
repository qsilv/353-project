# EEG Mental Attention State Classification — Implementation Plan (Final)

## Background & Framing

**Client scenario**: A BCI startup developing the NeuroPawn (a consumer-grade 125 Hz, 7-channel EEG headset) wants to know which classical ML techniques work best for real-time binary attention classification (focused vs. unfocused). We use the research-grade MEMA dataset but **downsample to 125 Hz** and **restrict to 7 channels** to simulate NeuroPawn conditions.

---

## Answers to Your Questions

### For_graph vs. raw_data — What's the Difference?

| | `raw_data/` | `For_graph/` |
|---|---|---|
| **Files** | 2 files per subject: `Data1.txt` (gyroscope, 11 cols) and `Data2.txt` (continuous EEG, 34 cols) | 12 `.txt` files per subject — **one per trial**, already segmented by attention state |
| **Size** | ~800 MB per subject (Data2), ~50 min continuous recording | ~60–350s per trial file, ~500 MB total per subject |
| **Labeling** | No labels in the file — you'd have to align with trial timestamps manually | **Labels are implicit from filename**: `_a` = concentrating, `_n` = neutral, `_r` = relaxing |
| **Columns** | 34 columns (32 EEG + 2 extra) | 32 columns (30 EEG + 2 EOG) |
| **Format** | One continuous stream — includes breaks between trials, hints, rest periods | Clean segments — only task-relevant EEG data |

> [!TIP]
> **Recommendation**: Use `For_graph/` — it's already segmented by trial with labels encoded in filenames. This still gives us genuine raw data processing (parsing CSV, identifying channels, filtering, downsampling) without the nightmare of aligning timestamps in a 50-minute continuous stream.

### Spark for Speed? — Not Needed

Benchmarked reading speeds:
- **Full 32 columns**: pandas reads 173k lines in **0.6s**
- **7-column subset** (using `usecols`): 173k lines in **0.4s**
- **All 20 subjects** (estimated): ~27M total lines → **~1 minute** with `usecols`

pandas `read_csv` with `usecols=[2,3,4,12,13,14,23]` is plenty fast. Spark would be overkill and add unnecessary complexity to the project. The bottleneck isn't reading — it's feature extraction, which is also fast with numpy/scipy on 7-channel data.

---

## Confirmed Dataset Details

### Channel Mapping (Verified)

The 32 columns in the `.txt` files map to the ZhenTec-NT1-32 (10-10 system):

| Col | Channel | Col | Channel | Col | Channel | Col | Channel |
|---|---|---|---|---|---|---|---|
| 0 | FP1 | 8 | FCz | 16 | T8 | 24 | P4 |
| 1 | FP2 | 9 | FC4 | **17** | **CPz (REF)** | 25 | P7 |
| 2 | **F3** ✓ | 10 | FT7 | 18 | CP3 | 26 | P8 |
| 3 | **Fz** ✓ | 11 | FT8 | 19 | CP4 | 27 | O1 |
| 4 | **F4** ✓ | 12 | **C3** ✓ | 20 | TP7 | 28 | Oz |
| 5 | F7 | 13 | **Cz** ✓ | 21 | TP8 | 29 | O2 |
| 6 | F8 | 14 | **C4** ✓ | 22 | P3 | 30 | HEOG |
| 7 | FC3 | 15 | T7 | 23 | **Pz** ✓ | 31 | VEOG |

**7 channels used** (matching Papers 1 & 3): F3 (col 2), Fz (col 3), F4 (col 4), C3 (col 12), Cz (col 13), C4 (col 14), Pz (col 23)

Col 17 (CPz) = constant 187500 (reference electrode) — **dropped**.  
Cols 30–31 (HEOG, VEOG) = eye movement channels — **dropped**.

### Binary Labels

| Filename pattern | State | Binary label |
|---|---|---|
| `SubjectN_a{1-4}.txt` | Concentrating | **1** (focused) |
| `SubjectN_n{1-4}.txt` | Neutral | **0** (unfocused) |
| `SubjectN_r{1-4}.txt` | Relaxing | **0** (unfocused) |

### Train/Test Split

- **Train**: Rounds 1–3 (trials `_a1`, `_a2`, `_a3`, `_n1`, `_n2`, `_n3`, `_r1`, `_r2`, `_r3`)
- **Test**: Round 4 (trials `_a4`, `_n4`, `_r4`)
- Per-subject evaluation (subject-dependent)

---

## Proposed Project Structure

```
353 project/
├── data_processing.py          # [NEW] Step 1: Raw .txt → cleaned .npz
├── feature_extraction.py       # [NEW] Step 2: EEG windows → feature matrix
├── model_training.py           # [NEW] Step 3: Train & evaluate all models
├── visualization.py            # [NEW] Step 4: Generate report figures
├── results/
│   ├── figures/                # [NEW] Saved plots
│   └── metrics/                # [NEW] CSV result tables
├── requirements.txt            # [NEW]
├── README.md                   # [NEW]
├── MEMA Dataset/               # (existing)
└── Papers/                     # (existing)
```

---

## Script Details

### Script 1: `data_processing.py`

#### [NEW] [data_processing.py](file:///c:/Users/mfath/Desktop/353%20project/data_processing.py)

1. **Read raw `.txt` files** — `pd.read_csv(path, header=None, usecols=[2,3,4,12,13,14,23])` 
2. **Assign column names** — F3, Fz, F4, C3, Cz, C4, Pz
3. **Downsample 500 Hz → 125 Hz** — `scipy.signal.decimate(data, q=4, axis=0)` with built-in anti-aliasing
4. **Bandpass filter** — Butterworth 4th order, 0.5–45 Hz at 125 Hz
5. **Segment** — 1-second non-overlapping windows (125 samples each)
6. **Label** — from filename (`_a` → 1 focused, `_n`/`_r` → 0 unfocused)
7. **Split** — rounds 1–3 = train, round 4 = test
8. **Save** — one `.npz` per subject with `X_train`, `y_train`, `X_test`, `y_test`

Estimated runtime: **~2 minutes** for all 20 subjects.

---

### Script 2: `feature_extraction.py`

#### [NEW] [feature_extraction.py](file:///c:/Users/mfath/Desktop/353%20project/feature_extraction.py)

Per 1-second window, per channel:

| Feature | Count/channel | Method |
|---|---|---|
| Band powers (delta, theta, alpha, beta, gamma) | 5 | `scipy.signal.welch()` |
| Band ratios (alpha/beta, theta/beta, theta/alpha) | 3 | derived from band powers |
| Statistical (mean, std, skewness, kurtosis) | 4 | `scipy.stats` + numpy |
| **Total per channel** | **12** | |
| **Total (7 channels)** | **84** | |

Feature selection methods (from Paper 3):

| Method | Implementation |
|---|---|
| None (baseline) | All 84 features |
| ANOVA | `sklearn.feature_selection.f_classif`, keep p ≤ 0.05 |
| Feature Importance | `RandomForestClassifier.feature_importances_`, keep ≥ mean |
| LCC | `np.corrcoef`, keep |r| ≥ mean |
| PCA | `sklearn.decomposition.PCA(n_components='mle')` |

---

### Script 3: `model_training.py`

#### [NEW] [model_training.py](file:///c:/Users/mfath/Desktop/353%20project/model_training.py)

5 models (from Papers 1 & 3):

| Model | Hyperparameters |
|---|---|
| SVM | `kernel='rbf', C=1.0, probability=True` |
| KNN | `n_neighbors=5` |
| Random Forest | `n_estimators=100, criterion='entropy'` |
| Decision Tree | `criterion='entropy'` |
| Gradient Boosting | `n_estimators=100` |

**Experiment 1**: SVM on all features, per-subject → mean accuracy (Paper 1 reproduction)  
**Experiment 2**: KNN + Feature Importance selection, per-subject (Paper 3 reproduction)  
**Experiment 3**: All 5 models × all 5 feature methods → full comparison table  

All models wrapped in `sklearn.pipeline.Pipeline` with `StandardScaler`.  
Evaluation: accuracy, precision, recall, F1 (weighted), confusion matrix, 5-fold CV, training time.

---

### Script 4: `visualization.py`

#### [NEW] [visualization.py](file:///c:/Users/mfath/Desktop/353%20project/visualization.py)

| Figure | Purpose |
|---|---|
| Raw EEG traces (3 states) | Show what the data looks like |
| PSD comparison by state | Frequency-domain differences |
| Model accuracy bar chart | Main results figure |
| Feature selection impact | Grouped bars: model × method |
| Confusion matrix heatmap | Best model error analysis |
| Feature importance ranking | Top features driving classification |
| Class distribution | Show train/test balance |

---

## Verification Plan

```bash
python data_processing.py        # ~2 min
python feature_extraction.py     # ~1-2 min
python model_training.py         # ~5 min
python visualization.py          # ~30 sec
```

### Expected Results

For binary classification (focused vs. unfocused), subject-dependent, 7 channels at 125 Hz:
- Should achieve **85–95%+ accuracy** (binary is easier than 3-class, and the MEMA paper's 3-class baseline is already 78–85%)
- KNN + Feature Importance should be among the top performers (per Paper 3)
- Feature selection should reduce from 84 → ~30–50 features without hurting accuracy

### Subset for Development

Use **5 subjects** (Subject1–5) during development. Run all 20 for final results only.
