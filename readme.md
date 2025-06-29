# Brain Signal Processing and Classification Framework

This project provides a comprehensive pipeline for processing and analyzing brain signals (LFP data), feature extraction, and machine learning classification. It implements the methods used in our team's research paper:

> S Liu, Y Qi, S Hu, N Wei, J Zhang, J Zhu, H Wu, H Hu, Y Yang*, Y Wang*. A Habenula Neural Biomarker Simultaneously Tracks Weekly and Daily Symptom Variations during Deep Brain Stimulation Therapy for Depression[J]. IEEE Journal of Biomedical and Health Informatics, 2025, doi: 10.1109/JBHI.2025.3566057.

## Project Structure

```
project/
├── dataset/                     # Data storage
│   ├── feature/                 # Extracted features
│   ├── preprocessed_data/       # Preprocessed signal data
│   └── raw_data/                # Original raw data files
├── matlab_script/               # MATLAB processing scripts
│   ├── utils/                   # Utility functions
│   ├── M0_preprocess.m          # Data preprocessing
│   └── M1_feature_extraction.m  # Feature extraction
├── python_script/               # Python analysis scripts
│   ├── M2_classification_ml_model.py  # ML classification
│   └── M3_ml_model_interpret.py       # Model interpretation
└── result/                      # Results and outputs
  ├── plot_data/               # Visualization data
  └── plot_pdf/                # Visualization output
    └── png/                 # PNG format plots
```

## Workflow

The project follows a four-stage data processing and analysis pipeline:

### Stage 0: Environment Setup

**Prerequisites:**
- MATLAB (≥R2019b) with Signal Processing Toolbox
- Python environment manager (conda or pip)

**Installation Steps:**
1. Install MATLAB and verify version compatibility
2. Create Python virtual environment:
   ```bash
   conda create -n brain_signal_analysis python=3.8.20
   conda activate brain_signal_analysis
   pip install -r requirement.txt
   ```

### Stage 1: Data Preprocessing (`M0_preprocess.m`)

**Purpose:** Clean and filter raw LFP/EEG signals to remove noise and artifacts.

**Processing Steps:**
- Load raw data from `.mat` files
- Apply low-pass filter to remove high-frequency noise
- Apply high-pass filter to remove DC offset and low-frequency drift
- Apply notch filters to eliminate power line interference

**Note:** DC (Direct Current) refers to the constant voltage offset in the signal that doesn't carry useful neural information and can interfere with analysis.

**Key Parameters:**
- Low-pass cutoff: 25 Hz
- High-pass cutoff: 20 Hz
- Notch frequency: 50 Hz

**References:**
- [Low-pass filter](https://en.wikipedia.org/wiki/Low-pass_filter)
- [High-pass filter](https://en.wikipedia.org/wiki/High-pass_filter)
- [Notch filter](https://en.wikipedia.org/wiki/Band-stop_filter)
- [MATLAB designfilt](https://ww2.mathworks.cn/help/signal/ref/designfilt.html)

**Exercise 1: Filter Parameter Optimization**

*Objective:* Modify preprocessing parameters for optimal signal filtering.

*Current Settings:*
- Low-pass: 25 Hz
- High-pass: 20 Hz
- Notch: 50 Hz

*Required Settings:*
- Low-pass: 30 Hz
- High-pass: 1 Hz
- Notch: 40 Hz

*Implementation:*
1. Open `M0_preprocess.m` in MATLAB
2. Locate filter parameter definitions
3. Update cutoff frequencies as specified
4. Execute script and verify results
5. Review visualization: `result/plot_png/M0_preprocess_visualization_20220419.png`

*Expected Outcome:* Improved signal preservation with effective noise reduction.

### Stage 2: Feature Extraction (`M1_feature_extraction.m`)

**Purpose:** Extract Power Spectral Density (PSD) features from preprocessed signals.

**Processing Steps:**
- Calculate PSD using Welch's method
- Extract features for standard frequency bands:
  - Delta: 1-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-12 Hz
  - Beta: 12-30 Hz
- Use 10-second windows with 50% overlap and Hamming window
- Apply log transformation for power normalization
- Generate spectrum visualization and band feature plots

**References:**
- [Power Spectral Density](https://en.wikipedia.org/wiki/Spectral_density)
- [MATLAB pwelch](https://ww2.mathworks.cn/help/signal/ref/pwelch.html)
- [MATLAB pow2db](https://ww2.mathworks.cn/help/signal/ref/pow2db.html)

**Exercise 2: Feature Extraction Enhancement**

*Objective:* Improve frequency band analysis and power representation.

*Current Issues:*
- Power values plotted in linear scale instead of dB scale
- Potential frequency band definition mismatches
- Suboptimal PSD visualization

*Required Modifications:*
1. Convert power to decibel scale using `pow2db()` function
2. Verify and correct frequency band definitions:
   - Delta: 1-4 Hz
   - Theta: 4-8 Hz
   - Alpha: 8-12 Hz
   - Beta: 12-30 Hz

*Implementation:*
1. Open `M1_feature_extraction.m` in MATLAB
2. Replace linear power values with `pow2db(power_values)` for visualization
3. Update frequency band boundaries in feature extraction loop
4. Execute script and verify output: `result/plot_png/M1_feature_extraction_visualization_20220419.png`

*Expected Outcome:* Standardized dB-scale visualization and accurate frequency band representation.

### Stage 3: Machine Learning Classification (`M2_classification_ml_model.py`)

**Purpose:** Implement and evaluate classification models using extracted features.

**Processing Steps:**
- Load and standardize features using StandardScaler
- Implement Logistic Regression classification
- Apply stratified k-fold cross-validation (k=5)
- Calculate performance metrics:
  - **Accuracy:** (TP + TN) / (TP + TN + FP + FN)
  - **Sensitivity (Recall):** TP / (TP + FN)
  - **Specificity:** TN / (TN + FP)
  - **F1-score:** 2 × (Precision × Recall) / (Precision + Recall)
  - **AUC:** Area under ROC curve
- Extract model coefficients for interpretation

*Where: TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives*

**References:**
- [Classification Metrics](https://medium.com/analytics-vidhya/notes-on-sensitivity-specificity-precision-recall-and-f1-score-e34204d0bb9b)
- [Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [ROC Curve](https://www.evidentlyai.com/classification-metrics/explain-roc-curve)

**Exercise 3: Classification Parameter Correction**

*Objective:* Fix metric calculations and improve cross-validation robustness.

*Current Issues:*
- Incorrect sensitivity and specificity formulas
- Insufficient cross-validation folds (2 instead of 5)
- Unreliable performance estimates

*Required Modifications:*
1. Correct sensitivity and specificity calculation formulas
2. Update StratifiedKFold from 2 to 5 folds
3. Ensure proper confusion matrix indexing

*Implementation:*
1. Open `M2_classification_ml_model.py`
2. Fix metric calculation formulas in the evaluation section
3. Change StratifiedKFold parameter to 5 folds
4. Test with sample data to verify corrections
5. Execute script and review: `result/plot_png/M2_classification_performance_results.png`

*Expected Outcome:* Accurate performance metrics and robust cross-validation results.

### Stage 4: Model Interpretation (`M3_ml_model_interpret.py`)

**Purpose:** Analyze and visualize model decision-making through coefficient interpretation.

**Processing Steps:**
- Extract Logistic Regression coefficients
- Identify most important features
- Generate coefficient visualization plots

**References:**
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Coefficient Interpretation](https://medium.com/data-science/a-simple-interpretation-of-logistic-regression-coefficients-e3a40a62e8cf)

**Exercise 4: Coefficient Visualization Enhancement**

*Objective:* Display absolute coefficient values for better feature importance assessment.

*Current Issue:* Mixed positive and negative coefficient display obscures feature importance ranking.

*Required Modification:* Plot absolute coefficient values to emphasize magnitude-based importance.

*Implementation:*
1. Open `M3_ml_model_interpret.py`
2. Apply absolute value transformation: `np.abs(coefficients)` before plotting
3. Update plot labels and title to reflect absolute values
4. Execute script and verify: `result/plot_png/M3_interprete.png`

*Expected Outcome:* Clear feature importance visualization based on coefficient magnitude.

## Usage Instructions

1. **Data Preparation:** Place raw data files in `dataset/raw_data/`
2. **Preprocessing:** Execute `M0_preprocess.m` in MATLAB
3. **Feature Extraction:** Run `M1_feature_extraction.m` in MATLAB
4. **Classification:** Execute `M2_classification_ml_model.py` in Python
5. **Interpretation:** Run `M3_ml_model_interpret.py` in Python

## System Requirements

**MATLAB:**
- Version: R2019b or later
- Required Toolbox: Signal Processing Toolbox

**Python:**
- Python 3.8+
- Required packages:
  - scikit-learn
  - numpy
  - matplotlib
  - scipy
