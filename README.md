# 💳 Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using Logistic Regression and XGBoost, with SHAP explainability and threshold tuning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

The dataset contains **284,807 transactions** made by European cardholders in September 2013. Only **492 (0.17%)** are fraudulent, making this a highly imbalanced classification problem.

**Key challenges:**
- Extreme class imbalance (99.83% normal vs 0.17% fraud)
- Anonymized features (PCA-transformed V1–V28)
- Business need: maximize fraud recall without too many false positives

## 🏗️ Project Pipeline

| Step | Description |
|------|-------------|
| 1. Data Loading | Load CSV, basic statistics |
| 2. EDA | Class distribution, Amount analysis |
| 3. Preprocessing | StandardScaler on Amount & Time, stratified split |
| 4. Logistic Regression | Baseline model with balanced class weights |
| 5. XGBoost | Gradient boosting with `scale_pos_weight` |
| 6. PR-AUC | Precision-Recall curves comparison |
| 7. SHAP | Feature importance & explainability |
| 8. Threshold Tuning | Optimal decision threshold via F1 maximization |
| 9. Summary | Model comparison table |

## 📊 Results

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|----|--------|---------|
| Logistic Regression | ~0.06 | ~0.92 | ~0.11 | ~0.75 | ~0.97 |
| XGBoost (default) | ~0.88 | ~0.82 | ~0.85 | ~0.86 | ~0.98 |
| XGBoost (tuned threshold) | varies | varies | optimized | ~0.86 | ~0.98 |

> *Results may vary slightly depending on random state and library versions.*

## 📁 Project Structure

```
credit-card-fraud-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── fraud_detection.py          # Full pipeline as Python script
├── fraud_detection.ipynb       # Jupyter Notebook (recommended)
└── outputs/                    # Generated plots (after running)
    ├── krok2_eda.png
    ├── krok4_cm_lr.png
    ├── krok5_cm_xgb.png
    ├── krok6_pr_auc.png
    ├── krok7_shap_bar.png
    ├── krok7_shap_beeswarm.png
    ├── krok8_threshold.png
    ├── krok9_porownanie.png
    └── krok9_wyniki.csv
```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 4. Run
**Option A — Jupyter Notebook (recommended):**
```bash
jupyter notebook fraud_detection.ipynb
```

**Option B — Python script:**
```bash
python fraud_detection.py
```

## 🔍 Key Findings

- **XGBoost significantly outperforms Logistic Regression** on this imbalanced dataset
- **SHAP analysis** reveals that V14, V4, V12, and V10 are the most important features for fraud detection
- **Threshold tuning** allows trading off precision for recall depending on business requirements
- **PR-AUC is the better metric** than ROC-AUC for imbalanced classification

## 🛠️ Technologies

- **pandas** & **numpy** — data manipulation
- **matplotlib** & **seaborn** — visualization
- **scikit-learn** — preprocessing, Logistic Regression, metrics
- **XGBoost** — gradient boosting classifier
- **SHAP** — model explainability

## 📄 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** Time, V1–V28 (PCA), Amount, Class
- **Size:** 284,807 transactions
- **Note:** The dataset is not included in this repo due to size. Download it from Kaggle.

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
