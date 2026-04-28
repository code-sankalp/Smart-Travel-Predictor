# ✈️ Smart Travel Predictor
### Trips & Travel Analytics Platform

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://smart-travel-predictor-mmvormu78mieoygbhfh8un.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> An end-to-end machine learning web application that predicts whether a travel customer will purchase a holiday package — combining interactive EDA, multi-model training, hyperparameter tuning, and real-time inference in a single polished dashboard.

---

## 🌐 Live Demo

**[👉 Open the App](https://smart-travel-predictor-mmvormu78mieoygbhfh8un.streamlit.app/)**

No setup required — runs entirely in the browser.

---

## 📌 Problem Statement

Travel companies face high customer acquisition costs with low conversion rates. This project builds a **binary classification system** to identify customers most likely to purchase a travel package, enabling sales teams to prioritise outreach and improve conversion efficiency.

- **Dataset**: 4,888 customer records with 18 demographic and behavioural features
- **Target**: `ProdTaken` — whether a customer purchased a holiday package (Yes/No)
- **Challenge**: Class imbalance (~81% No, ~19% Yes)

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 📊 **EDA Dashboard** | 15+ interactive Plotly charts — univariate, bivariate, and correlation analysis |
| 🤖 **Multi-Model Training** | Trains 6 classifiers simultaneously with a single click |
| ⚙️ **Hyperparameter Tuning** | `RandomizedSearchCV` with `StratifiedKFold` for all models |
| ⚖️ **Imbalance Handling** | `BorderlineSMOTE + Tomek Links` — state-of-the-art for 4:1 imbalance |
| 📈 **11 Evaluation Metrics** | Accuracy, F1, ROC-AUC, PR-AUC, MCC, Cohen Kappa, Brier Score, and more |
| 🎯 **Live Prediction** | Manual input form for single-customer conversion prediction with confidence scores |
| 🔍 **Auto Insights** | Context-aware statistical insights for every chart and column selected |

---

## 🛠️ Tech Stack

**Frontend / UI**
- [Streamlit](https://streamlit.io) — app framework
- [Plotly](https://plotly.com/python/) — interactive charts
- Custom CSS with glassmorphism dark theme

**Machine Learning**
- [scikit-learn](https://scikit-learn.org) — model training, tuning, evaluation
- [XGBoost](https://xgboost.readthedocs.io) — gradient boosting
- [imbalanced-learn](https://imbalanced-learn.org) — SMOTE + Tomek Links resampling

**Data Processing**
- [pandas](https://pandas.pydata.org) + [NumPy](https://numpy.org)

---

## 🤖 Models Trained

| Model | Tuned |
|---|---|
| Logistic Regression | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| Gradient Boosting | ✅ |
| AdaBoost | ✅ |
| XGBoost | ✅ (if installed) |

All models are evaluated on the same held-out test set, ranked by F1-score and PR-AUC — the most reliable metrics for this imbalanced dataset.

---

## ⚖️ Handling Class Imbalance

The dataset has a **4.31:1 imbalance ratio**. The app applies:

- **BorderlineSMOTE** — synthesises minority samples only near the decision boundary (harder, more informative samples than vanilla SMOTE)
- **Tomek Links** — removes ambiguous majority-class samples that blur the boundary

This combination produces a cleaner, more discriminative training set without overfitting to synthetic noise.

---

## 📂 Project Structure

```
Smart-Travel-Predictor/
│
├── app.py                  # Main Streamlit app — UI, routing, page logic
├── analysis.py             # EDA engine — data loading, KPIs, all chart functions
├── prediction.py           # ML engine — training, evaluation, prediction, charts
│
├── travel_dataset.csv      # Raw dataset
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/code-sankalp/Smart-Travel-Predictor.git
cd Smart-Travel-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

**Requirements**
```
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
imbalanced-learn
```

---

## 📊 App Pages

### 1. Overview
KPI cards (total customers, conversion rate, average income, passport %, etc.) and summary charts — purchase distribution, product pitched breakdown, age histogram, gender split, city tier analysis.

### 2. Univariate Analysis
Select any column + chart type (Histogram, Box Plot, Violin, ECDF, Bar, Pie, Donut, Treemap) and get an auto-generated chart with statistical insights and preprocessing recommendations.

### 3. Bivariate Analysis
Select X and Y axes + chart type (10 options including Scatter, KDE Overlay, Heatmap 2D Bin, grouped histograms) with correlation analysis and conversion-rate breakdowns.

### 4. Train All Models
One-click training pipeline: data prep → encoding → balancing → tuning → evaluation. Outputs a ranked comparison table, ROC curves, confusion matrices, and feature importance plots.

### 5. Predict
Choose any trained model, fill in customer details (18 fields), and get an instant prediction with a confidence probability bar.

---

## 📈 Sample Results

> Results vary by run due to randomised search. Typical performance on this dataset:

| Metric | Typical Range |
|---|---|
| F1-Score (best model) | 0.62 – 0.72 |
| ROC-AUC | 0.84 – 0.90 |
| PR-AUC | 0.58 – 0.68 |
| MCC | 0.52 – 0.64 |

---

## 👤 Author

**Sankalp** — [GitHub](https://github.com/code-sankalp)

---

