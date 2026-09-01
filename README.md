# ChurnGuard 🛡️ — Customer Retention Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-gemzjcnzqmxrfbux39mfx2.streamlit.app/)

A production-grade machine learning application that predicts customer churn for a telecom company and provides actionable retention recommendations.

**[🚀 Live Demo →](https://customer-churn-prediction-gemzjcnzqmxrfbux39mfx2.streamlit.app/)**

---

## Features

- **📊 Executive Dashboard** — Churn KPIs, revenue at risk, segment breakdowns
- **🔮 Real-time Prediction** — Enter any customer's details and get an instant calibrated churn probability
- **📊 SHAP Explainability** — Waterfall & bar charts showing exactly which features drove each prediction
- **💡 Retention Engine** — Personalised recommendations + counterfactual simulation (what happens if we act?)
- **📈 Business Insights** — Heatmaps, revenue share analysis, segment risk tables

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost + Isotonic Calibration (`CalibratedClassifierCV`) |
| Explainability | SHAP `TreeExplainer` |
| Frontend | Streamlit |
| Visualisation | Plotly, Matplotlib |
| Data | IBM Telco Customer Churn dataset (7,043 customers) |

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.843 |
| Brier Score (calibrated) | 0.138 |
| Precision (churn class) | 0.63 |
| Recall (churn class) | 0.56 |

> **Probability calibration**: `scale_pos_weight` in XGBoost inflates raw probabilities for imbalanced datasets. A held-out isotonic calibration layer corrects this, making the displayed % trustworthy.

---

## Run Locally

```bash
git clone https://github.com/TheSpectre542005/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model from scratch:
```bash
python train.py
```

---

## Project Structure

```
├── app.py                  # Streamlit multi-page application
├── retention_engine.py     # Recommendations + counterfactual simulation
├── train.py                # Standalone training + calibration script
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── models/
    ├── xgb_model.pkl       # Calibrated model (predictions)
    ├── xgb_base.pkl        # Raw XGBClassifier (SHAP)
    ├── label_encoders.pkl
    ├── feature_names.pkl
    └── scaler.pkl
```
