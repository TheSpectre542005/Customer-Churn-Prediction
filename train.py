"""
train.py — Standalone training script for ChurnGuard.

Outputs
-------
models/xgb_base.pkl        Raw XGBClassifier  (used by SHAP TreeExplainer)
models/xgb_model.pkl       CalibratedClassifierCV wrapper  (used for predictions)
models/scaler.pkl          StandardScaler  (kept for reference; XGB doesn't need it)
models/feature_names.pkl   list[str]
models/label_encoders.pkl  dict[str, LabelEncoder]

Run
---
    python train.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import sklearn

# FrozenEstimator was added in sklearn 1.6 to replace cv='prefit'
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df.drop(columns=["customerID"], inplace=True)

# ── 2. Feature engineering ────────────────────────────────────────────────────
service_cols = [
    "PhoneService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
df["ServiceCount"]      = df[service_cols].apply(lambda r: sum(v == "Yes" for v in r), axis=1)
df["CLV"]               = df["MonthlyCharges"] * 24
df["ContractRiskScore"] = df["Contract"].map({"Month-to-month": 3, "One year": 2, "Two year": 1})

# ── 3. Label-encode categoricals (fit on FULL dataset) ───────────────────────
os.makedirs("models", exist_ok=True)
df_ml    = df.copy()
cat_cols = df_ml.select_dtypes(include="object").columns.tolist()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(df_ml[col].astype(str))
    label_encoders[col] = le
    df_ml[col] = le.transform(df_ml[col].astype(str))

joblib.dump(label_encoders, "models/label_encoders.pkl")
print(f"Saved label_encoders.pkl  ({len(label_encoders)} columns)")

# ── 4. Split ──────────────────────────────────────────────────────────────────
X = df_ml.drop(columns=["Churn"])
y = df_ml["Churn"]
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/feature_names.pkl")
print(f"Saved feature_names.pkl   ({len(feature_names)} features)")

# Main split: 70% train | 15% calibration | 15% test
X_temp,  X_test,  y_temp,  y_test  = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_cal,   y_train, y_cal   = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
# 0.176 ≈ 15% of full dataset

print(f"\nSplit: train={len(X_train)}, calibration={len(X_cal)}, test={len(X_test)}")

# ── 5. Scaler (XGB doesn't need it; kept for reference) ──────────────────────
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, "models/scaler.pkl")
print("Saved scaler.pkl")

# ── 6. Train raw XGBoost ──────────────────────────────────────────────────────
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
print(f"\nTraining XGBClassifier  (neg={neg}, pos={pos}, spw={neg/pos:.2f})")

xgb_base = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=neg / pos,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
xgb_base.fit(X_train, y_train)
joblib.dump(xgb_base, "models/xgb_base.pkl")   # saved separately for SHAP
print("Saved xgb_base.pkl (raw, for SHAP)")

# ── 7. Calibrate on the held-out calibration set ─────────────────────────────
print("\nCalibrating with isotonic regression...")
if _HAS_FROZEN:
    # sklearn >= 1.6: use FrozenEstimator (no deprecation warning)
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(xgb_base), method="isotonic")
else:
    # sklearn < 1.6: use legacy cv='prefit'
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = CalibratedClassifierCV(estimator=xgb_base, method="isotonic", cv="prefit")
calibrated.fit(X_cal, y_cal)
joblib.dump(calibrated, "models/xgb_model.pkl")
print("Saved xgb_model.pkl (calibrated, for predictions)")

# ── 8. Evaluate ───────────────────────────────────────────────────────────────
y_prob_raw  = xgb_base.predict_proba(X_test)[:, 1]
y_prob_cal  = calibrated.predict_proba(X_test)[:, 1]
y_pred      = (y_prob_cal >= 0.4).astype(int)

print("\n=== Evaluation on hold-out test set ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC  (raw)       : {roc_auc_score(y_test, y_prob_raw):.4f}")
print(f"ROC-AUC  (calibrated): {roc_auc_score(y_test, y_prob_cal):.4f}")
print(f"Brier    (raw)       : {brier_score_loss(y_test, y_prob_raw):.4f}  (lower = better)")
print(f"Brier    (calibrated): {brier_score_loss(y_test, y_prob_cal):.4f}  (lower = better)")
print("\nAll model artifacts saved to models/")
