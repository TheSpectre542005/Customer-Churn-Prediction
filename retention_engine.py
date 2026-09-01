"""
retention_engine.py — Retention scoring, recommendations, and counterfactual simulation.
"""


def get_risk_category(probability):
    """Map churn probability to a labelled risk tier."""
    if probability < 0.25:
        return "Low Risk", "#22c55e"
    elif probability < 0.50:
        return "Medium Risk", "#f59e0b"
    elif probability < 0.75:
        return "High Risk", "#f97316"
    else:
        return "Critical Risk", "#ef4444"


def get_recommendations(customer, probability):
    """
    Generate personalised retention recommendations based on
    customer profile and churn probability.
    """
    recommendations = []

    if customer.get("Contract") == "Month-to-month":
        recommendations.append({
            "action": "Upgrade to Annual Contract",
            "reason": "Month-to-month customers churn at 2× the rate of annual customers",
            "impact": "High",
        })

    if customer.get("tenure", 0) < 12:
        recommendations.append({
            "action": "Enroll in Onboarding Program",
            "reason": "First 12 months are the critical retention window",
            "impact": "High",
        })

    if customer.get("MonthlyCharges", 0) > 65:
        recommendations.append({
            "action": "Offer 15% Loyalty Discount",
            "reason": "High monthly charges are a top churn driver per SHAP analysis",
            "impact": "High",
        })

    if customer.get("TechSupport") == "No":
        recommendations.append({
            "action": "Provide 3 Months Free Tech Support",
            "reason": "Customers without tech support churn significantly more",
            "impact": "Medium",
        })

    if customer.get("ServiceCount", 0) >= 4:
        recommendations.append({
            "action": "Offer Bundled Services Discount",
            "reason": "Customer uses multiple services — bundle pricing increases stickiness",
            "impact": "Medium",
        })

    if customer.get("CLV", 0) > 3000:
        recommendations.append({
            "action": "Assign Dedicated Account Manager",
            "reason": "High lifetime value customer — priority retention is justified",
            "impact": "High",
        })

    if not recommendations:
        recommendations.append({
            "action": "Schedule Quarterly Check-in Call",
            "reason": "Proactive engagement maintains satisfaction",
            "impact": "Low",
        })

    return recommendations


def simulate_post_retention_risk(customer, probability, model, feature_names, label_encoders):
    """
    Simulate churn probability AFTER applying the recommended interventions.

    FIX: Uses the pre-trained label_encoders (saved during training) so that
    categorical values are mapped to the same integer codes the model was
    trained on.  The old code re-fit a fresh LabelEncoder on a single value,
    which always returned 0 — giving completely wrong counterfactual scores.

    Parameters
    ----------
    customer       : dict   — raw customer feature dict (string categoricals)
    probability    : float  — original predicted churn probability
    model          : XGBClassifier
    feature_names  : list[str]
    label_encoders : dict[str, LabelEncoder]  — from models/label_encoders.pkl

    Returns
    -------
    (new_prob, improvement) : (float, float)
    """
    import pandas as pd

    customer_improved = customer.copy()

    # Apply the most impactful interventions
    if customer.get("Contract") == "Month-to-month":
        customer_improved["Contract"] = "One year"
        customer_improved["ContractRiskScore"] = 2

    if customer.get("MonthlyCharges", 0) > 65:
        customer_improved["MonthlyCharges"] = customer["MonthlyCharges"] * 0.85
        customer_improved["CLV"] = customer_improved["MonthlyCharges"] * 24

    if customer.get("TechSupport") == "No":
        customer_improved["TechSupport"] = "Yes"

    # Encode using the ACTUAL trained label encoders
    row = {}
    for feat in feature_names:
        val = customer_improved.get(feat, 0)
        if feat in label_encoders:
            le = label_encoders[feat]
            val_str = str(val)
            val = int(le.transform([val_str])[0]) if val_str in le.classes_ else 0
        row[feat] = val

    df_row = pd.DataFrame([row])
    new_prob = float(model.predict_proba(df_row)[0][1])
    improvement = probability - new_prob

    return round(new_prob, 4), round(improvement, 4)
