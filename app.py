import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from retention_engine import get_risk_category, get_recommendations, simulate_post_retention_risk

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Retention Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.6rem; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }

/* Page title styling */
h1 { color: #f1f5f9 !important; }
h2, h3 { color: #cbd5e1 !important; }

/* Info/warning boxes */
.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
}

/* Custom card containers */
.card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

/* Step indicator */
.step-pill {
    background: #3b82f6;
    color: white;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Calibrated model — used for all probability predictions
    model          = joblib.load("models/xgb_model.pkl")
    # Raw XGBClassifier — required by shap.TreeExplainer (doesn't accept wrappers)
    base_model     = joblib.load("models/xgb_base.pkl")
    feature_names  = joblib.load("models/feature_names.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    return model, base_model, feature_names, label_encoders


model, base_model, feature_names, label_encoders = load_model()


# ── Load and enrich dataset ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New (0-12m)", "Growing (1-2y)", "Established (2-4y)", "Loyal (4+y)"],
    )
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["ServiceCount"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    df["CLV"] = df["MonthlyCharges"] * 24
    df["ContractRiskScore"] = df["Contract"].map(
        {"Month-to-month": 3, "One year": 2, "Two year": 1}
    )
    return df


df = load_data()


# ── Encode a customer dict → model-ready DataFrame ───────────────────────────
def encode_customer(customer: dict) -> pd.DataFrame:
    """Uses saved label_encoders so integer codes match training-time values."""
    row = {}
    for feat in feature_names:
        val = customer.get(feat, 0)
        if feat in label_encoders:
            le = label_encoders[feat]
            val_str = str(val)
            val = int(le.transform([val_str])[0]) if val_str in le.classes_ else 0
        row[feat] = val
    return pd.DataFrame([row])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=56)
    st.title("ChurnGuard")
    st.caption("Customer Retention Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠  Dashboard",
            "🔮  Predict Customer",
            "📊  SHAP Explainer",
            "💡  Retention Engine",
            "📈  Business Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Workflow guide in the sidebar
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
**Recommended workflow**

1. **Dashboard** — get a bird's-eye view of churn KPIs
2. **Predict Customer** — enter customer details and run prediction
3. **SHAP Explainer** — see which features drove the prediction
4. **Retention Engine** — get personalised actions + simulate impact
5. **Business Insights** — segment & revenue analysis
""")

    # Prediction status in sidebar
    if "probability" in st.session_state:
        st.divider()
        prob = st.session_state["probability"]
        cat, col = get_risk_category(prob)
        st.markdown("**Active Prediction**")
        st.markdown(
            f'<span class="risk-badge" style="background:{col}22;color:{col};border:1px solid {col};">'
            f"{cat}  {prob:.1%}</span>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear prediction", use_container_width=True):
            for k in ["customer", "probability"]:
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.title("🛡️ ChurnGuard — Retention Analytics")
    st.caption("Real-time churn intelligence for customer success teams")
    st.divider()

    total        = len(df)
    churned      = int(df["Churn_Binary"].sum())
    churn_rate   = churned / total
    revenue_loss = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum() * 12
    avg_clv      = df["CLV"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers",         f"{total:,}")
    k2.metric("Churned Customers",       f"{churned:,}")
    k3.metric("Churn Rate",              f"{churn_rate:.1%}", delta="-2.3% vs last quarter")
    k4.metric("Est. Annual Revenue Risk",f"${revenue_loss:,.0f}")
    k5.metric("Avg Customer CLV",        f"${avg_clv:,.0f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        contract_churn = df.groupby("Contract")["Churn_Binary"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        contract_churn["Churn Rate"] *= 100
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate",
            color="Churn Rate", color_continuous_scale="Reds",
            title="Churn Rate by Contract Type (%)",
            text_auto=".1f",
        )
        fig.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#cbd5e1", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tenure_churn = (
            df.groupby("TenureGroup", observed=True)["Churn_Binary"].mean().reset_index()
        )
        tenure_churn.columns = ["Tenure Group", "Churn Rate"]
        tenure_churn["Churn Rate"] *= 100
        fig2 = px.bar(
            tenure_churn, x="Tenure Group", y="Churn Rate",
            color="Churn Rate", color_continuous_scale="Oranges",
            title="Churn Rate by Tenure Group (%)",
            text_auto=".1f",
        )
        fig2.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#cbd5e1", showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df, x="MonthlyCharges", color="Churn",
            barmode="overlay", opacity=0.75,
            title="Monthly Charges — Stayed vs Churned",
            color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
        )
        fig3.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#cbd5e1",
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        pay_churn = df.groupby("PaymentMethod")["Churn_Binary"].mean().reset_index()
        pay_churn.columns = ["Payment Method", "Churn Rate"]
        pay_churn["Churn Rate"] *= 100
        pay_churn = pay_churn.sort_values("Churn Rate", ascending=True)
        fig4 = px.bar(
            pay_churn, x="Churn Rate", y="Payment Method",
            orientation="h", color="Churn Rate",
            color_continuous_scale="Blues",
            title="Churn Rate by Payment Method (%)",
            text_auto=".1f",
        )
        fig4.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#cbd5e1", showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Churn rate by service count
    svc_churn = df.groupby("ServiceCount")["Churn_Binary"].mean().reset_index()
    svc_churn.columns = ["Services", "Churn Rate"]
    svc_churn["Churn Rate"] *= 100
    fig5 = px.line(
        svc_churn, x="Services", y="Churn Rate",
        markers=True, title="Churn Rate vs Number of Services Used",
        color_discrete_sequence=["#6366f1"],
    )
    fig5.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#cbd5e1",
    )
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict Customer":
    st.title("🔮 Customer Churn Prediction")
    st.caption("Enter customer details and click **Predict** to get a real-time risk score.")
    st.divider()

    # ── Input form ──────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Demographics")
            gender     = st.selectbox("Gender",         ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
            partner    = st.selectbox("Partner",        ["Yes", "No"])
            dependents = st.selectbox("Dependents",     ["Yes", "No"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12,
                                   help="How long the customer has been with you")

        with col2:
            st.markdown("##### 📡 Services")
            phone    = st.selectbox("Phone Service",     ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines",    ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service",  ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security",   ["Yes", "No", "No internet service"])
            backup   = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
            device   = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech     = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
            tv       = st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])
            movies   = st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("##### 💳 Account")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            billing  = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            monthly  = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            # Auto-estimate TotalCharges from tenure × monthly; let user override
            est_total = round(float(monthly) * max(tenure, 1), 2)
            total_c  = st.number_input(
                "Total Charges ($)",
                0.0, 12_000.0, est_total, step=1.0,
                help=f"Auto-estimated from tenure × monthly = ${est_total:,.2f}. Override if you have the actual value.",
            )

        submitted = st.form_submit_button(
            "🔮  Predict Churn Risk", type="primary", use_container_width=True
        )

    # ── Run prediction ───────────────────────────────────────────────────────
    if submitted:
        service_count = sum(
            1 for v in [phone, security, backup, device, tech, tv, movies] if v == "Yes"
        )
        clv           = monthly * 24
        contract_risk = {"Month-to-month": 3, "One year": 2, "Two year": 1}[contract]

        customer = {
            "gender": gender,        "SeniorCitizen": senior,
            "Partner": partner,      "Dependents": dependents,
            "tenure": tenure,        "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security,"OnlineBackup": backup,
            "DeviceProtection": device,"TechSupport": tech,
            "StreamingTV": tv,        "StreamingMovies": movies,
            "Contract": contract,     "PaperlessBilling": billing,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total_c,  "ServiceCount": service_count,
            "CLV": clv,               "ContractRiskScore": contract_risk,
        }

        df_row = encode_customer(customer)
        prob   = float(model.predict_proba(df_row)[0][1])
        category, color = get_risk_category(prob)

        # Persist for other pages
        st.session_state["customer"]    = customer
        st.session_state["probability"] = prob

        st.divider()

        # Risk summary banner
        st.markdown(
            f'<div style="background:{color}18;border:1.5px solid {color};border-radius:12px;'
            f'padding:18px 24px;margin-bottom:16px;">'
            f'<span style="font-size:1.4rem;font-weight:700;color:{color};">{category}</span>'
            f'&nbsp;&nbsp;&nbsp;<span style="color:#94a3b8;font-size:1rem;">'
            f'Churn Probability: <strong style="color:{color};">{prob:.1%}</strong></span></div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Category",     category)
        r3.metric("Customer CLV",      f"${clv:,.0f}")
        r4.metric("Services Used",     service_count)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={"text": "Churn Risk Score", "font": {"color": "#cbd5e1"}},
            number={"suffix": "%", "font": {"color": "#f1f5f9"}},
            delta={"reference": 26.5, "suffix": "%", "decreasing": {"color": "#22c55e"},
                   "increasing": {"color": "#ef4444"}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#475569"},
                "bar":     {"color": color, "thickness": 0.25},
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0,  25],  "color": "#dcfce720"},
                    {"range": [25, 50],  "color": "#fef9c320"},
                    {"range": [50, 75],  "color": "#ffedd520"},
                    {"range": [75, 100], "color": "#fee2e220"},
                ],
                "threshold": {
                    "line": {"color": "#f1f5f9", "width": 2},
                    "thickness": 0.8,
                    "value": prob * 100,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="#0f172a",
            font_color="#cbd5e1",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick-navigate CTA
        st.info(
            "✅ Prediction saved. Use the **sidebar** to navigate to "
            "📊 SHAP Explainer or 💡 Retention Engine for next steps."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  SHAP Explainer":
    st.title("📊 SHAP Explainability")
    st.caption("Understand *why* the model predicted this churn probability for the customer.")

    if "customer" not in st.session_state:
        st.warning(
            "⚠️ No prediction found. Please go to **🔮 Predict Customer** first and run a prediction."
        )
        st.stop()

    customer = st.session_state["customer"]
    prob     = st.session_state["probability"]
    cat, col = get_risk_category(prob)

    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            f"**Active customer risk:** "
            f'<span class="risk-badge" style="background:{col}22;color:{col};border:1px solid {col};">'
            f"{cat}  {prob:.1%}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.caption("Features in red push toward churn; blue pushes away.")

    st.divider()

    # Compute SHAP — uses the raw base XGBClassifier (TreeExplainer requires it)
    df_row    = encode_customer(customer)
    with st.spinner("Computing SHAP values…"):
        explainer = shap.TreeExplainer(base_model)
        shap_vals = explainer.shap_values(df_row)

    # Waterfall plot — individual customer
    st.subheader("Feature Contributions for This Customer")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f172a")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=df_row.iloc[0],
            feature_names=feature_names,
        ),
        show=False,
    )
    ax.set_facecolor("#0f172a")
    ax.tick_params(colors="#94a3b8")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Bar chart of absolute SHAP contributions
    st.subheader("Top 10 Contributing Features")
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals[0],
        "Abs SHAP": np.abs(shap_vals[0]),
    }).sort_values("Abs SHAP", ascending=False).head(10)

    shap_df["Direction"] = shap_df["SHAP Value"].apply(
        lambda v: "Increases churn risk" if v > 0 else "Decreases churn risk"
    )

    fig2 = px.bar(
        shap_df, x="SHAP Value", y="Feature", orientation="h",
        color="Direction",
        color_discrete_map={
            "Increases churn risk":  "#ef4444",
            "Decreases churn risk": "#22c55e",
        },
        title="SHAP Feature Contributions",
    )
    fig2.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#cbd5e1", yaxis={"autorange": "reversed"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Full SHAP values table"):
        st.dataframe(
            shap_df.drop(columns=["Abs SHAP"]).rename(columns={"SHAP Value": "SHAP Impact"}),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RETENTION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡  Retention Engine":
    st.title("💡 Retention Recommendation Engine")
    st.caption("Personalised actions and counterfactual impact simulation.")

    if "customer" not in st.session_state:
        st.warning(
            "⚠️ No prediction found. Please go to **🔮 Predict Customer** first and run a prediction."
        )
        st.stop()

    customer        = st.session_state["customer"]
    prob            = st.session_state["probability"]
    category, color = get_risk_category(prob)

    st.divider()

    # Risk header
    st.markdown(
        f'<div style="background:{color}18;border:1.5px solid {color};border-radius:12px;'
        f'padding:16px 22px;margin-bottom:16px;">'
        f'<span style="font-size:1.2rem;font-weight:700;color:{color};">{category}</span>'
        f'&nbsp;&nbsp;<span style="color:#94a3b8;">Current churn probability: '
        f'<strong style="color:{color};">{prob:.1%}</strong></span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Charges",  f"${customer.get('MonthlyCharges', 0):,.2f}")
    c2.metric("Customer Tenure",  f"{customer.get('tenure', 0)} months")
    c3.metric("CLV (24 months)",  f"${customer.get('CLV', 0):,.0f}")

    st.divider()

    # ── Recommendations ────────────────────────────────────────────────────
    recommendations = get_recommendations(customer, prob)
    st.subheader("🎯 Recommended Retention Actions")

    impact_icon  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    impact_label = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}

    for i, rec in enumerate(recommendations):
        ic = impact_icon[rec["impact"]]
        lc = impact_label[rec["impact"]]
        with st.expander(
            f"{ic} **{rec['action']}** — Impact: {rec['impact']}", expanded=True
        ):
            st.markdown(f"📌 {rec['reason']}")
            st.markdown(
                f'<span style="background:{lc}22;color:{lc};border:1px solid {lc};'
                f'border-radius:8px;padding:2px 10px;font-size:0.8rem;font-weight:600;">'
                f"Impact: {rec['impact']}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Counterfactual simulation ───────────────────────────────────────────
    st.subheader("📉 Simulate Post-Retention Risk")
    st.caption(
        "This simulation applies the highest-impact interventions (contract upgrade, "
        "15% discount, tech support) and re-runs the model to estimate the new risk."
    )

    with st.spinner("Running counterfactual simulation…"):
        new_prob, improvement = simulate_post_retention_risk(
            customer, prob, model, feature_names, label_encoders
        )

    s1, s2, s3 = st.columns(3)
    s1.metric("Current Risk",              f"{prob:.1%}")
    s2.metric("Projected Risk After Actions", f"{new_prob:.1%}",
              delta=f"-{improvement:.1%}", delta_color="inverse")
    s3.metric("Estimated Improvement",     f"{improvement:.1%}")

    # Before / after gauge
    fig = go.Figure()
    for val, name, clr in [
        (prob * 100,     "Before",  "#ef4444"),
        (new_prob * 100, "After",   "#22c55e"),
    ]:
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": name, "font": {"color": "#cbd5e1", "size": 14}},
            number={"suffix": "%", "font": {"color": "#f1f5f9"}},
            gauge={
                "axis":    {"range": [0, 100], "tickcolor": "#475569"},
                "bar":     {"color": clr, "thickness": 0.25},
                "bgcolor": "#1e293b",
                "steps": [
                    {"range": [0,  25],  "color": "#dcfce720"},
                    {"range": [25, 50],  "color": "#fef9c320"},
                    {"range": [50, 75],  "color": "#ffedd520"},
                    {"range": [75, 100], "color": "#fee2e220"},
                ],
            },
            domain={"x": [0, 0.45], "y": [0, 1]} if name == "Before"
                   else {"x": [0.55, 1], "y": [0, 1]},
        ))

    fig.update_layout(
        paper_bgcolor="#0f172a",
        font_color="#cbd5e1",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    if improvement > 0:
        st.success(
            f"✅ Applying the recommended actions could reduce churn risk "
            f"by **{improvement:.1%}** (from {prob:.1%} → {new_prob:.1%})."
        )
    else:
        st.info(
            "ℹ️ Simulated interventions did not significantly reduce risk for this customer. "
            "Consider a more tailored approach."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Business Insights":
    st.title("📈 Business Insights")
    st.caption("Segment analysis and revenue impact across the customer base.")
    st.divider()

    # KPI summary strip
    total_rev_risk = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum() * 12
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Annual Revenue at Risk", f"${total_rev_risk:,.0f}")
    k2.metric(
        "Avg Monthly Charges — Churned",
        f"${df[df['Churn']=='Yes']['MonthlyCharges'].mean():,.2f}",
    )
    k2.metric(
        "Avg Monthly Charges — Retained",
        f"${df[df['Churn']=='No']['MonthlyCharges'].mean():,.2f}",
    )
    k3.metric(
        "High-risk Segment Size",
        f"{(df['Churn_Binary'] == 1).sum():,}",
        help="Customers who actually churned in this dataset",
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        seg = (
            df.groupby(["Contract", "InternetService"])["Churn_Binary"]
            .mean()
            .reset_index()
        )
        seg.columns = ["Contract", "Internet", "Churn Rate"]
        seg["Churn Rate"] *= 100
        fig = px.bar(
            seg, x="Contract", y="Churn Rate", color="Internet",
            barmode="group", title="Churn by Contract + Internet Service (%)",
            text_auto=".1f",
        )
        fig.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#cbd5e1"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        revenue = df.groupby("Contract")["MonthlyCharges"].sum().reset_index()
        fig2 = px.pie(
            revenue, values="MonthlyCharges", names="Contract",
            title="Monthly Revenue Share by Contract Type",
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4,
        )
        fig2.update_layout(
            paper_bgcolor="#0f172a", font_color="#cbd5e1"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heat-map: churn rate by tenure group × contract
    st.subheader("Churn Rate Heatmap — Tenure × Contract")
    pivot = (
        df.groupby(["TenureGroup", "Contract"], observed=True)["Churn_Binary"]
        .mean()
        .unstack("Contract") * 100
    )
    fig3 = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="RdYlGn_r",
        title="Churn Rate (%) by Tenure Group × Contract Type",
        labels={"color": "Churn Rate (%)"},
    )
    fig3.update_layout(paper_bgcolor="#0f172a", font_color="#cbd5e1")
    st.plotly_chart(fig3, use_container_width=True)

    # Revenue at risk table
    st.subheader("Revenue at Risk by Segment")
    risk_rev = (
        df[df["Churn"] == "Yes"]
        .groupby("Contract")["MonthlyCharges"]
        .agg(["sum", "count"])
        .reset_index()
    )
    risk_rev.columns = ["Contract", "Monthly Revenue at Risk", "Customers at Risk"]
    risk_rev["Annual Revenue at Risk"] = risk_rev["Monthly Revenue at Risk"] * 12
    risk_rev["Avg Monthly / Customer"] = (
        risk_rev["Monthly Revenue at Risk"] / risk_rev["Customers at Risk"]
    )
    st.dataframe(
        risk_rev.style.format({
            "Monthly Revenue at Risk":  "${:,.0f}",
            "Annual Revenue at Risk":   "${:,.0f}",
            "Avg Monthly / Customer":   "${:,.2f}",
        }),
        use_container_width=True,
    )