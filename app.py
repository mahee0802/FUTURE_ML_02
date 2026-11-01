"""Churn prediction demo.  Run from the repo root:  streamlit run app.py"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "notebook" / "churn_model.joblib"

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["threshold"]


if not MODEL_PATH.exists():
    st.error("Model file not found. Run `notebook/ML_2.ipynb` first to create `notebook/churn_model.joblib`.")
    st.stop()

pipeline, threshold = load_model()

st.title("📉 Customer Churn Prediction")
st.caption("Logistic Regression pipeline (one-hot + scaling) trained on the Telco Customer Churn dataset.")

YN = ["No", "Yes"]
ADDON = ["No", "Yes", "No internet service"]

with st.form("customer"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", YN, index=1)
        monthly = st.number_input("Monthly charges ($)", 15.0, 125.0, 70.0, step=0.5)
        total = st.number_input("Total charges ($)", 0.0, 9000.0,
                                float(round(tenure * monthly, 2)), step=10.0,
                                help="Defaults to tenure × monthly charges; edit if known.")

    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", YN, index=1)
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", ADDON)
        backup = st.selectbox("Online backup", ADDON)
        protection = st.selectbox("Device protection", ADDON)

    with c3:
        st.subheader("Other")
        tech = st.selectbox("Tech support", ADDON)
        tv = st.selectbox("Streaming TV", ADDON)
        movies = st.selectbox("Streaming movies", ADDON)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", YN)
        dependents = st.selectbox("Dependents", YN)

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    no_internet = internet == "No"
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else (multi if multi != "No phone service" else "No"),
        "InternetService": internet,
        "OnlineSecurity": "No internet service" if no_internet else (security if security != "No internet service" else "No"),
        "OnlineBackup": "No internet service" if no_internet else (backup if backup != "No internet service" else "No"),
        "DeviceProtection": "No internet service" if no_internet else (protection if protection != "No internet service" else "No"),
        "TechSupport": "No internet service" if no_internet else (tech if tech != "No internet service" else "No"),
        "StreamingTV": "No internet service" if no_internet else (tv if tv != "No internet service" else "No"),
        "StreamingMovies": "No internet service" if no_internet else (movies if movies != "No internet service" else "No"),
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])

    prob = float(pipeline.predict_proba(row)[0, 1])
    at_risk = prob >= threshold

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{prob:.1%}")
    m2.metric("Decision threshold", f"{threshold:.3f}")
    m3.metric("Prediction", "⚠️ Likely to churn" if at_risk else "✅ Likely to stay")
    st.progress(min(prob, 1.0))

    if at_risk:
        st.warning("Flag for the retention team.")
    else:
        st.success("Below the churn threshold.")

    with st.expander("Top factors moving this prediction"):
        names = pipeline.named_steps["pre"].get_feature_names_out()
        coefs = pipeline.named_steps["clf"].coef_[0]
        x = pipeline.named_steps["pre"].transform(row)
        x = x.toarray()[0] if hasattr(x, "toarray") else x[0]
        contrib = pd.Series(coefs * x, index=names)
        contrib = contrib[contrib != 0]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        st.dataframe(pd.DataFrame({
            "feature": [n.split("__", 1)[1] for n in top.index],
            "effect": ["↑ raises risk" if v > 0 else "↓ lowers risk" for v in top.values],
            "contribution": top.values.round(3),
        }), hide_index=True)
        st.caption("Contribution = coefficient × scaled feature value (log-odds).")