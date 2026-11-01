"""Customer churn analytics: dashboard, model performance and live predictor.

Run from the repo root:  streamlit run app.py
(Run notebook/ML_2.ipynb first so data/churn_predictions.csv and
 notebook/churn_model.joblib exist.)
"""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

ROOT = Path(__file__).parent
PRED_PATH = ROOT / "data" / "churn_predictions.csv"
CV_PATH = ROOT / "data" / "cv_results.csv"
MODEL_PATH = ROOT / "notebook" / "churn_model.joblib"

# ── Palette ──────────────────────────────────────────────────────────────────
NAVY, BLUE, LIGHT_BLUE = "#0B2545", "#1D6FB8", "#9CC3E8"
CHURN, STAY, GREY = "#E4572E", "#1D6FB8", "#94A3B8"
OUTCOME_COLORS = {"Retained": STAY, "Churned": CHURN}

st.set_page_config(page_title="Churn Analytics", page_icon="📉", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 3.2rem; max-width: 1400px; }}
    .hero {{ background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 100%); color: #fff;
             padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.1rem; }}
    .hero h1 {{ margin: 0; font-size: 2rem; font-weight: 700; color: #fff; padding: 0; }}
    .hero p  {{ margin: .35rem 0 0 0; font-size: .98rem; opacity: .85; }}
    .kpi {{ background: #fff; border: 1px solid #E3E8EF; border-radius: 14px;
            padding: .95rem 1.2rem; box-shadow: 0 1px 3px rgba(16,24,40,.06); min-height: 138px; }}
    .kpi-label {{ font-size: .78rem; font-weight: 600; letter-spacing: .04em;
                  text-transform: uppercase; color: #64748B; }}
    .kpi-value {{ font-size: 2.1rem; font-weight: 700; color: {NAVY}; line-height: 1.25; }}
    .kpi-sub   {{ font-size: .8rem; color: #64748B; min-height: 1.1rem; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; border-radius: 14px; border-color: #E3E8EF; }}
    button[data-baseweb="tab"] {{ font-size: 1rem; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_PATH)
    df["Outcome"] = df["Actual_Churn"].map({0: "Retained", 1: "Churned"})
    df["Internet"] = df["InternetService"].replace({"No": "No internet"})
    bins = [-1, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
    df["TenureBand"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


@st.cache_data
def load_cv():
    if not CV_PATH.exists():
        return None
    cv = pd.read_csv(CV_PATH)
    first = cv.columns[0]
    if first != "Model":
        if not pd.api.types.is_numeric_dtype(cv[first]):   # model names saved under a blank/"Unnamed: 0" header
            cv = cv.rename(columns={first: "Model"})
        else:                                   # saved with index=False, so names were lost
            cv.insert(0, "Model", ["Logistic Regression", "Random Forest", "XGBoost"][:len(cv)])
    return cv


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["threshold"]


# ── UI helpers ───────────────────────────────────────────────────────────────
def kpi(col, label, value, sub="", accent=BLUE):
    col.markdown(
        f'<div class="kpi" style="border-top:4px solid {accent}">'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style(fig, title, height=340, legend=True):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", y=0.97, yanchor="top", yref="container",
                   font=dict(size=15, color=NAVY)),
        height=height, margin=dict(l=10, r=10, t=78 if legend else 55, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155"), showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.04, yanchor="bottom", title_text=""),
        hoverlabel=dict(bgcolor="#fff", font_color=NAVY),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E8EDF3", zeroline=False)
    return fig


def show(fig):
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── Tab 1: dashboard ─────────────────────────────────────────────────────────
def render_dashboard(df: pd.DataFrame):
    contracts = ["Month-to-month", "One year", "Two year"]
    internets = ["DSL", "Fiber optic", "No internet"]
    payments = sorted(df["PaymentMethod"].unique())

    with st.container(border=True):
        f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1.5, 1.2, 1.2])
        c_sel = f1.multiselect("Contract", contracts, default=contracts)
        i_sel = f2.multiselect("Internet service", internets, default=internets)
        p_sel = f3.multiselect("Payment method", payments, default=payments)
        t_lo, t_hi = f4.slider("Tenure (months)", 0, 72, (0, 72))
        cutoff = f5.slider("High-risk cutoff", 0.50, 0.95, 0.70, 0.05,
                           help="Customers with churn probability at or above this value count as high-risk.")

    d = df[df["Contract"].isin(c_sel) & df["Internet"].isin(i_sel) & df["PaymentMethod"].isin(p_sel)
           & df["tenure"].between(t_lo, t_hi)]
    if d.empty:
        st.warning("No customers match the current filters.")
        return

    high = d[d["Churn_Probability"] >= cutoff]

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, "Customers analysed", f"{len(d):,}", "held-out test set", GREY)
    kpi(k2, "Churn rate", f"{d['Actual_Churn'].mean():.1%}", f"{int(d['Actual_Churn'].sum()):,} customers churned", CHURN)
    kpi(k3, "Avg churn probability", f"{d['Churn_Probability'].mean():.2f}", "model risk score*", BLUE)
    hr_sub = f"{high['Actual_Churn'].mean():.0%} of them actually churned" if len(high) else "no customers at this cutoff"
    kpi(k4, "High-risk customers", f"{len(high):,}", hr_sub, CHURN)
    kpi(k5, "Monthly revenue at risk", f"${high['MonthlyCharges'].sum():,.0f}", "high-risk customers only", NAVY)
    st.caption("*Risk score, not a calibrated probability. See “How to read these numbers” at the bottom.")

    # Row 1: split / internet / contract
    r1a, r1b, r1c = st.columns([1, 1.3, 1.2])

    with r1a:
        counts = d["Outcome"].value_counts().reindex(["Retained", "Churned"]).fillna(0)
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.64, sort=False,
            marker=dict(colors=[STAY, CHURN], line=dict(color="#fff", width=2)),
            textinfo="percent", insidetextorientation="horizontal", textfont=dict(size=13, color="#fff"),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
        fig.add_annotation(text=f"<b>{d['Actual_Churn'].mean():.1%}</b><br><span style='font-size:11px'>churned</span>",
                           showarrow=False, font=dict(size=22, color=NAVY))
        show(style(fig, "Churn vs non-churn"))

    with r1b:
        g = d.groupby(["Internet", "Outcome"], observed=True).size().reset_index(name="Customers")
        fig = px.bar(g, x="Internet", y="Customers", color="Outcome", color_discrete_map=OUTCOME_COLORS,
                     category_orders={"Internet": internets, "Outcome": ["Retained", "Churned"]}, text="Customers")
        fig.update_traces(textposition="inside", textfont_color="#fff", marker_line_width=0)
        fig.update_layout(barmode="stack", xaxis_title="", yaxis_title="Customers")
        show(style(fig, "Internet service vs churn"))

    with r1c:
        g = d.groupby("Contract")["Churn_Probability"].mean().reindex(
            [c for c in contracts if c in d["Contract"].unique()]).reset_index()
        fig = px.bar(g, x="Contract", y="Churn_Probability", text="Churn_Probability")
        fig.update_traces(marker_color=BLUE, texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="", yaxis_title="Avg churn probability")
        fig.update_yaxes(range=[0, max(g["Churn_Probability"].max() * 1.2, 0.1)])
        show(style(fig, "Avg churn probability by contract", legend=False))

    # Row 2: tenure / payment / distribution
    r2a, r2b, r2c = st.columns([1.3, 1.1, 1.2])

    with r2a:
        t = d.groupby("TenureBand", observed=True).agg(
            actual=("Actual_Churn", "mean"), predicted=("Churn_Probability", "mean")).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=t["TenureBand"], y=t["actual"], name="Actual churn rate", mode="lines+markers",
                        line=dict(color=CHURN, width=3), marker=dict(size=8))
        fig.add_scatter(x=t["TenureBand"], y=t["predicted"], name="Avg predicted probability", mode="lines+markers",
                        line=dict(color=BLUE, width=3, dash="dot"), marker=dict(size=8))
        fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Churn risk", yaxis_tickformat=".0%")
        show(style(fig, "Churn risk falls sharply with tenure"))

    with r2b:
        g = d.groupby("PaymentMethod")["Churn_Probability"].mean().sort_values().reset_index()
        fig = px.bar(g, x="Churn_Probability", y="PaymentMethod", orientation="h", text="Churn_Probability")
        fig.update_traces(marker_color=BLUE, texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Avg churn probability", yaxis_title="")
        fig.update_xaxes(range=[0, max(g["Churn_Probability"].max() * 1.2, 0.1)])
        fig.update_yaxes(showgrid=False)
        show(style(fig, "Avg churn probability by payment method", legend=False))

    with r2c:
        fig = px.histogram(d, x="Churn_Probability", color="Outcome", nbins=40, color_discrete_map=OUTCOME_COLORS,
                           category_orders={"Outcome": ["Retained", "Churned"]})
        fig.update_layout(barmode="stack", xaxis_title="Churn probability", yaxis_title="Customers", bargap=0.04)
        fig.add_vline(x=cutoff, line_dash="dash", line_color=NAVY,
                      annotation_text=f"High-risk ≥ {cutoff:.2f}", annotation_position="top left",
                      annotation_font=dict(size=11, color=NAVY))
        show(style(fig, "Churn probability distribution"))

    # At-risk customer list
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.markdown(f"#### 🎯 Retention call list: high-risk customers (probability ≥ {cutoff:.2f})")
        if high.empty:
            st.info("No customers reach this cutoff with the current filters. Lower the high-risk cutoff.")
        else:
            lst = high.sort_values("Churn_Probability", ascending=False)[
                ["customerID", "Contract", "InternetService", "PaymentMethod", "tenure",
                 "MonthlyCharges", "Churn_Probability", "Outcome"]].copy()
            lst["Churn_Probability"] = (lst["Churn_Probability"] * 100).round(1)
            lst.columns = ["Customer ID", "Contract", "Internet", "Payment method", "Tenure (mo)",
                           "Monthly ($)", "Churn risk", "Actual outcome"]
            h2.download_button("⬇ Download CSV", lst.to_csv(index=False).encode(), "high_risk_customers.csv",
                               "text/csv", width="stretch")
            st.dataframe(
                lst, hide_index=True, width="stretch", height=300,
                column_config={
                    "Churn risk": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=100, format="%.0f%%"),
                    "Monthly ($)": st.column_config.NumberColumn(format="$%.2f"),
                })

    with st.expander("ℹ️ How to read these numbers"):
        st.markdown(
            f"""
- **Data:** the {len(df):,} customers in the 20% **held-out test set**. The model never saw them in training,
  so these are honest out-of-sample results.
- **Churn probability is a risk *score*, not a calibrated probability.** The model is trained with
  `class_weight='balanced'` to catch more churners, which pushes scores upward: the average score is
  **{df['Churn_Probability'].mean():.2f}** versus a true churn rate of **{df['Actual_Churn'].mean():.1%}**.
  The *ranking* is what matters (ROC-AUC ≈ {roc_auc_score(df['Actual_Churn'], df['Churn_Probability']):.2f}).
  Probability calibration is listed as a next step.
- **High-risk** is a business cutoff you can move with the slider. The model's own decision threshold
  (tuned for best F1) is separate and drives the *Predicted churn* flag.
"""
        )


# ── Tab 2: model performance ─────────────────────────────────────────────────
def render_model(df: pd.DataFrame):
    y, p, pred = df["Actual_Churn"], df["Churn_Probability"], df["Predicted_Churn"]
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary")
    threshold = load_model()[1] if MODEL_PATH.exists() else None

    st.markdown("##### Final model: Logistic Regression (class-weighted) · test set"
                + (f" · tuned threshold **{threshold:.3f}**" if threshold else ""))
    m1, m2, m3, m4, m5 = st.columns(5)
    kpi(m1, "ROC-AUC", f"{roc_auc_score(y, p):.3f}", "ranking quality", BLUE)
    kpi(m2, "Accuracy", f"{accuracy_score(y, pred):.1%}", f"baseline {1 - y.mean():.1%} (predict all stay)", GREY)
    kpi(m3, "Recall (churn)", f"{rec:.1%}", "churners caught", CHURN)
    kpi(m4, "Precision (churn)", f"{prec:.1%}", "flagged who really churn", BLUE)
    kpi(m5, "F1 (churn)", f"{f1:.3f}", "threshold-tuned objective", NAVY)
    st.write("")

    a, b, c = st.columns([1.2, 1, 1])
    with a:
        fpr, tpr, _ = roc_curve(y, p)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name="Logistic Regression", fill="tozeroy",
                        fillcolor="rgba(29,111,184,.12)", line=dict(color=BLUE, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color=GREY, dash="dash"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        show(style(fig, f"ROC curve (AUC = {roc_auc_score(y, p):.3f})", height=360))

    with b:
        cm = confusion_matrix(y, pred)
        fig = px.imshow(cm, text_auto=True, x=["Stay", "Churn"], y=["Stay", "Churn"],
                        color_continuous_scale=[[0, "#EAF2FB"], [1, BLUE]], aspect="auto")
        fig.update_traces(textfont=dict(size=20))
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", coloraxis_showscale=False)
        show(style(fig, "Confusion matrix", height=360, legend=False))

    with c:
        r = pd.DataFrame({"actual": y.values, "prob": p.values}).sort_values("prob", ascending=False).reset_index(drop=True)
        r["decile"] = pd.qcut(r.index, 10, labels=range(1, 11))
        lift = r.groupby("decile", observed=True)["actual"].agg(["sum", "mean"])
        top_share = lift.loc[1, "sum"] / r["actual"].sum()
        fig = px.bar(x=lift.index.astype(str), y=lift["mean"], text=lift["mean"])
        fig.update_traces(marker_color=[CHURN if i == 0 else LIGHT_BLUE for i in range(10)],
                          texttemplate="%{text:.0%}", textposition="outside", cliponaxis=False)
        fig.add_hline(y=y.mean(), line_dash="dash", line_color=NAVY,
                      annotation_text=f"overall {y.mean():.1%}", annotation_position="top right")
        fig.update_layout(xaxis_title="Risk decile (1 = highest risk)", yaxis_title="Actual churn rate",
                          yaxis_tickformat=".0%", xaxis_type="category")
        fig.update_yaxes(range=[0, lift["mean"].max() * 1.2])
        show(style(fig, f"Top 10% of customers hold {top_share:.1%} of churners", height=360, legend=False))

    d1, d2 = st.columns([1.1, 1])
    with d1:
        cv = load_cv()
        with st.container(border=True):
            st.markdown("**5-fold stratified cross-validation (mean)**")
            if cv is None:
                st.info("Run the notebook to create `data/cv_results.csv`.")
            else:
                st.dataframe(cv.set_index("Model").style.format("{:.3f}").highlight_max(axis=0, color="#D6E8F8"),
                             width="stretch")
                st.caption("Logistic Regression and XGBoost are almost tied on ROC-AUC, so the simpler, "
                           "more interpretable model is the final one. Highlighted = best per metric.")
        if threshold is not None:
            with st.container(border=True):
                st.markdown("**Default vs tuned decision threshold (test set)**")
                rows = []
                for label, t in [("Default 0.500", 0.5), (f"Tuned {threshold:.3f}", threshold)]:
                    yhat = (p >= t).astype(int)
                    pp, rr, ff, _ = precision_recall_fscore_support(y, yhat, average="binary")
                    rows.append({"Threshold": label, "Accuracy": accuracy_score(y, yhat),
                                 "Precision": pp, "Recall": rr, "F1": ff})
                st.dataframe(pd.DataFrame(rows).set_index("Threshold").style.format("{:.3f}"), width="stretch")
                st.caption("The threshold is picked at the best F1 on the precision-recall curve. "
                           "A cost-based threshold (offer cost vs customer value) is the natural next step.")
    with d2:
        if MODEL_PATH.exists():
            pipe, _ = load_model()
            names = [n.split("__", 1)[1] for n in pipe.named_steps["pre"].get_feature_names_out()]
            coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=names).sort_values()
            top = pd.concat([coefs.head(7), coefs.tail(7)])
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker_color=[STAY if v < 0 else CHURN for v in top.values]))
            fig.update_layout(xaxis_title="Coefficient (log-odds)", yaxis_title="")
            fig.update_yaxes(showgrid=False)
            show(style(fig, "Top churn drivers (red raises risk, blue lowers it)", height=500, legend=False))
        else:
            st.info("Run the notebook to create `notebook/churn_model.joblib`.")


# ── Tab 3: single-customer predictor ─────────────────────────────────────────
def render_predictor():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `notebook/ML_2.ipynb` first to create `notebook/churn_model.joblib`.")
        return
    pipeline, threshold = load_model()

    st.markdown("Enter a customer's details to score them with the trained pipeline.")
    YN = ["No", "Yes"]
    ADDON = ["No", "Yes", "No internet service"]

    with st.form("customer"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless billing", YN, index=1)
            monthly = st.number_input("Monthly charges ($)", 15.0, 125.0, 70.0, step=0.5)
            total = st.number_input("Total charges ($)", 0.0, 9000.0, float(round(tenure * monthly, 2)), step=10.0,
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

    if not submitted:
        return

    no_internet = internet == "No"

    def addon(v):
        return "No internet service" if no_internet else (v if v != "No internet service" else "No")

    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else (multi if multi != "No phone service" else "No"),
        "InternetService": internet,
        "OnlineSecurity": addon(security), "OnlineBackup": addon(backup),
        "DeviceProtection": addon(protection), "TechSupport": addon(tech),
        "StreamingTV": addon(tv), "StreamingMovies": addon(movies),
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])

    prob = float(pipeline.predict_proba(row)[0, 1])
    at_risk = prob >= threshold

    st.divider()
    g, f = st.columns([1, 1.4])
    with g:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(color=NAVY)),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=CHURN if at_risk else STAY),
                       steps=[dict(range=[0, threshold * 100], color="#E8F1FA"),
                              dict(range=[threshold * 100, 100], color="#FBE5DE")],
                       threshold=dict(line=dict(color=NAVY, width=4), value=threshold * 100))))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          title=dict(text="<b>Churn probability</b>", x=0.5, font=dict(size=15, color=NAVY)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if at_risk:
            st.warning(f"⚠️ **Likely to churn.** Flag for the retention team (threshold {threshold:.3f}).")
        else:
            st.success(f"✅ **Likely to stay.** Below the churn threshold ({threshold:.3f}).")

    with f:
        names = pipeline.named_steps["pre"].get_feature_names_out()
        coefs = pipeline.named_steps["clf"].coef_[0]
        x = pipeline.named_steps["pre"].transform(row)
        x = x.toarray()[0] if hasattr(x, "toarray") else x[0]
        contrib = pd.Series(coefs * x, index=names)
        contrib = contrib[contrib != 0]
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        fig = go.Figure(go.Bar(
            x=top.values, y=[n.split("__", 1)[1] for n in top.index], orientation="h",
            marker_color=[CHURN if v > 0 else STAY for v in top.values]))
        fig.update_layout(xaxis_title="Contribution to log-odds (red raises risk, blue lowers it)", yaxis_title="")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style(fig, "Top factors moving this prediction", height=330, legend=False),
                        width="stretch", config={"displayModeBar": False})


# ── Page ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📉 Churn Analytics")
    st.caption("End-to-end ML project on the Telco Customer Churn dataset (7,043 customers).")
    st.markdown(
        "**Model:** Logistic Regression (class-weighted)  \n"
        "**Pipeline:** one-hot + scaling in a scikit-learn `Pipeline`  \n"
        "**Validation:** 5-fold stratified CV  \n"
        "**Threshold:** tuned on the PR curve (best F1)"
    )
    st.divider()
    st.caption("Stack: Python · pandas · scikit-learn · XGBoost · Plotly · Streamlit")

st.markdown(
    '<div class="hero"><h1>Customer Churn Analytics</h1>'
    "<p>Who is likely to leave, why, and how much revenue is at stake. Scored by a class-weighted "
    "Logistic Regression on a held-out test set.</p></div>",
    unsafe_allow_html=True,
)

if not PRED_PATH.exists():
    st.error("`data/churn_predictions.csv` not found. Run `notebook/ML_2.ipynb` first.")
    st.stop()

data = load_predictions()
tab_dash, tab_model, tab_predict = st.tabs(["📊 Dashboard", "🧠 Model performance", "🔮 Predict a customer"])
with tab_dash:
    render_dashboard(data)
with tab_model:
    render_model(data)
with tab_predict:
    render_predictor()