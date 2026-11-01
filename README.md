# Customer Churn Prediction – Telco Dataset

End-to-end churn prediction project: data cleaning, EDA, imbalance-aware modelling, cross-validated model comparison, decision-threshold tuning, explainability, and an interactive Streamlit dashboard for business stakeholders.

---

## Problem

A telecom company loses revenue when customers leave. The goal is to **rank customers by churn risk** so the retention team can contact the most at-risk customers first.

- **Dataset:** [Telco Customer Churn (Kaggle / IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) – 7,043 customers, 20 features
- **Target:** `Churn` (1 = churned, 0 = retained)
- **Class balance:** 26.5% churned, so a model that predicts "no churn" for everyone already scores **73.5% accuracy**. Accuracy alone is therefore not used to judge models.

---

## Approach

| Step | What was done | Why |
|---|---|---|
| Cleaning | 11 blank `TotalCharges` values set to 0 (all are `tenure = 0` customers); `customerID` dropped from features | Keeps all rows instead of dropping them |
| Encoding | `OneHotEncoder` for categorical features, `StandardScaler` for numeric ones | Avoids the fake ordering that label encoding gives nominal features |
| Pipeline | Preprocessing + model in a single scikit-learn `Pipeline` | Prevents data leakage during cross-validation |
| Imbalance | `class_weight='balanced'` (LR, RF) and `scale_pos_weight` (XGBoost) | Stops models from ignoring the minority (churn) class |
| Oversampling check | SMOTENC inside an `imblearn` pipeline (synthetic rows created from training folds only), compared against class weighting | Tests whether synthetic oversampling beats class weighting; it did not, so class weighting is kept |
| Validation | 5-fold stratified cross-validation | The models score very similarly, so one split cannot separate them |
| Metrics | ROC-AUC, PR-AUC, recall, precision, F1 | More informative than accuracy on imbalanced data |
| Threshold | Best-F1 threshold chosen from **out-of-fold predictions on the training data**, then frozen and evaluated once on the test set | The default 0.5 cutoff gives up recall, and choosing the cutoff on training data keeps the test set untouched |
| Explainability | Logistic Regression coefficients, XGBoost feature importance, and **SHAP** (`LinearExplainer`): global summary plot and a per-customer waterfall | Shows which factors drive churn overall and why an individual customer is flagged |
| Business view | Top-decile lift table, revenue-at-risk and a retention call list | Shows how well the model prioritises customers and what it is worth |

---

## Results

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.845 | 0.655 | 0.803 | 0.513 | 0.626 |
| Random Forest | 0.823 | 0.602 | 0.642 | 0.560 | 0.598 |
| XGBoost | 0.848 | 0.666 | 0.796 | 0.521 | 0.630 |

**Held-out test set (20%), Logistic Regression at the frozen threshold (0.576, chosen from training out-of-fold predictions):**

| Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|
| 0.762 | 0.539 | 0.725 | 0.618 |

**Business view:** the 10% of customers with the highest predicted risk contain **28.6%** of all churners (a 75.9% churn rate in that group vs 26.5% overall).

**Key takeaways**
- Logistic Regression and XGBoost are almost tied on ROC-AUC, so the simpler and more interpretable model is used as the final model.
- Class weighting plus threshold tuning lowers accuracy compared with an unweighted Logistic Regression (about 81% to about 76%) but catches roughly 17 percentage points more churners (recall about 56% to about 73%). A missed churner costs more than an unnecessary retention offer.
- SMOTENC was tested against class weighting under the same 5-fold CV. For Logistic Regression it gave ROC-AUC 0.836 vs 0.845 and F1 0.623 vs 0.626 (recall 0.744 vs 0.803), so class weighting is kept: it is simpler and avoids generating synthetic customers.
- The threshold is selected without looking at the test set (out-of-fold predictions on the training data), so the test metrics above are an honest out-of-sample estimate.
- Class weighting inflates the raw scores (mean score 0.42 vs a 26.5% true churn rate), so the output is a **risk score for ranking**, not a calibrated probability.
- The strongest churn signals are contract type, tenure, internet service type, and payment method (see the feature plots in the notebook).
- SHAP confirms the main drivers and explains individual customers: month-to-month contracts and fiber optic internet raise churn risk, while long tenure and two-year contracts lower it. `tenure`, `MonthlyCharges` and `TotalCharges` are strongly correlated, so their individual signs should be read together rather than one at a time.

---

## Repository Structure

```
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw dataset
│   ├── churn_predictions.csv                  # test-set predictions (written by the notebook)
│   └── cv_results.csv                         # cross-validation table (written by the notebook)
├── app.py                                     # Streamlit app: dashboard, model performance, live predictor
├── .streamlit/config.toml                     # app theme
├── notebook/
│   ├── ML_2.ipynb                             # full analysis and modelling
│   └── churn_model.joblib                     # saved pipeline + threshold (written by the notebook)
├── requirements.txt
└── README.md
```

---

## Dashboard (Streamlit)

`streamlit run app.py` opens a three-tab app. All numbers are computed from the **20% held-out test set** (1,409 customers), so they are out-of-sample.

**📊 Dashboard**
- Slicers: contract, internet service, payment method, tenure range, and an adjustable high-risk cutoff
- KPI cards: customers analysed, churn rate, average churn probability, high-risk customers (and how many actually churned), monthly revenue at risk
- Churn vs non-churn split; churn by internet service; churn probability by contract and payment method; churn risk by tenure (actual vs predicted); churn probability distribution
- Retention call list of high-risk customers with CSV download

**🧠 Model performance:** ROC curve, confusion matrix, risk-decile lift, 5-fold CV model comparison, default vs tuned threshold, and the top churn drivers.

**🔮 Predict a customer:** enter a customer's details to get a churn probability gauge, the flag at the tuned threshold, and the features moving the prediction.

---

## How to Run

```bash
pip install -r requirements.txt
cd notebook
jupyter notebook ML_2.ipynb      # run all cells
cd ..
streamlit run app.py
```

Run the notebook from the `notebook/` folder; it reads data from `../data/` and writes `data/churn_predictions.csv`, `data/cv_results.csv` and `notebook/churn_model.joblib`, which the app reads. Re-run the notebook whenever the model changes and the dashboard updates with it.

**Scoring new customers in code:**

```python
import joblib
bundle = joblib.load("notebook/churn_model.joblib")
prob = bundle["pipeline"].predict_proba(new_customers_df)[:, 1]
at_risk = (prob >= bundle["threshold"]).astype(int)
```

`new_customers_df` must have the same columns as the training features (all columns except `customerID` and `Churn`).

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · SHAP · Matplotlib · Seaborn · Plotly · joblib · Streamlit

---

## Limitations and Next Steps

- The threshold is chosen from out-of-fold predictions on the training data. A separate validation split is an alternative, and a cost-based rule would be better still (see below).
- Calibrate the probabilities (for example `CalibratedClassifierCV`) so scores can be read as true probabilities.
- No hyperparameter search yet (`RandomizedSearchCV` / Optuna).
- No engineered features yet (for example tenure groups, number of add-on services, average monthly spend).
- Add SHAP to the Streamlit app (it currently lives in the notebook).
- Replace the F1-based threshold with a cost-based one (retention offer cost vs customer lifetime value).
- Host the Streamlit demo publicly (for example Streamlit Community Cloud) or wrap the pipeline in an API.