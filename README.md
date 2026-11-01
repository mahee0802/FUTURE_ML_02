# Customer Churn Prediction – Telco Dataset

End-to-end churn prediction project: data cleaning, EDA, imbalance-aware modelling, cross-validated model comparison, decision-threshold tuning, explainability, and a Power BI dashboard for business stakeholders.

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
| Validation | 5-fold stratified cross-validation | The models score very similarly, so one split cannot separate them |
| Metrics | ROC-AUC, PR-AUC, recall, precision, F1 | More informative than accuracy on imbalanced data |
| Threshold | Decision threshold tuned on the precision-recall curve (best F1) | The default 0.5 cutoff gives up recall |
| Explainability | Logistic Regression coefficients and XGBoost feature importance, with real feature names | Shows which factors drive churn |
| Business view | Top-decile lift table | Shows how well the model prioritises customers |

---

## Results

**5-fold cross-validation (mean):**

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.845 | 0.655 | 0.803 | 0.513 | 0.626 |
| Random Forest | 0.821 | 0.605 | 0.470 | 0.627 | 0.537 |
| XGBoost | 0.847 | 0.665 | 0.798 | 0.521 | 0.630 |

**Held-out test set (20%), Logistic Regression at the tuned threshold (0.544):**

| Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|
| 0.754 | 0.525 | 0.765 | 0.622 |

**Business view:** the 10% of customers with the highest predicted risk contain **28.6%** of all churners (a 75.9% churn rate in that group vs 26.5% overall).

**Key takeaways**
- Logistic Regression and XGBoost are almost tied on ROC-AUC, so the simpler and more interpretable model is used as the final model.
- Tuning for recall lowers accuracy compared with an untuned model (about 79% to about 75%) but catches roughly 20 percentage points more churners. A missed churner costs more than an unnecessary retention offer.
- The strongest churn signals are contract type, tenure, internet service type, and payment method (see the feature plots in the notebook).

---

## Repository Structure

```
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # raw dataset
│   └── churn_predictions.csv                  # test-set predictions (written by the notebook)
├── app.py                                     # Streamlit demo (loads the saved pipeline)
├── notebook/
│   ├── ML_2.ipynb                             # full analysis and modelling
│   └── churn_model.joblib                     # saved pipeline + threshold (written by the notebook)
├── outputs/                                   # exported charts (PNG)
├── powerbi/
│   └── ML_2.pbix                              # Power BI dashboard
├── requirements.txt
└── README.md
```

---

## Dashboard (Power BI)

`powerbi/ML_2.pbix` contains:
- KPI cards: churn rate, average churn probability, high-risk customers
- Churn vs non-churn split
- Churn by internet service, contract type, payment method, and tenure
- Churn probability distribution

---

## How to Run

```bash
pip install -r requirements.txt
cd notebook
jupyter notebook ML_2.ipynb
```

Run the notebook from the `notebook/` folder; it reads data from `../data/`.

**Demo app (run the notebook first so `notebook/churn_model.joblib` exists):**

```bash
streamlit run app.py
```

Enter a customer's details in the form to get a churn probability, a likely-to-churn flag using the tuned threshold, and the top factors moving the prediction.

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

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · joblib · Streamlit · Power BI

---

## Limitations and Next Steps

- The decision threshold is chosen on the test split. A cleaner approach is to choose it from cross-validated predictions or a separate validation set.
- No hyperparameter search yet (`RandomizedSearchCV` / Optuna).
- No engineered features yet (for example tenure groups, number of add-on services, average monthly spend).
- Add SHAP values for per-customer explanations.
- Replace the F1-based threshold with a cost-based one (retention offer cost vs customer lifetime value).
- Host the Streamlit demo publicly (for example Streamlit Community Cloud) or wrap the pipeline in an API.