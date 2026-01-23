# Customer Churn Prediction – Task 2 
## Project Overview
This repository contains **Task 2 of my Machine Learning Internship at Future Interns**, where I developed an **end-to-end Customer Churn Prediction System**.  
The goal was to **predict which customers are likely to churn** and generate **business insights** to help retention strategies in telecom, SaaS, and subscription-based services.

The project includes **data exploration, feature engineering, model training, evaluation, and visualization** using real-world churn data.

---

## Tech Stack
- **Python** – primary programming language  
- **Pandas, NumPy** – data manipulation and analysis  
- **Scikit-learn** – preprocessing, model evaluation  
- **XGBoost** – advanced classification model  
- **Matplotlib, Seaborn** – data visualization  
- **Power BI** – interactive dashboard and KPI visualization  

---


---

## Dataset

- The main dataset used is the **Telco Customer Churn Dataset** from Kaggle.  
- It contains customer information such as tenure, contract type, payment method, service usage, and the target column `Churn` (0 = retained, 1 = churned).  
- All datasets are stored under `data/`.

---

## Notebooks

**ML_2.ipynb** – Analyzes distributions, missing values, correlations, and initial insights.  Handles data preprocessing, encoding categorical variables, and creating additional features. Trains Logistic Regression, Random Forest, and XGBoost models, and evaluates them using accuracy, precision, recall, F1-score, and ROC-AUC curves. Generates visual insights such as feature importance, top churn drivers, and churn probability distributions.

---

## Dashboard

- **PowerBI_Dashboard.pbix** – Interactive dashboard with:
  - KPI cards: Total customers, churned customers, churn rate
  - Attribute-wise churn analysis (gender, contract, payment method, internet service)
  - Churn probability visualization and distribution

---

## Outputs

- **figures/** – Exported visualizations (PNG) for presentation

---

## Key Learnings

- Understanding how to **preprocess real-world data**, handle categorical variables, and perform feature engineering  
- Comparing **classification models** for predictive accuracy  
- Evaluating models using **confusion matrix, ROC-AUC, and feature importance**  
- Translating model outputs into **business insights** and **interactive dashboards**

---

## How to Use

1. Open the notebook in `notebook/` for step-by-step analysis.  
2. Open the Power BI file (`powerbi/`) to interact with KPIs and churn visualizations.  

---

## Requirements

To run the Python notebooks:

```bash
pip install -r requirements.txt


