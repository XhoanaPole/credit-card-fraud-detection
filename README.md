#  Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent 
credit card transactions, built on the Kaggle European Cardholders dataset.

##  Project Overview
Credit card fraud detection is a classic imbalanced classification 
problem. This project covers the full ML workflow — from raw data 
to a saved, deployment-ready model.

##  Pipeline Steps
1. Data Loading & Exploration
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing & Train/Val/Test Split
5. Handling Class Imbalance (SMOTE, SMOTETomek)
6. Model Training & Comparison
7. Model Evaluation & Visualization
8. Feature Importance Analysis
9. Model Saving (joblib)

## 🤖 Models Compared
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline |
| Decision Tree | Interpretable |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| XGBoost | Boosting |
| LightGBM | Fast boosting |

##  Handling Class Imbalance
- SMOTE (Synthetic Minority Oversampling)
- Random UnderSampler
- SMOTETomek (combined approach)

## Evaluation Metrics
- AUC-ROC
- F1-Score
- Average Precision
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Precision-Recall Curve

## Tools & Libraries
Python, scikit-learn, XGBoost, LightGBM, imbalanced-learn, 
Pandas, NumPy, Matplotlib, Seaborn, joblib

##  Dataset
Kaggle — Credit Card Fraud Detection
284,807 transactions | 492 fraud cases (0.172%)
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 👩‍💻 Author
Xhoana Pole — AI Student, University Metropolitan Tirana
