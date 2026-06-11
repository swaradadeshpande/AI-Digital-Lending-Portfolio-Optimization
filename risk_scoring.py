# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Risk Scoring Engine
# ==========================================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CREATE FOLDER
# ==========================================================

os.makedirs("risk_outputs", exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "lending_dataset_segmented.csv"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(
    "models/xgboost_model.pkl"
)

encoders = joblib.load(
    "models/label_encoder.pkl"
)

# ==========================================================
# PREPARE FEATURES
# ==========================================================

drop_cols = [

    "Customer_ID",
    "Loan_ID",

    "Default",

    "Default_Probability",
    "Expected_Loss",
    "Profit",

    "Loan_Status"
]

X = df.drop(
    columns=drop_cols,
    errors="ignore"
)

# ==========================================================
# ENCODE CATEGORICALS
# ==========================================================

for col, encoder in encoders.items():

    if col in X.columns:

        X[col] = encoder.transform(
            X[col].astype(str)
        )

# ==========================================================
# PREDICT PROBABILITY
# ==========================================================

probability = model.predict_proba(X)[:,1]

df["Predicted_Default_Probability"] = probability

# ==========================================================
# CREATE RISK SCORE
# ==========================================================

df["Risk_Score"] = (
    probability * 100
).round(2)

# ==========================================================
# CREATE RISK BANDS
# ==========================================================

def assign_band(score):

    if score < 25:
        return "Low"

    elif score < 50:
        return "Medium"

    elif score < 75:
        return "High"

    else:
        return "Critical"

df["Risk_Band"] = (
    df["Risk_Score"]
    .apply(assign_band)
)

# ==========================================================
# CUSTOMER RANKING
# ==========================================================

df["Risk_Rank"] = (
    df["Risk_Score"]
    .rank(
        ascending=False
    )
)

# ==========================================================
# SAVE MASTER FILE
# ==========================================================

df.to_csv(

    "risk_outputs/customer_risk_scores.csv",

    index=False
)

# ==========================================================
# TOP 100 RISKY CUSTOMERS
# ==========================================================

top100 = df.sort_values(

    by="Risk_Score",

    ascending=False

).head(100)

top100.to_csv(

    "risk_outputs/top_100_risky_customers.csv",

    index=False
)

# ==========================================================
# RISK DISTRIBUTION
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(

    df["Risk_Score"],

    bins=20,

    kde=True
)

plt.title(
    "Portfolio Risk Distribution"
)

plt.savefig(

    "risk_outputs/risk_distribution.png"
)

plt.close()

# ==========================================================
# BAND ANALYSIS
# ==========================================================

band_analysis = df.groupby(
    "Risk_Band"
).agg({

    "Loan_Amount":"mean",
    "Risk_Score":"mean",
    "Default":"mean"

})

band_analysis["Default"] *= 100

band_analysis.to_csv(

    "risk_outputs/risk_band_analysis.csv"
)

# ==========================================================
# POWER BI DATASET
# ==========================================================

df.to_csv(

    "risk_outputs/powerbi_risk_dataset.csv",

    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("="*60)
print("RISK SCORING COMPLETED")
print("="*60)

print("\nAverage Risk Score")

print(
    round(
        df["Risk_Score"].mean(),
        2
    )
)

print("\nRisk Bands")

print(
    df["Risk_Band"]
    .value_counts()
)

print(
    "\nOutputs Saved To risk_outputs/"
)