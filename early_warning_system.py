# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Early Warning System
# ==========================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs("warning_outputs", exist_ok=True)

# ==========================================================
# LOAD RISK DATASET
# ==========================================================

df = pd.read_csv(
    "risk_outputs/customer_risk_scores.csv"
)

print("=" * 60)
print("EARLY WARNING SYSTEM")
print("=" * 60)

# ==========================================================
# WARNING SCORE
# ==========================================================

warning_score = np.zeros(len(df))

# ==========================================================
# RULE 1
# MISSED PAYMENTS
# ==========================================================

warning_score += np.where(
    df["Missed_Payments"] >= 3,
    25,
    0
)

# ==========================================================
# RULE 2
# DAYS PAST DUE
# ==========================================================

warning_score += np.where(
    df["Days_Past_Due"] >= 15,
    25,
    0
)

# ==========================================================
# RULE 3
# BALANCE VOLATILITY
# ==========================================================

warning_score += np.where(
    df["Balance_Volatility"] > 0.7,
    20,
    0
)

# ==========================================================
# RULE 4
# SPENDING SHOCK
# ==========================================================

warning_score += np.where(
    df["Spending_Shock"] == 1,
    15,
    0
)

# ==========================================================
# RULE 5
# MODEL RISK SCORE
# ==========================================================

warning_score += np.where(
    df["Risk_Score"] > 75,
    30,
    np.where(
        df["Risk_Score"] > 50,
        15,
        0
    )
)

# ==========================================================
# SAVE WARNING SCORE
# ==========================================================

df["Warning_Score"] = warning_score

# ==========================================================
# WARNING LEVELS
# ==========================================================

def warning_level(score):

    if score < 25:
        return "Low Alert"

    elif score < 50:
        return "Medium Alert"

    elif score < 75:
        return "High Alert"

    else:
        return "Critical Alert"

df["Warning_Level"] = (
    df["Warning_Score"]
    .apply(warning_level)
)

# ==========================================================
# WARNING ANALYSIS
# ==========================================================

warning_analysis = (
    df["Warning_Level"]
    .value_counts()
)

warning_analysis.to_csv(
    "warning_outputs/warning_distribution.csv"
)

# ==========================================================
# HIGH ALERT CUSTOMERS
# ==========================================================

alerts = df[

    df["Warning_Level"].isin(

        [
            "High Alert",
            "Critical Alert"
        ]
    )
]

alerts.to_csv(

    "warning_outputs/high_alert_customers.csv",

    index=False
)

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(

    x=df["Warning_Level"]

)

plt.title(
    "Warning Level Distribution"
)

plt.xticks(rotation=20)

plt.savefig(
    "warning_outputs/warning_distribution.png"
)

plt.close()

# ==========================================================
# SAVE FINAL DATASET
# ==========================================================

df.to_csv(

    "warning_outputs/early_warning_dataset.csv",

    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\nWarning Levels")

print(
    df["Warning_Level"]
    .value_counts()
)

print("\nHigh Risk Customers")

print(
    len(alerts)
)

print(
    "\nOutputs Saved To warning_outputs/"
)