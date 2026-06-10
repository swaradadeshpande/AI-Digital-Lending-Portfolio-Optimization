# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Exploratory Data Analysis (EDA)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use("ggplot")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("data/lending_dataset.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

import os

if not os.path.exists("eda_outputs"):
    os.makedirs("eda_outputs")

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

print("\nSummary Statistics")
print(df.describe())

df.describe().to_csv(
    "eda_outputs/summary_statistics.csv"
)

# ==========================================================
# DEFAULT RATE
# ==========================================================

default_rate = round(
    df["Default"].mean() * 100,
    2
)

print(f"\nOverall Default Rate: {default_rate}%")

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

numeric_cols = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    numeric_cols.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "eda_outputs/correlation_heatmap.png"
)

plt.show()

# ==========================================================
# LOAN TYPE DEFAULT ANALYSIS
# ==========================================================

loan_default = df.groupby(
    "Loan_Type"
)["Default"].mean() * 100

plt.figure(figsize=(8, 5))

loan_default.sort_values().plot(
    kind="barh"
)

plt.title("Default Rate by Loan Type")
plt.xlabel("Default Rate (%)")

plt.tight_layout()

plt.savefig(
    "eda_outputs/default_by_loan_type.png"
)

plt.show()

print("\nDefault Rate By Loan Type")
print(
    loan_default.sort_values(
        ascending=False
    )
)

# ==========================================================
# PROFIT BY LOAN TYPE
# ==========================================================

profit_loan = df.groupby(
    "Loan_Type"
)["Profit"].mean()

plt.figure(figsize=(8, 5))

profit_loan.sort_values().plot(
    kind="barh"
)

plt.title(
    "Average Profit by Loan Type"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/profit_by_loan_type.png"
)

plt.show()

# ==========================================================
# CREDIT SCORE VS DEFAULT
# ==========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="Default",
    y="Credit_Score"
)

plt.title(
    "Credit Score vs Default"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/credit_score_default.png"
)

plt.show()

# ==========================================================
# INCOME VS DEFAULT
# ==========================================================

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=df,
    x="Default",
    y="Income"
)

plt.title(
    "Income Distribution by Default"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/income_default.png"
)

plt.show()

# ==========================================================
# RISK GRADE ANALYSIS
# ==========================================================

risk_default = df.groupby(
    "Risk_Grade"
)["Default"].mean() * 100

plt.figure(figsize=(8, 5))

sns.barplot(
    x=risk_default.index,
    y=risk_default.values
)

plt.title(
    "Default Rate by Risk Grade"
)

plt.ylabel(
    "Default Rate (%)"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/risk_grade_default.png"
)

plt.show()

# ==========================================================
# STATE LEVEL DEFAULT ANALYSIS
# ==========================================================

state_default = df.groupby(
    "State"
)["Default"].mean() * 100

plt.figure(figsize=(10, 5))

state_default.sort_values().plot(
    kind="bar"
)

plt.title(
    "State Wise Default Rate"
)

plt.ylabel(
    "Default Rate (%)"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/state_default.png"
)

plt.show()

# ==========================================================
# ACQUISITION CHANNEL ANALYSIS
# ==========================================================

channel_analysis = df.groupby(
    "Acquisition_Channel"
).agg(
    {
        "Customer_LTV": "mean",
        "Profit": "mean",
        "Default": "mean"
    }
)

channel_analysis["Default"] *= 100

print("\nChannel Analysis")
print(channel_analysis)

channel_analysis.to_csv(
    "eda_outputs/channel_analysis.csv"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=channel_analysis.index,
    y=channel_analysis["Profit"]
)

plt.xticks(rotation=20)

plt.title(
    "Profit by Acquisition Channel"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/channel_profit.png"
)

plt.show()

# ==========================================================
# TOP LOSS MAKING CUSTOMERS
# ==========================================================

loss_customers = df.sort_values(
    by="Expected_Loss",
    ascending=False
).head(20)

loss_customers.to_csv(
    "eda_outputs/top_loss_customers.csv",
    index=False
)

# ==========================================================
# TREEMAP
# ==========================================================

treemap = px.treemap(
    df,
    path=[
        "Loan_Type",
        "Risk_Grade"
    ],
    values="Loan_Amount",
    color="Profit",
    title="Loan Portfolio Treemap"
)

treemap.write_html(
    "eda_outputs/loan_treemap.html"
)

# ==========================================================
# SUNBURST CHART
# ==========================================================

sunburst = px.sunburst(
    df,
    path=[
        "Acquisition_Channel",
        "Loan_Type"
    ],
    values="Loan_Amount",
    color="Default"
)

sunburst.write_html(
    "eda_outputs/sunburst_chart.html"
)

# ==========================================================
# DEFAULT RATE BY CREDIT SCORE BIN
# ==========================================================

df["Credit_Bin"] = pd.cut(
    df["Credit_Score"],
    bins=[300, 500, 600, 700, 800, 900]
)

credit_default = df.groupby(
    "Credit_Bin"
)["Default"].mean() * 100

plt.figure(figsize=(10, 5))

credit_default.plot(
    kind="bar"
)

plt.title(
    "Default Rate by Credit Score Range"
)

plt.ylabel(
    "Default Rate (%)"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/credit_bin_default.png"
)

plt.show()

# ==========================================================
# MISSED PAYMENTS IMPACT
# ==========================================================

missed_default = df.groupby(
    "Missed_Payments"
)["Default"].mean() * 100

plt.figure(figsize=(10, 5))

missed_default.plot(
    marker="o"
)

plt.title(
    "Missed Payments vs Default Rate"
)

plt.ylabel(
    "Default Rate (%)"
)

plt.tight_layout()

plt.savefig(
    "eda_outputs/missed_payment_default.png"
)

plt.show()

# ==========================================================
# PROFITABILITY DASHBOARD TABLE
# ==========================================================

dashboard_table = df.groupby(
    "Loan_Type"
).agg(
    {
        "Loan_Amount": "sum",
        "Profit": "sum",
        "Expected_Loss": "sum",
        "Default": "mean"
    }
)

dashboard_table["Default"] *= 100

dashboard_table.to_csv(
    "eda_outputs/dashboard_table.csv"
)

# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

highest_default_loan = (
    loan_default.idxmax()
)

print(
    f"\nHighest Default Loan Type: "
    f"{highest_default_loan}"
)

best_channel = (
    channel_analysis["Profit"]
    .idxmax()
)

print(
    f"\nBest Acquisition Channel: "
    f"{best_channel}"
)

highest_profit_loan = (
    profit_loan.idxmax()
)

print(
    f"\nMost Profitable Loan Type: "
    f"{highest_profit_loan}"
)

print(
    "\nEDA Completed Successfully"
)

print(
    "\nOutputs saved in eda_outputs/"
)