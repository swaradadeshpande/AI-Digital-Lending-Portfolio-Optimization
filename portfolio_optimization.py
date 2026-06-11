# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Portfolio Optimization & Business Intelligence
# ==========================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs("portfolio_outputs", exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "warning_outputs/early_warning_dataset.csv"
)

print("=" * 60)
print("PORTFOLIO OPTIMIZATION")
print("=" * 60)

# ==========================================================
# EXPECTED LOSS
# ==========================================================

df["Expected_Loss_Model"] = (
    df["Loan_Amount"]
    * (df["Risk_Score"] / 100)
)

# ==========================================================
# EXPECTED INTEREST INCOME
# ==========================================================

df["Expected_Interest_Income"] = (

    df["Loan_Amount"]

    * (df["Interest_Rate"] / 100)

)

# ==========================================================
# RISK ADJUSTED PROFIT
# ==========================================================

df["Risk_Adjusted_Profit"] = (

    df["Expected_Interest_Income"]

    - df["Expected_Loss_Model"]

)

# ==========================================================
# PORTFOLIO SUMMARY
# ==========================================================

portfolio_summary = {

    "Total_Customers":
        len(df),

    "Total_Loan_Amount":
        round(
            df["Loan_Amount"].sum(),
            2
        ),

    "Average_Risk_Score":
        round(
            df["Risk_Score"].mean(),
            2
        ),

    "Portfolio_Default_Rate":
        round(
            df["Default"].mean() * 100,
            2
        ),

    "Expected_Portfolio_Loss":
        round(
            df["Expected_Loss_Model"].sum(),
            2
        ),

    "Expected_Portfolio_Profit":
        round(
            df["Risk_Adjusted_Profit"].sum(),
            2
        )
}

portfolio_df = pd.DataFrame(

    portfolio_summary.items(),

    columns=[
        "Metric",
        "Value"
    ]
)

portfolio_df.to_csv(

    "portfolio_outputs/portfolio_summary.csv",

    index=False
)

# ==========================================================
# RISK BAND ANALYSIS
# ==========================================================

risk_band_analysis = (

    df.groupby(
        "Risk_Band"
    )

    .agg({

        "Loan_Amount":"sum",

        "Risk_Score":"mean",

        "Default":"mean",

        "Risk_Adjusted_Profit":"sum"

    })

)

risk_band_analysis["Default"] *= 100

risk_band_analysis.to_csv(

    "portfolio_outputs/risk_band_analysis.csv"
)

# ==========================================================
# WARNING ANALYSIS
# ==========================================================

warning_analysis = (

    df.groupby(
        "Warning_Level"
    )

    .agg({

        "Loan_Amount":"sum",

        "Default":"mean",

        "Risk_Adjusted_Profit":"sum"

    })

)

warning_analysis["Default"] *= 100

warning_analysis.to_csv(

    "portfolio_outputs/warning_level_analysis.csv"
)

# ==========================================================
# SEGMENT ANALYSIS
# ==========================================================

segment_analysis = (

    df.groupby(
        "Risk_Segment"
    )

    .agg({

        "Loan_Amount":"sum",

        "Default":"mean",

        "Risk_Adjusted_Profit":"sum"

    })

)

segment_analysis["Default"] *= 100

segment_analysis.to_csv(

    "portfolio_outputs/segment_analysis.csv"
)

# ==========================================================
# VISUALIZATION 1
# ==========================================================

plt.figure(figsize=(10,6))

sns.barplot(

    x=risk_band_analysis.index,

    y=risk_band_analysis[
        "Risk_Adjusted_Profit"
    ]

)

plt.title(
    "Profit by Risk Band"
)

plt.savefig(

    "portfolio_outputs/profit_by_risk_band.png"
)

plt.close()

# ==========================================================
# VISUALIZATION 2
# ==========================================================

plt.figure(figsize=(10,6))

sns.barplot(

    x=risk_band_analysis.index,

    y=risk_band_analysis[
        "Default"
    ]

)

plt.title(
    "Default Rate by Risk Band"
)

plt.savefig(

    "portfolio_outputs/default_rate_by_risk_band.png"
)

plt.close()

# ==========================================================
# BUSINESS RECOMMENDATIONS
# ==========================================================

with open(

    "portfolio_outputs/business_recommendations.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "PORTFOLIO OPTIMIZATION RECOMMENDATIONS\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        "1. Reduce exposure to Critical Risk customers.\n"
    )

    f.write(
        "2. Monitor High Alert accounts proactively.\n"
    )

    f.write(
        "3. Increase lending to Low Risk customers.\n"
    )

    f.write(
        "4. Use risk-based pricing strategy.\n"
    )

    f.write(
        "5. Strengthen collection efforts for High Risk segments.\n"
    )

# ==========================================================
# SAVE POWER BI DATASET
# ==========================================================

df.to_csv(

    "portfolio_outputs/powerbi_portfolio_dataset.csv",

    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\nPortfolio Summary\n")

for k, v in portfolio_summary.items():

    print(f"{k}: {v}")

print(
    "\nOutputs Saved To portfolio_outputs/"
)