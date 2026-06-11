# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# SHAP Explainable AI Analysis
# ==========================================================

"""
PURPOSE:

This file explains WHY the model predicts a customer
will default.

Outputs:
1. Global Feature Importance
2. SHAP Summary Plot
3. Waterfall Plot (Single Customer)
4. Dependence Plot
5. Top Risk Drivers CSV
6. Business Insights Report

"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs("shap_outputs", exist_ok=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = joblib.load(
    "models/xgboost_model.pkl"
)

print("Model Loaded Successfully")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "lending_dataset_segmented.csv"
)

print("\nDataset Loaded")
print("Shape:", df.shape)

# ==========================================================
# REMOVE NON-FEATURE COLUMNS
# ==========================================================

drop_columns = [

    "Customer_ID",
    "Loan_ID",
    "Loan_Status",

    # target
    "Default",

    # leakage columns
    "Default_Probability",
    "Expected_Loss",
    "Profit"

]

df = df.drop(
    columns=drop_columns,
    errors="ignore"
)

# ==========================================================
# ENCODE CATEGORICAL FEATURES
# ==========================================================

encoders = joblib.load(
    "models/label_encoder.pkl"
)

for col, encoder in encoders.items():

    if col in df.columns:

        df[col] = encoder.transform(
            df[col].astype(str)
        )

# ==========================================================
# SAMPLE DATA
# ==========================================================

# SHAP on full dataset can be slow.
# Use sample for faster execution.

sample_size = min(2000, len(df))

X = df.sample(
    sample_size,
    random_state=42
)

print("\nUsing Sample Size:", len(X))

# ==========================================================
# CREATE SHAP EXPLAINER
# ==========================================================

print("\nCreating SHAP Explainer...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(X)

print("SHAP Values Generated")

# ==========================================================
# GLOBAL FEATURE IMPORTANCE
# ==========================================================

print("\nGenerating Feature Importance...")

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)

importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Mean_SHAP_Value": mean_abs_shap

})

importance_df = importance_df.sort_values(

    by="Mean_SHAP_Value",

    ascending=False

)

importance_df.to_csv(

    "shap_outputs/top_risk_drivers.csv",

    index=False
)

print("\nTop Features")

print(
    importance_df.head(10)
)

# ==========================================================
# SHAP BAR PLOT
# ==========================================================

plt.figure()

shap.summary_plot(

    shap_values,

    X,

    plot_type="bar",

    show=False

)

plt.tight_layout()

plt.savefig(

    "shap_outputs/shap_bar.png",

    bbox_inches="tight"

)

plt.close()

print("Saved shap_bar.png")

# ==========================================================
# SHAP SUMMARY PLOT
# ==========================================================

plt.figure()

shap.summary_plot(

    shap_values,

    X,

    show=False

)

plt.tight_layout()

plt.savefig(

    "shap_outputs/shap_summary.png",

    bbox_inches="tight"

)

plt.close()

print("Saved shap_summary.png")

# ==========================================================
# WATERFALL PLOT
# ==========================================================

print("\nGenerating Waterfall Plot")

customer_index = 0

explanation = shap.Explanation(

    values=shap_values[customer_index],

    base_values=explainer.expected_value,

    data=X.iloc[customer_index],

    feature_names=X.columns

)

plt.figure()

shap.plots.waterfall(

    explanation,

    max_display=10,

    show=False

)

plt.savefig(

    "shap_outputs/waterfall_plot.png",

    bbox_inches="tight"

)

plt.close()

print("Saved waterfall_plot.png")

# ==========================================================
# DEPENDENCE PLOT
# ==========================================================

top_feature = importance_df.iloc[0]["Feature"]

print(
    f"\nCreating Dependence Plot for: {top_feature}"
)

shap.dependence_plot(

    top_feature,

    shap_values,

    X,

    show=False

)

plt.savefig(

    "shap_outputs/dependence_plot.png",

    bbox_inches="tight"

)

plt.close()

print("Saved dependence_plot.png")

# ==========================================================
# BUSINESS INSIGHTS REPORT
# ==========================================================

print("\nGenerating Business Report")

top10 = importance_df.head(10)

with open(

    "shap_outputs/business_insights.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "AI DIGITAL LENDING PORTFOLIO OPTIMIZATION\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "TOP RISK DRIVERS IDENTIFIED BY SHAP\n\n"
    )

    for i, row in enumerate(
        top10.itertuples(),
        start=1
    ):

        f.write(
            f"{i}. {row.Feature}"
            f" (SHAP={row.Mean_SHAP_Value:.4f})\n"
        )

    f.write("\n")

    f.write(
        "BUSINESS INTERPRETATION\n"
    )

    f.write(
        "-----------------------\n"
    )

    f.write(
        "Customers with low credit scores,\n"
    )

    f.write(
        "high missed payments and unstable\n"
    )

    f.write(
        "financial behavior contribute most\n"
    )

    f.write(
        "to default risk.\n\n"
    )

    f.write(
        "Recommendation:\n"
    )

    f.write(
        "- Tighten underwriting for high-risk customers\n"
    )

    f.write(
        "- Monitor customers with increasing missed payments\n"
    )

    f.write(
        "- Introduce proactive intervention programs\n"
    )

    f.write(
        "- Reprice loans according to risk score\n"
    )

print(
    "Saved business_insights.txt"
)

# ==========================================================
# CUSTOMER LEVEL EXPLANATION
# ==========================================================

print("\nExample Customer Explanation")

top_customer_features = pd.DataFrame({

    "Feature": X.columns,

    "Impact": shap_values[0]

})

top_customer_features = top_customer_features.sort_values(

    by="Impact",

    ascending=False

)

top_customer_features.to_csv(

    "shap_outputs/customer_explanation.csv",

    index=False
)

print(
    top_customer_features.head(10)
)

# ==========================================================
# COMPLETION
# ==========================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED")
print("=" * 60)

print("\nOutputs Saved In:")

print("shap_outputs/")