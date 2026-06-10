# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Customer Risk Segmentation
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

import os

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

if not os.path.exists("segmentation_outputs"):
    os.makedirs("segmentation_outputs")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("data/lending_dataset.csv")

print("=" * 60)
print("CUSTOMER SEGMENTATION")
print("=" * 60)

# ==========================================================
# FEATURE SELECTION
# ==========================================================

features = [

    "Income",
    "Credit_Score",
    "Loan_Amount",
    "Monthly_Cashflow",
    "Balance_Volatility",
    "Missed_Payments"

]

X = df[features]

# ==========================================================
# SCALING
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# ELBOW METHOD
# ==========================================================

wcss = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.savefig(
    "segmentation_outputs/elbow_curve.png"
)

plt.show()

# ==========================================================
# FINAL MODEL
# ==========================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ==========================================================
# CLUSTER PROFILE
# ==========================================================

cluster_profile = df.groupby(
    "Cluster"
)[features].mean()

print("\nCluster Profile:")
print(cluster_profile)

cluster_profile.to_csv(
    "segmentation_outputs/cluster_profile.csv"
)

# ==========================================================
# RISK LABELS
# ==========================================================

risk_order = cluster_profile[
    "Missed_Payments"
].sort_values().index.tolist()

risk_mapping = {
    risk_order[0]: "Low Risk",
    risk_order[1]: "Medium Risk",
    risk_order[2]: "High Risk",
    risk_order[3]: "Critical Risk"
}

df["Risk_Segment"] = (
    df["Cluster"]
    .map(risk_mapping)
)

# ==========================================================
# SEGMENT COUNTS
# ==========================================================

segment_counts = (
    df["Risk_Segment"]
    .value_counts()
)

print("\nSegment Counts:")
print(segment_counts)

segment_counts.to_csv(
    "segmentation_outputs/segment_counts.csv"
)

# ==========================================================
# PCA VISUALIZATION
# ==========================================================

pca = PCA(n_components=2)

pca_features = pca.fit_transform(
    X_scaled
)

pca_df = pd.DataFrame()

pca_df["PCA1"] = pca_features[:,0]
pca_df["PCA2"] = pca_features[:,1]

pca_df["Risk_Segment"] = (
    df["Risk_Segment"]
)

plt.figure(figsize=(10,7))

sns.scatterplot(

    data=pca_df,

    x="PCA1",
    y="PCA2",

    hue="Risk_Segment",

    alpha=0.7

)

plt.title(
    "Customer Risk Segmentation (PCA)"
)

plt.savefig(
    "segmentation_outputs/pca_clusters.png"
)

plt.show()

# ==========================================================
# SEGMENT DEFAULT RATE
# ==========================================================

default_analysis = df.groupby(
    "Risk_Segment"
)["Default"].mean() * 100

print("\nDefault Rate by Segment")
print(default_analysis)

default_analysis.to_csv(
    "segmentation_outputs/default_rate_by_segment.csv"
)

# ==========================================================
# SEGMENT PROFITABILITY
# ==========================================================

profit_analysis = df.groupby(
    "Risk_Segment"
)["Profit"].mean()

print("\nProfit by Segment")
print(profit_analysis)

profit_analysis.to_csv(
    "segmentation_outputs/profit_by_segment.csv"
)

# ==========================================================
# BAR CHART
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(

    x=default_analysis.index,
    y=default_analysis.values

)

plt.title(
    "Default Rate by Risk Segment"
)

plt.ylabel(
    "Default Rate (%)"
)

plt.savefig(
    "segmentation_outputs/default_segment.png"
)

plt.show()

# ==========================================================
# SAVE DATASET
# ==========================================================

df.to_csv(
    "lending_dataset_segmented.csv",
    index=False
)

print("\nDataset Saved:")
print("lending_dataset_segmented.csv")

print("\nSegmentation Complete")