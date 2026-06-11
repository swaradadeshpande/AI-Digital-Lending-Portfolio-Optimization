# ==========================================================
# AI-Powered Digital Lending Portfolio Optimization
# Default Prediction Module
# ==========================================================

import pandas as pd
import numpy as np
import os
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==========================================================
# CREATE FOLDERS
# ==========================================================

os.makedirs("ml_outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    "lending_dataset_segmented.csv"
)

print("="*60)
print("DEFAULT PREDICTION MODEL")
print("="*60)

# ==========================================================
# DROP IDS
# ==========================================================

drop_cols = [
    "Customer_ID",
    "Loan_ID",
    "Loan_Status"
]

df = df.drop(
    columns=drop_cols,
    errors="ignore"
)

# ==========================================================
# ENCODE CATEGORICAL FEATURES
# ==========================================================

encoders = {}

cat_cols = df.select_dtypes(
    include="object"
).columns

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(
        df[col].astype(str)
    )

    encoders[col] = le

joblib.dump(
    encoders,
    "models/label_encoder.pkl"
)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

leakage_cols = [

    "Default",

    "Default_Probability",

    "Expected_Loss",

    "Profit",

    "Loan_Status"

]

X = df.drop(
    columns=leakage_cols,
    errors="ignore"
)

y = df["Default"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==========================================================
# MODELS
# ==========================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=2000
        ),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "XGBoost":

        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss"
        )
}

results = []

best_model = None
best_auc = 0

# ==========================================================
# TRAINING LOOP
# ==========================================================

for name, model in models.items():

    print("\n" + "="*40)
    print(name)
    print("="*40)

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    probs = model.predict_proba(
        X_test
    )[:,1]

    acc = accuracy_score(
        y_test,
        preds
    )

    precision = precision_score(
        y_test,
        preds
    )

    recall = recall_score(
        y_test,
        preds
    )

    f1 = f1_score(
        y_test,
        preds
    )

    auc = roc_auc_score(
        y_test,
        probs
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    results.append([

        name,
        acc,
        precision,
        recall,
        f1,
        auc
    ])

    if auc > best_auc:

        best_auc = auc

        best_model = model

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC"
    ]
)

print("\n")
print(results_df)

results_df.to_csv(

    "ml_outputs/model_comparison.csv",

    index=False
)

# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(

    best_model,

    "models/xgboost_model.pkl"
)

print("\nBest Model Saved")

# ==========================================================
# FINAL PREDICTIONS
# ==========================================================

preds = best_model.predict(
    X_test
)

probs = best_model.predict_proba(
    X_test
)[:,1]

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    preds
)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"
)

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "ml_outputs/confusion_matrix.png"
)

plt.show()

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report = classification_report(

    y_test,
    preds
)

print("\nClassification Report")
print(report)

with open(

    "ml_outputs/classification_report.txt",

    "w"

) as f:

    f.write(report)

# ==========================================================
# ROC CURVE
# ==========================================================

fpr, tpr, thresholds = roc_curve(

    y_test,
    probs
)

plt.figure(figsize=(8,6))

plt.plot(

    fpr,
    tpr,

    label=f"AUC = {best_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.savefig(
    "ml_outputs/roc_curve.png"
)

plt.show()

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance":
        best_model.feature_importances_
    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False
    )

    importance.to_csv(

        "ml_outputs/feature_importance.csv",

        index=False
    )

    plt.figure(figsize=(10,6))

    sns.barplot(

        data=importance.head(15),

        x="Importance",

        y="Feature"
    )

    plt.title(
        "Top 15 Important Features"
    )

    plt.savefig(

        "ml_outputs/feature_importance.png"
    )

    plt.show()

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction_df = pd.DataFrame({

    "Actual": y_test,

    "Predicted": preds,

    "Probability": probs

})

prediction_df.to_csv(

    "ml_outputs/test_predictions.csv",

    index=False
)

print("\nModel Training Completed")
print("\nOutputs Saved In ml_outputs/")