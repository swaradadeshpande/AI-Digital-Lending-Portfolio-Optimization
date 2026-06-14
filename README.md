# 💳 AI-Powered Digital Lending Portfolio Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An industry-grade fintech analytics platform for loan default prediction, risk scoring, early warning, and portfolio optimization.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [ML Pipeline](#-ml-pipeline)
- [Model Comparison](#-model-comparison)
- [Streamlit Dashboard Pages](#-streamlit-dashboard-pages)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Deployment](#-deployment)
- [Tech Stack](#-tech-stack)
- [Future Scope](#-future-scope)

---

## 🎯 Project Overview

This project applies **advanced machine learning** to assess and manage loan default risk in a digital lending portfolio. It goes beyond a simple classification model — it delivers a complete, production-ready analytics ecosystem:

| Component | Description |
|-----------|-------------|
| 🤖 ML Models | Logistic Regression, Random Forest, **XGBoost** (best) |
| 🧠 Explainability | SHAP global + local explanations for every prediction |
| 📈 Risk Scoring | 0–100 risk score with Low / Medium / High / Critical bands |
| 🚨 Early Warning | Multi-rule alert system flagging customers before default |
| 💰 Portfolio Optimization | Expected loss, risk-adjusted profit, business recommendations |
| 🖥️ Analytics Dashboard | 8-page Streamlit dashboard with interactive Plotly charts |

---

## 🔍 Problem Statement

Digital lending institutions lose significant capital to unexpected loan defaults. Traditional credit scoring systems are:

- **Static** — do not capture real-time behavioural changes
- **Opaque** — lenders cannot explain why a customer was rejected
- **Reactive** — act only after default occurs, not before

This platform solves all three problems by combining predictive ML, model transparency (SHAP), and early warning signals into a single, interactive dashboard.

---

## ✨ Key Features

- ✅ **Three-model comparison** with automatic selection of the best model (XGBoost, ROC-AUC based)
- ✅ **Live customer prediction** form with real-time inference
- ✅ **SHAP summary, bar, waterfall, and dependence plots** with business insights
- ✅ **Customer risk score explorer** with city / loan type / band filters and CSV export
- ✅ **Early warning alerts** (Low / Medium / High / Critical) with actionable recommendations
- ✅ **Portfolio-level KPIs**: total loan book, expected loss, risk-adjusted profit
- ✅ **Modern fintech dark-mode UI** with gradient headers, hover cards, interactive Plotly charts
- ✅ **Zero code duplication** — all shared logic in `utils/`
- ✅ **Fully cached** with `@st.cache_data` and `@st.cache_resource`
- ✅ **Deployable** on Streamlit Community Cloud, Render, and Railway

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                 │
│  lending_dataset.csv (10K synthetic customers)               │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                    ML PIPELINE                                │
│  1. EDA & Visualisation        (eda.py)                      │
│  2. Customer Segmentation      (segmentation.py)  → KMeans   │
│  3. Default Prediction         (default_prediction.py)       │
│     ├── Logistic Regression                                   │
│     ├── Random Forest                                         │
│     └── XGBoost ← BEST (saved to models/)                    │
│  4. SHAP Explainability        (shap_analysis.py)            │
│  5. Risk Scoring Engine        (risk_scoring.py)             │
│  6. Early Warning System       (early_warning_system.py)     │
│  7. Portfolio Optimization     (portfolio_optimization.py)   │
└─────────────────────────┬────────────────────────────────────┘
                          │  outputs (CSVs, PKLs, PNGs)
┌─────────────────────────▼────────────────────────────────────┐
│                  STREAMLIT DASHBOARD                          │
│  app.py → pages/ → utils/                                    │
│  8 pages · Plotly charts · Live inference · CSV export       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🤖 ML Pipeline

### Step-by-Step

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `dataset_generation.py` | Generate 10K synthetic loan records | `data/lending_dataset.csv` |
| 2 | `eda.py` | Exploratory data analysis | EDA plots |
| 3 | `segmentation.py` | KMeans 4-cluster risk segmentation | `lending_dataset_segmented.csv` |
| 4 | `default_prediction.py` | Train 3 models, save best | `models/`, `ml_outputs/` |
| 5 | `shap_analysis.py` | SHAP global & local explanations | `shap_outputs/` |
| 6 | `risk_scoring.py` | 0–100 risk score per customer | `risk_outputs/` |
| 7 | `early_warning_system.py` | Multi-rule warning alerts | `warning_outputs/` |
| 8 | `portfolio_optimization.py` | Portfolio P&L, recommendations | `portfolio_outputs/` |

### Running the Full Pipeline

```bash
python dataset_generation.py
python eda.py
python segmentation.py
python default_prediction.py
python shap_analysis.py
python risk_scoring.py
python early_warning_system.py
python portfolio_optimization.py
```

---

## 📊 Model Comparison

Three models were trained and evaluated. **XGBoost was automatically selected** as the deployed model based on the highest ROC-AUC score.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.78 | ~0.75 | ~0.72 | ~0.73 | ~0.85 |
| Random Forest | ~0.85 | ~0.83 | ~0.81 | ~0.82 | ~0.92 |
| **XGBoost** ✅ | **~0.89** | **~0.87** | **~0.85** | **~0.86** | **~0.96** |

> Exact values are displayed on the Dashboard after running the pipeline.

---

## 🖥️ Streamlit Dashboard Pages

### 🏠 Home
- Project overview, KPI summary cards
- System architecture diagram
- ML pipeline workflow (step-by-step)
- Dataset summary with sample data viewer
- Technology stack badges

### 📊 Dashboard
- Executive KPI row (customers, loan book, default rate, P&L)
- **3-model comparison** — table + radar chart + grouped bar chart
- Risk score histogram, risk band donut chart
- Warning level distribution, loan type bar chart
- Default rate by segment, top 15 feature importances
- Risk score vs credit score scatter plot

### 👤 Customer Prediction
- **Full input form** (28+ features matching training data exactly)
- Live XGBoost inference with preprocessing via saved encoders
- Animated gauge chart showing risk score (0–100)
- Decision display: **Approve / Approve with Monitoring / Reject**
- Key risk indicator progress bars
- Note: "Model auto-selected based on highest ROC-AUC"

### 🧠 Explainable AI (SHAP)
- SHAP summary (beeswarm), bar, waterfall, and dependence plots
- Business insights text report
- Top risk drivers interactive bar chart with search

### 📈 Risk Scoring
- Summary KPIs (total, average, critical count, low count)
- Risk score histogram + risk band pie chart
- Filterable customer table (city, loan type, risk band, customer ID)
- Top 100 riskiest customers
- CSV download of filtered data

### 🚨 Early Warning System
- Alert count KPIs (Low / Medium / High / Critical)
- Warning level pie + warning score histogram
- Colour-coded recommendation cards per alert level
- Filterable customer browser
- High & Critical alert customer table

### 💰 Portfolio Optimization
- Portfolio KPI row (loan book, default rate, expected loss, profit)
- Risk band analysis charts & table
- Customer segment analysis
- Warning level portfolio breakdown
- Business recommendation cards

### ℹ️ About
- Problem statement, objectives
- Architecture table
- Tech stack by category
- Future scope
- Project footer with links

---

## 📁 Project Structure

```
AI-Powered-Digital-Lending/
│
├── 📄 app.py                          # Streamlit entry point
│
├── 📁 pages/                          # One file per page
│   ├── home.py
│   ├── dashboard.py
│   ├── prediction.py
│   ├── shap_page.py
│   ├── risk_scoring.py
│   ├── early_warning.py
│   ├── portfolio.py
│   └── about.py
│
├── 📁 utils/                          # Shared utility modules
│   ├── __init__.py
│   ├── data_loader.py                 # Cached CSV / image / text loaders
│   ├── model_utils.py                 # load_model, preprocess, predict
│   ├── charts.py                      # Plotly chart factory functions
│   └── ui_components.py               # CSS, sidebar, cards, badges
│
├── 📄 requirements.txt
│
├── 📁 data/
│   └── lending_dataset.csv            # Raw dataset
│
├── 📁 models/                         # Generated by ML pipeline
│   ├── xgboost_model.pkl
│   └── label_encoder.pkl
│
├── 📁 ml_outputs/                     # Generated by default_prediction.py
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── 📁 shap_outputs/                   # Generated by shap_analysis.py
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── waterfall_plot.png
│   ├── dependence_plot.png
│   ├── top_risk_drivers.csv
│   └── business_insights.txt
│
├── 📁 risk_outputs/                   # Generated by risk_scoring.py
│   ├── customer_risk_scores.csv
│   └── top_100_risky_customers.csv
│
├── 📁 warning_outputs/                # Generated by early_warning_system.py
│   └── early_warning_dataset.csv
│
├── 📁 portfolio_outputs/              # Generated by portfolio_optimization.py
│   ├── portfolio_summary.csv
│   ├── risk_band_analysis.csv
│   ├── segment_analysis.csv
│   ├── warning_level_analysis.csv
│   └── business_recommendations.txt
│
└── 📁 segmentation_outputs/          # Generated by segmentation.py
    └── cluster_profile.csv
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Powered-Digital-Lending.git
cd AI-Powered-Digital-Lending
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the ML pipeline (first time only)

```bash
python dataset_generation.py
python eda.py
python segmentation.py
python default_prediction.py
python shap_analysis.py
python risk_scoring.py
python early_warning_system.py
python portfolio_optimization.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended — Free)

1. Push your repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the entry point
5. Click **Deploy** — done!

> **Note:** Commit all generated output files (`ml_outputs/`, `models/`, `risk_outputs/`, etc.) to the repo so the cloud app has access to them without re-running the pipeline.

### Render / Railway

```bash
# Procfile (for Render)
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| ML Models | XGBoost, Random Forest, Logistic Regression |
| Explainability | SHAP |
| Data Science | Scikit-learn, Pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Web Framework | Streamlit 1.32+ |
| Model Persistence | Joblib |
| Clustering | KMeans, PCA (Scikit-learn) |

---

## 🔮 Future Scope

- 🔄 **Real-time pipeline** using Apache Kafka + Spark Streaming
- 🌐 **REST API** deployment via FastAPI for integration with LOS systems
- ☁️ **Cloud-native** deployment on AWS SageMaker / GCP Vertex AI
- 🤖 **LLM integration** — natural language query interface for non-technical users
- 📱 **Mobile-first UI** redesign as a Progressive Web App
- 📊 **Power BI connector** for enterprise reporting
- 🔐 **RBAC** — role-based access control for risk managers vs. executives

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting framework
- [SHAP](https://shap.readthedocs.io/) — explainable AI library
- [Streamlit](https://streamlit.io/) — rapid ML app framework
- [Plotly](https://plotly.com/) — interactive visualisation library

---

<div align="center">
  <strong>Built with ❤️ · Python · XGBoost · SHAP · Streamlit · Plotly</strong>
</div>
