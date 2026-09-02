# Smart Healthcare Analytics and Decision Support System (HADSS)

**Course:** CSA16 – Data Warehousing and Data Mining  
**Institution / Assignment:** CSA1614 Final End-to-End Implementation  
**Dataset Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals for Years 1999-2008 (Dataset ID: 296)  

---

## 1. Project Title & Problem Statement

### Project Title
**Smart Healthcare Analytics and Decision Support System (HADSS)**

### Problem Statement
Unplanned 30-day hospital readmissions impose significant clinical and financial burdens on healthcare systems worldwide. Identifying high-risk patients early, understanding treatment patterns, optimizing resource allocation, and providing evidence-based decision support require integrating heterogeneous healthcare data into a unified analytical system. HADSS addresses this challenge by combining Data Warehousing, Data Preprocessing, Supervised Machine Learning Classification, Unsupervised Patient Clustering, Association Rule Mining, and an Interactive Decision-Support Dashboard.

---

## 2. Objectives & Course Outcomes (CO1 - CO5)

- **CO1 – Data Warehouse Architecture**: Design and build a Star Schema Data Warehouse in SQLite (`DIM_PATIENT`, `DIM_TIME`, `DIM_HOSPITAL`, `DIM_DIAGNOSIS`, `DIM_MEDICATION`, `DIM_ADMISSION`, `FACT_ENCOUNTER`) with foreign key integrity and 12 pre-built OLAP analytical SQL queries.
- **CO2 – Data Preprocessing**: Implement complete data cleaning (`?` replacement, missing value imputation, duplicate removal), data integration (6 logical source views), data transformation (ICD-9 mapping to 9 clinical categories, age grouping, utilization flags, binary readmission target creation), and feature reduction.
- **CO3 – Classification & Prediction**: Evaluate four supervised machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) on an 80/20 stratified split to predict 30-day early hospital readmission (`readmission_30d`).
- **CO4 – Clustering & Visualization**: Segment patient cohorts using K-Means clustering, evaluate optimal $K$ via Elbow and Silhouette methods, and project cluster boundaries in 2D space using Principal Component Analysis (PCA).
- **CO5 – Association Rule Mining**: Discover co-occurring healthcare event patterns using the Apriori algorithm (Support, Confidence, Lift).

---

## 3. Dataset Source & Acknowledgment

- **Dataset Name:** Diabetes 130-US Hospitals for Years 1999-2008
- **Source:** Official UCI Machine Learning Repository (Dataset ID: 296)
- **Total Records:** 101,766 clinical hospital encounters
- **Total Features:** 47 clinical, administrative, and diagnostic features
- **Acknowledgement:** This project strictly uses the real public UCI Dataset 296 for academic benchmarking of the HADSS architecture. It is NOT fabricated or synthetic data, and is not claimed as private hospital data.

---

## 4. Target Definition Notice

In accordance with strict healthcare classification standards:
- **Binary Target (`readmission_30d`)**:
  - `<30` -> `1` (Readmitted within 30 days)
  - `NO` -> `0` (No early readmission)
  - Encounters with `>30` are excluded from the primary binary classification dataset to avoid label ambiguity (or analyzed separately in multi-class EDA).

---

## 5. System Architecture & Project Structure

```
smart-healthcare-hadss/
│
├── app.py                     # Streamlit 9-Page Interactive Dashboard
├── requirements.txt           # Dependency Specifications
├── README.md                  # Project Master Documentation
├── .gitignore                 # Version Control Ignore Rules
│
├── data/
│   ├── raw/                   # Raw UCI Dataset (diabetic_data.csv)
│   ├── processed/             # Cleaned & Binary Transformed Datasets
│   └── warehouse/             # SQLite Data Warehouse Database (hadss_dw.db)
│
├── src/
│   ├── __init__.py            # Package Init
│   ├── data_loader.py         # Module 1: Programmatic UCI Dataset Downloader & Validator
│   ├── preprocessing.py       # Module 2: CO2 Cleaning, Integration, Transformation, Reduction
│   ├── warehouse.py           # Module 3 & 4: Star Schema DDL, ETL & 12 OLAP Queries
│   ├── classification.py      # Module 5: CO3 Supervised Classifiers & Risk Predictor
│   ├── clustering.py          # Module 6: CO4 K-Means, Elbow, Silhouette & PCA 2D
│   ├── association_rules.py   # Module 7: CO5 Apriori Rule Mining (Support, Confidence, Lift)
│   ├── visualization.py       # Module 8: Publication Charts Generator (15 Plots)
│   └── utils.py               # Master Pipeline Execution & Markdown Results Generator
│
├── tests/                     # Pytest Unit Test Suite
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_classification.py
│   ├── test_clustering.py
│   └── test_association.py
│
├── notebooks/
│   └── HADSS_analysis.ipynb   # Interactive Demonstration Notebook
│
├── outputs/
│   ├── figures/               # Generated Publication Charts (15 PNGs)
│   ├── tables/                # Results Metrics CSV Tables
│   └── models/                # Trained Joblib Model Artifacts
│
└── docs/
    ├── architecture.md        # System Architecture & Star Schema Details
    ├── methodology.md         # Data Science & ML Methodology
    ├── report_content.md      # Full Academic Report Text Content
    └── evidence.md            # Execution Verification & Screenshot Mapping
```

---

## 6. Technologies Used

- **Language:** Python 3.10+
- **Data Engineering:** Pandas, NumPy
- **Data Warehouse:** SQLite 3 (`sqlite3`)
- **Machine Learning:** Scikit-Learn
- **Association Mining:** Mlxtend
- **Visualizations:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Testing:** Pytest

---

## 7. Installation & Quick Start Guide

### Step 1: Clone / Setup Workspace & Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset & Run Master Pipeline
Execute the full end-to-end pipeline (downloads dataset, runs preprocessing, populates data warehouse, trains classifiers, runs clustering, mines association rules, and generates 15 figures):
```bash
python -m src.utils
```

### Step 3: Launch Streamlit Dashboard
```bash
streamlit run app.py
```

### Step 4: Run Automated Pytest Suite
```bash
pytest
```

---

## 8. Summary of Results

All results are empirically calculated from full pipeline execution:

1. **Data Quality (CO2)**:
   - Raw records downloaded: **101,766**
   - 30-Day binary classification dataset: Cleaned and filtered to remove label ambiguity.
   - Missing values cleaned: Replaced `'?'` flags, imputed categorical attributes, and removed exact duplicates.
2. **Classification (CO3)**:
   - Models evaluated: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
   - Evaluation prioritized **F1-Score**, **Recall**, and **ROC-AUC** due to healthcare risk detection requirements.
   - Best model artifact saved to `outputs/models/best_readmission_model.joblib`.
3. **Clustering (CO4)**:
   - K-Means patient segmentation evaluated across K=2..8.
   - Optimal K selected based on Silhouette score and Elbow inflection.
   - Cluster profile table saved to `outputs/tables/cluster_summary.csv`.
4. **Association Rules (CO5)**:
   - Apriori algorithm extracted clinical treatment and diagnostic co-occurrence patterns.
   - Rule ranking: Lift $\rightarrow$ Confidence $\rightarrow$ Support. Top rules saved to `outputs/tables/association_rules.csv`.

---

## 9. Alignment with UN Sustainable Development Goals (SDGs)

- **SDG 3: Good Health and Well-being** — Reduces 30-day hospital readmissions through early risk prediction and preventive clinical decision support.
- **SDG 10: Reduced Inequalities** — Enables objective, evidence-based patient risk evaluation across age and demographic brackets.
- **SDG 12: Responsible Consumption and Production** — Optimizes hospital bed utilization, laboratory procedure testing, and medication management.

---

## 10. Ethical Considerations & Disclaimer

> [!WARNING]
> **ACADEMIC DEMONSTRATION ONLY:**
> This system is an academic decision-support prototype. Model predictions, cluster personas, and association rules must NOT be used as a substitute for professional clinical medical judgment or diagnosis. Association does not imply causation.
