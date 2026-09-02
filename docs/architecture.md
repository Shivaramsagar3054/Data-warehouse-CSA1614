# HADSS Architecture & System Design Document

## 1. High-Level System Architecture

The **Smart Healthcare Analytics and Decision Support System (HADSS)** is designed as an end-to-end clinical analytics pipeline that ingests real hospital encounter records, executes multi-stage data preprocessing, populates an enterprise Data Warehouse (SQLite Star Schema), trains machine learning classification models, segments patient cohorts via unsupervised clustering, extracts clinical association rules, and serves findings through an interactive Streamlit decision-support dashboard.

```
Heterogeneous Healthcare Sources (UCI Dataset ID 296: ~101,766 Encounters)
        │
        ▼
   ETL / ELT Ingestion Subsystem (data_loader.py)
        │
        ▼
   Data Preprocessing Pipeline (preprocessing.py)
   ├── A. Data Cleaning (Missing value imputation, ? replacement, duplicate removal)
   ├── B. Data Integration (6 Logical views: Patient, Encounter, Diagnosis, Lab, Med, Admission)
   ├── C. Data Transformation (ICD-9 clinical mapping, age grouping, utilization flags)
   └── D. Data Reduction (Identifier removal, feature space optimization)
        │
        ▼
   Healthcare Data Warehouse (warehouse.py - SQLite Star Schema)
   ├── DIM_PATIENT, DIM_TIME, DIM_HOSPITAL, DIM_DIAGNOSIS, DIM_MEDICATION, DIM_ADMISSION
   └── FACT_ENCOUNTER (Foreign keys & Quantitative Measures)
        │
 ┌──────┴───────────────────────┬───────────────────────────────┐
 │                              │                               │
 ▼                              ▼                               ▼
Module 5: Classification        Module 6: Clustering            Module 7: Association Rules
(classification.py)             (clustering.py)                 (association_rules.py)
├── Logistic Regression         ├── K-Means Segmentation        ├── Apriori Algorithm
├── Decision Tree               ├── Elbow Method & Silhouette   ├── Support / Confidence / Lift
├── Random Forest               └── 2D PCA Cluster Projection   └── Clinical Event Patterns
└── Gradient Boosting
 │                              │                               │
 └──────┬───────────────────────┴───────────────────────────────┘
        │
        ▼
   Interactive Streamlit Dashboard (app.py) & Decision Support Prototype
```

---

## 2. Data Warehouse Star Schema

The HADSS Data Warehouse enforces a strict **Star Schema** architecture tailored for dimensional modeling and decision support in clinical operations.

### Schema Blueprint

```
                      ┌───────────────────────────┐
                      │        DIM_PATIENT        │
                      ├───────────────────────────┤
                      │ PK  patient_key (INT)     │
                      │     patient_nbr (INT)     │
                      │     race (TEXT)           │
                      │     gender (TEXT)         │
                      │     age_group (TEXT)      │
                      │     age_num (INT)         │
                      └─────────────┬─────────────┘
                                    │
                                    │
 ┌───────────────────────────┐      │      ┌───────────────────────────┐
 │       DIM_DIAGNOSIS       │      │      │         DIM_TIME          │
 ├───────────────────────────┤      │      ├───────────────────────────┤
 │ PK  diagnosis_key (INT)   │      │      │ PK  time_key (INT)        │
 │     diag_1 (TEXT)         │      │      │     time_in_hospital (INT)│
 │     diag_2 (TEXT)         │      │      │     length_of_stay_cat  │
 │     diag_3 (TEXT)         │      │      └─────────────┬─────────────┘
 │     diag_1_cat (TEXT)     │      │                    │
 │     number_diagnoses (INT)│      │                    │
 └─────────────┬─────────────┘      │                    │
               │                    ▼                    │
               │      ┌───────────────────────────┐      │
               └─────►│      FACT_ENCOUNTER       │◄─────┘
                      ├───────────────────────────┤
                      │ PK  encounter_key (INT)   │
                      │     encounter_id (INT)    │
                      │ FK  patient_key (INT)     │
                      │ FK  time_key (INT)        │
                      │ FK  hospital_key (INT)    │
                      │ FK  diagnosis_key (INT)   │
                      │ FK  medication_key (INT)  │
                      │ FK  admission_key (INT)   │
                      │ --- Measures ---          │
                      │     time_in_hospital (INT)│
                      │     num_lab_procedures    │
                      │     num_procedures (INT)  │
                      │     num_medications (INT) │
                      │     readmission_30d (INT) │
                      │     readmission_status    │
                      └─────────────▲─────────────┘
                                    │
 ┌───────────────────────────┐      │      ┌───────────────────────────┐
 │      DIM_MEDICATION       │──────┼─────►│       DIM_ADMISSION       │
 ├───────────────────────────┤      │      ├───────────────────────────┤
 │ PK  medication_key (INT)  │      │      │ PK  admission_key (INT)   │
 │     num_medications (INT) │      │      │     number_outpatient (INT│
 │     medication_burden(INT)│      │      │     number_emergency(INT│
 │     insulin (TEXT)        │      │      │     number_inpatient(INT│
 │     metformin (TEXT)      │      │      │     prior_visit_count   │
 └───────────────────────────┘      │      └───────────────────────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      │       DIM_HOSPITAL        │
                      ├───────────────────────────┤
                      │ PK  hospital_key (INT)    │
                      │     admission_type_id(INT)│
                      │     discharge_dispos_id   │
                      │     admission_source_id   │
                      └───────────────────────────┘
```

---

## 3. Technology Stack Specification

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Industry standard for Data Engineering & Data Science |
| **Data Engine** | Pandas, NumPy | High-performance vectorized operations |
| **Data Warehouse** | SQLite (`sqlite3`) | Embedded SQL engine for relational star-schema modeling |
| **Machine Learning** | Scikit-Learn | Robust pipelines, classifiers, metrics, and K-Means |
| **Association Mining**| Mlxtend | Standard Apriori algorithm implementation |
| **Visualization** | Matplotlib, Seaborn, Plotly | Publication-ready static & interactive charts |
| **Dashboard UI** | Streamlit | Rapid, responsive full-stack decision-support dashboard |
| **Unit Testing** | Pytest | Automated test coverage for data quality and model integrity |
