# HADSS Complete Academic Project Report Content

**Course:** CSA16 – Data Warehousing and Data Mining  
**Project Title:** Smart Healthcare Analytics and Decision Support System (HADSS)  
**Dataset:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999-2008), ID: 296  

---

## 1. Abstract
Unplanned 30-day hospital readmissions present a major challenge to healthcare systems worldwide, imposing severe financial penalties and reflecting potential gaps in post-discharge patient care. This project presents the Smart Healthcare Analytics and Decision Support System (HADSS), an integrated decision-support system built upon a benchmark dataset of 101,766 clinical hospital encounters across 130 US hospitals. We implement an enterprise Data Warehouse utilizing a Star Schema in SQLite, perform multi-technique data preprocessing (cleaning, integration, transformation, reduction), compare four supervised machine learning classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), execute unsupervised patient segmentation using K-Means clustering (evaluated via Elbow and Silhouette methods), and mine clinical association rules using the Apriori algorithm. All analytical workflows are served through a Streamlit dashboard featuring an interactive patient readmission risk calculator. HADSS directly aligns with UN Sustainable Development Goals (SDG 3, SDG 10, SDG 12) by providing evidence-based analytical decision support.

## 2. Introduction
Modern healthcare organizations generate massive volumes of electronic health record (EHR) data. Transforming raw transactional clinical data into actionable operational and clinical intelligence requires robust Data Warehousing and Data Mining frameworks. HADSS demonstrates how multidimensional star schema modeling, supervised risk classification, unsupervised cohort discovery, and association pattern mining can be combined into a runnable decision-support application.

## 3. Problem Statement
Unplanned 30-day hospital readmissions lead to increased healthcare costs, bed congestion, and adverse patient outcomes. Healthcare administrators and clinical teams lack unified decision-support tools that combine historical warehousing analytical queries with predictive risk estimation and pattern mining.

## 4. Problem Formulation
Let $E = \{e_1, e_2, \dots, e_N\}$ denote a dataset of $N$ hospital encounters, where each encounter $e_i = (\mathbf{x}_i, y_i)$ consists of a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ and a binary 30-day readmission outcome $y_i \in \{0, 1\}$. The primary objective is to learn a predictive classifier $f: \mathbf{x} \mapsto \hat{y} \in [0, 1]$ that maximizes F1-score and Recall, while partitioning $E$ into $K$ cohesive patient clusters $C = \{c_1, \dots, c_K\}$ and discovering association rules $A \Rightarrow B$ with high Lift.

## 5. Objectives
1. **CO1**: Design and implement a Star Schema Data Warehouse architecture for decision support.
2. **CO2**: Apply complete data preprocessing (Cleaning, Integration, Transformation, Reduction).
3. **CO3**: Train and evaluate machine learning classifiers for 30-day readmission prediction.
4. **CO4**: Perform K-Means clustering, evaluate optimal $K$, and visualize segments via 2D PCA.
5. **CO5**: Extract actionable healthcare association rules using Apriori.
6. Support UN SDGs (SDG 3, SDG 10, SDG 12) and provide interactive risk decision support.

## 6. Requirements
- Real public dataset with $\ge 2,000$ rows (UCI Dataset 296: ~101,766 rows).
- Python-based technology stack (Pandas, Scikit-learn, Mlxtend, SQLite, Streamlit).
- Automated dataset downloading, preprocessing reports, model persistence, and unit testing suite.

## 7. Constraints
- Pure academic demonstration prototype; strictly not for independent clinical diagnosis.
- Avoid target leakage during model training.
- No fabricated data or results; all metrics must be empirically computed from execution.

## 8. Assumptions
- Encounter records in the UCI dataset represent independent clinical episodes across 130 hospitals.
- `<30` readmission label corresponds strictly to unplanned 30-day early readmission.

## 9. Dataset Description
- **Source**: UCI Machine Learning Repository (Dataset ID: 296).
- **Records**: 101,766 encounters.
- **Attributes**: 47 features including demographic info (race, gender, age), hospital stay metrics (`time_in_hospital`), procedure/medication counts, prior visit counts (`number_outpatient`, `number_emergency`, `number_inpatient`), ICD-9 diagnosis codes, 23 diabetic drug adjustments, and `readmitted` status.

## 10. Data Warehouse Architecture
Implemented in SQLite (`data/warehouse/hadss_dw.db`).

## 11. Star Schema Design
- **Fact Table**: `FACT_ENCOUNTER`
- **Dimension Tables**: `DIM_PATIENT`, `DIM_TIME`, `DIM_HOSPITAL`, `DIM_DIAGNOSIS`, `DIM_MEDICATION`, `DIM_ADMISSION`.

## 12. Data Preprocessing (CO2)
- **Cleaning**: Resolved missing `'?'` values, removed high-missing columns (`weight`, `payer_code`, `medical_specialty`), imputed categorical missing values, dropped exact duplicates.
- **Integration**: Created 6 logical source views (`patient_source`, `encounter_source`, `diagnosis_source`, `laboratory_source`, `medication_source`, `admission_source`) for ETL demonstration.
- **Transformation**: Mapped ICD-9 codes to 9 clinical categories, engineered `prior_visit_count`, `high_utilization_flag`, `medication_burden`, and created binary target `readmission_30d`.
- **Reduction**: Removed identifier columns (`encounter_id`, `patient_nbr`), reduced dimensionality via PCA.

## 13. Classification Methodology (CO3)
- **Target**: `readmission_30d` (0 vs 1).
- **Split**: 80% train, 20% test (Stratified).
- **Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (`class_weight='balanced'`).

## 14. Classification Results
Evaluated across Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Random Forest / Gradient Boosting achieved superior balance on F1-score and ROC-AUC for imbalanced readmission detection.

## 15. Clustering Methodology (CO4)
- **Algorithm**: K-Means clustering on standardized numeric clinical features.
- **Evaluation**: Tested $K=2..8$ using Elbow Method and Silhouette Coefficient.

## 16. Clustering Results
Identified optimal $K$, generated 2D PCA projection plots, and compiled a Cluster Profile Summary table linking clinical utilization to readmission rate.

## 17. Association Rule Mining (CO5)
- **Algorithm**: Apriori on discretized clinical transactions.
- **Metrics**: Support, Confidence, Lift.

## 18. Association Mining Results
Extracted strong rules connecting diagnosis categories, insulin adjustments, high lab procedure counts, and prior hospital utilization.

## 19. Dashboard Design
9-page Streamlit application (`app.py`) providing interactive visualizations, OLAP query execution, patient risk calculator, and cluster inspection.

## 20. Testing & Validation
Automated `pytest` test suite (`tests/`) verifying dataset loading, row counts, cleaning logic, schema build, ML model training, clustering, association rules, and risk prediction interface.

## 21. Results and Validation
All output figures (`outputs/figures/`), metrics tables (`outputs/tables/`), trained model artifacts (`outputs/models/`), and summary documents (`outputs/results_summary.md`) are empirically generated.

## 22. Analytical Synthesis & Analysis
High prior hospital utilization ($\ge 3$ visits) and high medication burden ($\ge 15$ drugs) correlate strongly with 30-day early readmission.

## 23. Model Architecture Comparison
Decision trees provide high interpretability, while ensemble models (Random Forest, Gradient Boosting) achieve higher non-linear predictive capacity on clinical features.

## 24. Trade-offs
Balanced class weights improve positive class Recall (minimizing False Negatives) at the cost of slightly lower Precision (higher False Positives).

## 25. SDG 3: Good Health and Well-being
Supports early identification of high-risk readmission patients to enable preventive discharge planning.

## 26. SDG 10: Reduced Inequalities
Facilitates objective, data-driven clinical risk assessment across diverse patient demographic groups.

## 27. SDG 12: Responsible Consumption and Production
Optimizes hospital resource allocation, bed usage, and laboratory test utilization.

## 28. Limitations
- Retrospective observational dataset from 1999-2008.
- Missing specific clinical laboratory numerical values beyond categorical indicator flags.

## 29. Future Enhancements
- Integration of deep learning architectures (e.g. Temporal Convolutional Networks).
- Inclusion of real-time EHR API streaming integrations (HL7 / FHIR standards).

## 30. Conclusion
HADSS successfully satisfies all five course outcomes (CO1-CO5) by delivering a complete, runnable end-to-end healthcare analytics decision-support system.

## 31. Individual Contribution Template
- **Data Engineering & DW Design**: Data loading, Star Schema DDL, SQLite ETL.
- **Preprocessing & Analytics**: Preprocessing pipeline, ICD-9 mapping, feature engineering.
- **Machine Learning & Modeling**: Classification pipelines, K-Means clustering, Apriori rule mining.
- **Full-Stack Dashboard & Docs**: Streamlit UI, Pytest suite, report documentation.

## 32. Individual Reflection Template
This project demonstrated the practical challenges of handling real-world imbalanced clinical datasets, executing star schema dimensional transformations, balancing precision-recall trade-offs, and building interactive decision-support applications.

## 33. References
1. UCI Machine Learning Repository: Diabetes 130-US Hospitals for Years 1999-2008 Dataset. Dataset ID: 296.
2. Strack, B., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records. *BioMed Research International*.
3. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.
