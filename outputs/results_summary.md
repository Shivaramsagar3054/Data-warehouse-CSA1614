# Smart Healthcare Analytics and Decision Support System (HADSS)
## Empirical Results and Analytical Summary Report

**Execution Timestamp:** 2026-09-02 16:35:44
**Dataset Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999-2008), ID: 296

---

### 1. Data Quality & Preprocessing Overview (CO2)
- **Total Raw Records Downloaded:** 101,766.0 encounters
- **Cleaned Dataset Records:** 101,763.0 encounters
- **30-Day Readmission Binary Dataset:** 66,218.0 encounters
- **Initial Features Count:** 50.0
- **Transformed Features Count:** 56.0
- **Missing Values Cleaned:** 374,017.0 → 182,964.0
- **Duplicate Records Resolved:** 0.0 → 0.0
- **Empirical 30-Day Readmission Rate:** 17.15% (11,357.0 positive cases)

---

### 2. Classification & Prediction Performance (CO3)
**Primary Binary Target:** 30-Day Early Readmission (`readmission_30d`)
**Selected Best Classifier:** `Random Forest`

#### Model Performance Comparison Table:
| Model               |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression |     0.6788 |      0.2787 |   0.5381 |     0.3672 |    0.6728 |
| Decision Tree       |     0.5998 |      0.2468 |   0.6386 |     0.356  |    0.6501 |
| Random Forest       |     0.7364 |      0.3229 |   0.4758 |     0.3847 |    0.6937 |
| Gradient Boosting   |     0.8352 |      0.6567 |   0.1016 |     0.176  |    0.7148 |

*Note: Models were trained with `class_weight='balanced'` on an 80/20 stratified train/test split. Evaluation prioritizes F1-Score, Recall, and ROC-AUC over pure Accuracy due to healthcare risk detection requirements.*

---

### 3. Patient Clustering & Unsupervised Segmentation (CO4)
**Algorithm:** K-Means with Feature Standardization & 2D PCA

#### Segment Profiles Summary:
| Cluster   |   Patient Count |   Percentage (%) |   Avg Age (Years) |   Avg Length of Stay (Days) |   Avg Medications |   Avg Lab Procedures |   Avg Prior Visits |   30-Day Readmission Rate (%) | Analytical Segment Persona                        |
|:----------|----------------:|-----------------:|------------------:|----------------------------:|------------------:|---------------------:|-------------------:|------------------------------:|:--------------------------------------------------|
| Cluster 0 |           24859 |            37.54 |              69.4 |                        6.84 |             22.21 |                54.46 |               1.41 |                         21.86 | High Utilization / Complex Case Patients          |
| Cluster 1 |           41359 |            62.46 |              63.5 |                        2.84 |             12.08 |                35.63 |               0.72 |                         14.32 | Moderate Utilization / Standard Inpatient Segment |

---

### 4. Association Rule Mining (CO5)
**Algorithm:** Apriori (Discretized Clinical Event Transactions)

#### Top Discovered Clinical Rules:
| Antecedents                                                                        | Consequents                                                         |   Support |   Confidence |   Lift | Clinical Pattern Observation                                                                                                                                                                                                             |
|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------|----------:|-------------:|-------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Med_Insulin_Prescribed, Admission_Emergency, Stay_Long (>=7 days)                  | Med_Change_Yes, High_Lab_Procedures (>=50)                          |    0.0316 |       0.5056 | 2.7587 | Encounters with 'Med_Insulin_Prescribed, Admission_Emergency, Stay_Long (>=7 days)' frequently exhibit 'Med_Change_Yes, High_Lab_Procedures (>=50)' (Lift: 2.76x baseline). Note: Association does not imply causation.                  |
| Med_Insulin_Prescribed, Stay_Long (>=7 days), Admission_Emergency, DiabetesMed_Yes | Med_Change_Yes, High_Lab_Procedures (>=50)                          |    0.0316 |       0.5056 | 2.7587 | Encounters with 'Med_Insulin_Prescribed, Stay_Long (>=7 days), Admission_Emergency, DiabetesMed_Yes' frequently exhibit 'Med_Change_Yes, High_Lab_Procedures (>=50)' (Lift: 2.76x baseline). Note: Association does not imply causation. |
| Med_Insulin_Prescribed, Admission_Emergency, Stay_Long (>=7 days)                  | Med_Change_Yes, High_Lab_Procedures (>=50), DiabetesMed_Yes         |    0.0316 |       0.5056 | 2.7587 | Encounters with 'Med_Insulin_Prescribed, Admission_Emergency, Stay_Long (>=7 days)' frequently exhibit 'Med_Change_Yes, High_Lab_Procedures (>=50), DiabetesMed_Yes' (Lift: 2.76x baseline). Note: Association does not imply causation. |
| Med_Change_Yes, Admission_Emergency, Stay_Long (>=7 days)                          | Med_Insulin_Prescribed, High_Lab_Procedures (>=50)                  |    0.0316 |       0.5951 | 2.7002 | Encounters with 'Med_Change_Yes, Admission_Emergency, Stay_Long (>=7 days)' frequently exhibit 'Med_Insulin_Prescribed, High_Lab_Procedures (>=50)' (Lift: 2.70x baseline). Note: Association does not imply causation.                  |
| Med_Change_Yes, Stay_Long (>=7 days), Admission_Emergency, DiabetesMed_Yes         | Med_Insulin_Prescribed, High_Lab_Procedures (>=50)                  |    0.0316 |       0.5951 | 2.7002 | Encounters with 'Med_Change_Yes, Stay_Long (>=7 days), Admission_Emergency, DiabetesMed_Yes' frequently exhibit 'Med_Insulin_Prescribed, High_Lab_Procedures (>=50)' (Lift: 2.70x baseline). Note: Association does not imply causation. |
| Med_Change_Yes, Admission_Emergency, Stay_Long (>=7 days)                          | Med_Insulin_Prescribed, High_Lab_Procedures (>=50), DiabetesMed_Yes |    0.0316 |       0.5951 | 2.7002 | Encounters with 'Med_Change_Yes, Admission_Emergency, Stay_Long (>=7 days)' frequently exhibit 'Med_Insulin_Prescribed, High_Lab_Procedures (>=50), DiabetesMed_Yes' (Lift: 2.70x baseline). Note: Association does not imply causation. |
| A1C_High (>7/8), Med_Insulin_Prescribed                                            | Med_Change_Yes, High_Lab_Procedures (>=50)                          |    0.0381 |       0.4774 | 2.6051 | Encounters with 'A1C_High (>7/8), Med_Insulin_Prescribed' frequently exhibit 'Med_Change_Yes, High_Lab_Procedures (>=50)' (Lift: 2.61x baseline). Note: Association does not imply causation.                                            |
| A1C_High (>7/8), Med_Insulin_Prescribed, DiabetesMed_Yes                           | Med_Change_Yes, High_Lab_Procedures (>=50)                          |    0.0381 |       0.4774 | 2.6051 | Encounters with 'A1C_High (>7/8), Med_Insulin_Prescribed, DiabetesMed_Yes' frequently exhibit 'Med_Change_Yes, High_Lab_Procedures (>=50)' (Lift: 2.61x baseline). Note: Association does not imply causation.                           |

*Observation: Association rules identify co-occurring treatment and diagnostic patterns. Association does not imply causation.*

---

### 5. Sustainability & UN Sustainable Development Goals (SDGs)
- **SDG 3 (Good Health & Well-Being):** Reduces avoidable 30-day readmissions through early risk prediction.
- **SDG 10 (Reduced Inequalities):** Enables objective risk assessment across diverse patient age and demographic groups.
- **SDG 12 (Responsible Consumption & Production):** Optimizes hospital resource allocation, medication burden, and laboratory testing utilization.
