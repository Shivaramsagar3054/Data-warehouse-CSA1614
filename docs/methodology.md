# HADSS Technical Methodology Document

## 1. Data Acquisition & Validation
The project programmatically ingests the real UCI Machine Learning Repository dataset **"Diabetes 130-US Hospitals for Years 1999-2008" (Dataset ID: 296)**.
- Total raw encounter records: **101,766**
- Features: **47**
- Validation checks verify that the row count exceeds the mandatory threshold of 2,000 rows and that all required clinical attributes are present.

## 2. Preprocessing Methodology (CO2)
Data preprocessing incorporates four foundational techniques:

1. **Data Cleaning**:
   - Replaced missing character flags (`?`) with standard `NaN`.
   - Dropped high-missingness columns (`weight` ~97% missing, `payer_code` ~40% missing, `medical_specialty` ~47% missing).
   - Imputed categorical missing attributes (e.g. `race`) using modal/`Unknown` labels.
   - Resolved duplicate records and dropped invalid gender records.

2. **Data Integration**:
   - Derived 6 logical source tables (`patient_source`, `encounter_source`, `diagnosis_source`, `laboratory_source`, `medication_source`, `admission_source`) from the single UCI source to demonstrate heterogeneous ETL integration.

3. **Data Transformation**:
   - Encoded 3-digit ICD-9 diagnosis codes into 9 clinical categories (Circulatory, Diabetes, Respiratory, Digestive, Genitourinary, Neoplasms, Musculoskeletal, Injury, Other).
   - Created derived metrics: `prior_visit_count`, `high_utilization_flag`, `medication_burden`.
   - Mapped binary target `readmission_30d`: `<30` mapped to 1 (early 30-day readmission), `NO` mapped to 0 (no early readmission). Encounters with `>30` were excluded from the binary classification dataset to maintain strict label validity.

4. **Data Reduction**:
   - Removed non-predictive identifier columns (`encounter_id`, `patient_nbr`).
   - Standardized numeric feature space for K-Means and PCA dimensionality reduction.

## 3. Supervised Classification Strategy (CO3)
- **Target Variable**: `readmission_30d` (0 vs 1).
- **Split**: 80% training set, 20% testing set (Stratified by target variable).
- **Algorithms**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
- **Class Imbalance**: Managed via `class_weight='balanced'` to ensure high sensitivity to positive 30-day readmission cases.
- **Evaluation Criteria**: F1-Score, Recall, and ROC-AUC are prioritized over pure Accuracy.

## 4. Unsupervised Clustering Strategy (CO4)
- **Algorithm**: K-Means clustering on standardized clinical numeric features.
- **Optimal K Determination**: Evaluated K=2 through K=8 using Within-Cluster Sum of Squares (Elbow Method) and Mean Silhouette Scores.
- **Visualization**: 2D Principal Component Analysis (PCA) projection of cluster boundaries.

## 5. Association Rule Mining Strategy (CO5)
- **Algorithm**: Apriori via `mlxtend`.
- **Itemsets**: Discretized healthcare transactions containing diagnosis categories, prescribed medications, high lab result flags, and utilization indicators.
- **Metrics**: Support, Confidence, and Lift.
- **Interpretation Rule**: "Association does not imply causation." Rules serve for resource planning and co-occurrence analysis.
