"""
Utility Module — Pipeline Runner & Automated Results Summary Generator
Executes end-to-end data processing, warehouse build, modeling, and output generation.
"""

import os
import pandas as pd

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
SUMMARY_MD_PATH = os.path.join(OUTPUTS_DIR, "results_summary.md")

def generate_results_summary_markdown(
    quality_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    summary_cluster_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    best_model_name: str
) -> str:
    """Generates human-readable Markdown summary report from empirical execution results."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    qual_dict = quality_df.set_index('metric')['value'].to_dict()

    md_content = f"""# Smart Healthcare Analytics and Decision Support System (HADSS)
## Empirical Results and Analytical Summary Report

**Execution Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset Source:** UCI Machine Learning Repository — Diabetes 130-US Hospitals (1999-2008), ID: 296

---

### 1. Data Quality & Preprocessing Overview (CO2)
- **Total Raw Records Downloaded:** {qual_dict.get('Total Rows (Original Raw)', 0):,} encounters
- **Cleaned Dataset Records:** {qual_dict.get('Total Rows (After Cleaning)', 0):,} encounters
- **30-Day Readmission Binary Dataset:** {qual_dict.get('Total Rows (30-Day Binary ML Dataset)', 0):,} encounters
- **Initial Features Count:** {qual_dict.get('Total Features (Original)', 0)}
- **Transformed Features Count:** {qual_dict.get('Total Features (After Transformation)', 0)}
- **Missing Values Cleaned:** {qual_dict.get('Missing Values (Before)', 0):,} → {qual_dict.get('Missing Values (After)', 0):,}
- **Duplicate Records Resolved:** {qual_dict.get('Duplicate Records (Before)', 0):,} → {qual_dict.get('Duplicate Records (After)', 0):,}
- **Empirical 30-Day Readmission Rate:** {qual_dict.get('30-Day Readmission Rate (%)', 0):.2f}% ({qual_dict.get('30-Day Readmission Count (<30)', 0):,} positive cases)

---

### 2. Classification & Prediction Performance (CO3)
**Primary Binary Target:** 30-Day Early Readmission (`readmission_30d`)
**Selected Best Classifier:** `{best_model_name}`

#### Model Performance Comparison Table:
{metrics_df.to_markdown(index=False)}

*Note: Models were trained with `class_weight='balanced'` on an 80/20 stratified train/test split. Evaluation prioritizes F1-Score, Recall, and ROC-AUC over pure Accuracy due to healthcare risk detection requirements.*

---

### 3. Patient Clustering & Unsupervised Segmentation (CO4)
**Algorithm:** K-Means with Feature Standardization & 2D PCA

#### Segment Profiles Summary:
{summary_cluster_df.to_markdown(index=False)}

---

### 4. Association Rule Mining (CO5)
**Algorithm:** Apriori (Discretized Clinical Event Transactions)

#### Top Discovered Clinical Rules:
{rules_df.head(8).to_markdown(index=False) if len(rules_df) > 0 else "No rules passed thresholds."}

*Observation: Association rules identify co-occurring treatment and diagnostic patterns. Association does not imply causation.*

---

### 5. Sustainability & UN Sustainable Development Goals (SDGs)
- **SDG 3 (Good Health & Well-Being):** Reduces avoidable 30-day readmissions through early risk prediction.
- **SDG 10 (Reduced Inequalities):** Enables objective risk assessment across diverse patient age and demographic groups.
- **SDG 12 (Responsible Consumption & Production):** Optimizes hospital resource allocation, medication burden, and laboratory testing utilization.
"""

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[SUMMARY] Generated empirical results report: {SUMMARY_MD_PATH}")
    return md_content

def run_full_pipeline():
    """Master execution pipeline."""
    from src.data_loader import load_raw_data, validate_dataset
    from src.preprocessing import preprocess_pipeline
    from src.warehouse import populate_warehouse, run_olap_queries
    from src.classification import train_and_evaluate_models
    from src.clustering import run_kmeans_evaluation, generate_cluster_profiles
    from src.association_rules import mine_association_rules
    from src.visualization import generate_all_visualizations

    print("============================================================")
    print("STARTING HADSS COMPLETE END-TO-END PIPELINE EXECUTION")
    print("============================================================")

    # 1. Data Loader
    raw_df = load_raw_data()
    val_res = validate_dataset(raw_df)
    print(f"Data Loaded: {val_res['row_count']} rows, {val_res['column_count']} cols.")

    # 2. Preprocessing
    full_df, bin_df, qual_df = preprocess_pipeline(raw_df)

    # 3. Warehouse
    populate_warehouse(full_df)
    olap_res = run_olap_queries()

    # 4. Classification
    clf_res, metrics_df, best_payload = train_and_evaluate_models(bin_df)

    # 5. Clustering
    eval_res, best_k = run_kmeans_evaluation(bin_df)
    df_clust, summary_df = generate_cluster_profiles(bin_df, eval_res, best_k)

    # 6. Association Rules
    freq, rules_df = mine_association_rules(bin_df)

    # 7. Visualization
    generate_all_visualizations(full_df, bin_df, qual_df, clf_res, metrics_df, eval_res, df_clust, summary_df, rules_df)

    # 8. Summary Report
    generate_results_summary_markdown(qual_df, metrics_df, summary_df, rules_df, best_payload["model_name"])

    print("============================================================")
    print("HADSS PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print("============================================================")

if __name__ == "__main__":
    run_full_pipeline()
