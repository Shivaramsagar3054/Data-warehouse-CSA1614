"""
Module 8 — Visualization Subsystem
Generates all 15 required publication-grade visualizations for HADSS report & dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")

# Set global publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.0,
    'grid.color': '#eeeeee',
    'figure.autolayout': True
})

def save_fig(filename: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[VISUALIZATION] Saved figure: {filename}")

# Chart 1: Missing values before/after
def plot_missing_values_before_after(quality_report_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    metrics = quality_report_df.set_index('metric')['value'].to_dict()
    
    before = metrics.get('Missing Values (Before)', 0)
    after = metrics.get('Missing Values (After)', 0)
    
    bars = plt.bar(['Before Preprocessing', 'After Preprocessing'], [before, after], color=['#e74c3c', '#2ecc71'], width=0.5)
    plt.title('Data Quality: Missing Values Count Before vs After Preprocessing', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Missing Values Count', fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(before, 1)*0.02), f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    save_fig('missing_values_before_after.png')

# Chart 2: Readmission Distribution
def plot_readmission_distribution(full_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    counts = full_df['readmitted'].value_counts()
    palette = {'NO': '#3498db', '>30': '#f39c12', '<30': '#e74c3c'}
    
    bars = plt.bar(counts.index, counts.values, color=[palette.get(k, '#95a5a6') for k in counts.index], width=0.5)
    plt.title('Hospital Readmission Distribution (Original UCI Dataset)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Readmission Category (<30 days, >30 days, NO)', fontsize=11)
    plt.ylabel('Encounter Count', fontsize=11)
    
    total = len(full_df)
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2., h + (total * 0.01), f'{h:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    save_fig('readmission_distribution.png')

# Chart 3: Age Distribution
def plot_age_distribution(full_df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    age_counts = full_df['age'].value_counts().sort_index()
    
    plt.bar(age_counts.index, age_counts.values, color='#2c3e50', width=0.6)
    plt.title('Patient Encounters by Age Bracket', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Age Group', fontsize=11)
    plt.ylabel('Encounter Count', fontsize=11)
    plt.xticks(rotation=45)
    
    save_fig('age_distribution.png')

# Chart 4: Admission Type Distribution
def plot_admission_type_distribution(full_df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    adm_counts = full_df['admission_type_id'].value_counts().sort_index()
    
    plt.bar(adm_counts.index.astype(str), adm_counts.values, color='#16a085', width=0.5)
    plt.title('Patient Encounters by Admission Type ID', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Admission Type ID (1: Emergency, 2: Urgent, 3: Elective, etc.)', fontsize=11)
    plt.ylabel('Encounter Count', fontsize=11)
    
    save_fig('admission_type_distribution.png')

# Chart 5: Length of Stay Distribution
def plot_length_of_stay_distribution(full_df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    sns.histplot(data=full_df, x='time_in_hospital', hue='readmitted', multiple='stack', discrete=True, palette={'NO': '#3498db', '>30': '#f39c12', '<30': '#e74c3c'})
    plt.title('Hospital Length of Stay (Days) by Readmission Category', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Length of Stay (Days)', fontsize=11)
    plt.ylabel('Encounter Count', fontsize=11)
    
    save_fig('length_of_stay_distribution.png')

# Chart 6: Confusion Matrix
def plot_confusion_matrices(clf_results: dict):
    n_models = len(clf_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, res) in enumerate(clf_results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['No Readmit', '<30 Readmit'], yticklabels=['No Readmit', '<30 Readmit'])
        axes[idx].set_title(f'{name}\nF1: {res["f1_score"]:.3f}', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
    
    plt.suptitle('Classification Confusion Matrices (30-Day Hospital Readmission)', fontsize=12, fontweight='bold', y=1.05)
    save_fig('classification_confusion_matrices.png')

# Chart 7: Model Comparison Chart
def plot_model_comparison(metrics_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    df_melt = metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title('Predictive Performance Comparison Across Classifier Architectures', fontsize=12, fontweight='bold', pad=15)
    plt.ylim(0, 1.05)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_fig('model_comparison_chart.png')

# Chart 8: ROC Curves
def plot_roc_curves(clf_results: dict):
    plt.figure(figsize=(8, 6))
    for name, res in clf_results.items():
        plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['roc_auc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance Baseline (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.legend(loc='lower right')
    
    save_fig('roc_curves.png')

# Chart 9: Precision-Recall Curves
def plot_precision_recall_curves(clf_results: dict):
    plt.figure(figsize=(8, 6))
    for name, res in clf_results.items():
        plt.plot(res['recall_curve'], res['precision_curve'], label=f"{name} (F1 = {res['f1_score']:.3f})", linewidth=2)
    
    plt.title('Precision-Recall Curves for Imbalanced 30-Day Readmission Task', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
    plt.legend(loc='upper right')
    
    save_fig('precision_recall_curves.png')

# Chart 10: Elbow Plot
def plot_elbow(eval_res: dict):
    plt.figure(figsize=(8, 5))
    plt.plot(eval_res['k_values'], eval_res['inertias'], 'bo-', linewidth=2, markersize=8)
    plt.title('K-Means Clustering: Elbow Method for Optimal K Selection', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=11)
    plt.axvline(x=eval_res['best_k'], color='r', linestyle='--', label=f'Optimal K = {eval_res["best_k"]}')
    plt.legend()
    
    save_fig('elbow_plot.png')

# Chart 11: Silhouette Scores
def plot_silhouette(eval_res: dict):
    plt.figure(figsize=(8, 5))
    plt.plot(eval_res['k_values'], eval_res['silhouettes'], 'gs-', linewidth=2, markersize=8)
    plt.title('K-Means Clustering: Silhouette Evaluation Curve', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Mean Silhouette Coefficient', fontsize=11)
    plt.axvline(x=eval_res['best_k'], color='r', linestyle='--', label=f'Best Silhouette K = {eval_res["best_k"]}')
    plt.legend()
    
    save_fig('silhouette_plot.png')

# Chart 12: PCA Cluster Scatter Plot
def plot_pca_clusters(df_clustered: pd.DataFrame, best_k: int):
    plt.figure(figsize=(9, 6))
    palette = sns.color_palette('tab10', n_colors=best_k)
    
    sns.scatterplot(
        data=df_clustered, x='pca_1', y='pca_2', hue='cluster',
        palette=palette, alpha=0.6, s=25, edgecolor=None
    )
    plt.title(f'2D Principal Component Projection of K-Means Patient Segments (K={best_k})', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Principal Component 1 (Variance Explaining Axis)', fontsize=11)
    plt.ylabel('Principal Component 2', fontsize=11)
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_fig('pca_clusters.png')

# Chart 13: Cluster Profiles Visualization
def plot_cluster_profiles(summary_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df['Cluster'], summary_df['30-Day Readmission Rate (%)'], color='#8e44ad', width=0.5)
    plt.title('30-Day Readmission Rate Across Identified Patient Clusters', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Patient Cluster', fontsize=11)
    plt.ylabel('30-Day Readmission Rate (%)', fontsize=11)
    
    for idx, row in summary_df.iterrows():
        plt.text(idx, row['30-Day Readmission Rate (%)'] + 0.5, f"{row['30-Day Readmission Rate (%)']:.1f}%", ha='center', fontweight='bold')
        
    save_fig('cluster_profiles.png')

# Chart 14: Top Association Rules Chart
def plot_top_association_rules(rules_df: pd.DataFrame):
    plt.figure(figsize=(9, 5))
    if len(rules_df) > 0:
        top_rules = rules_df.head(10).copy()
        labels = [f"R{i+1}: {r['Antecedents']} → {r['Consequents']}" for i, r in top_rules.iterrows()]
        
        sns.barplot(data=top_rules, x='Lift', y=labels, palette='mako')
        plt.title('Top 10 Clinical Association Rules Ranked by Lift', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Lift Metric (> 1.0 indicates strong co-occurrence)', fontsize=11)
        plt.ylabel('Clinical Rule (Antecedent → Consequent)', fontsize=10)
    else:
        plt.text(0.5, 0.5, 'No Association Rules Meets Thresholds', ha='center', va='center')
        
    save_fig('top_association_rules.png')

# Chart 15: Healthcare Utilization Charts
def plot_healthcare_utilization_charts(full_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.boxplot(data=full_df, x='readmitted', y='num_lab_procedures', ax=axes[0], palette={'NO': '#3498db', '>30': '#f39c12', '<30': '#e74c3c'})
    axes[0].set_title('Lab Procedures Count by Readmission Status', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Readmission Category')
    axes[0].set_ylabel('Number of Lab Procedures')
    
    sns.boxplot(data=full_df, x='readmitted', y='num_medications', ax=axes[1], palette={'NO': '#3498db', '>30': '#f39c12', '<30': '#e74c3c'})
    axes[1].set_title('Medications Count by Readmission Status', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Readmission Category')
    axes[1].set_ylabel('Number of Prescribed Medications')
    
    plt.suptitle('Healthcare Resource Utilization Analysis', fontsize=13, fontweight='bold', y=1.03)
    save_fig('healthcare_utilization_charts.png')

def generate_all_visualizations(full_df, bin_df, quality_df, clf_res, metrics_df, eval_res, df_clust, summary_df, rules_df):
    """Executes master pipeline generating all 15 required publication charts."""
    plot_missing_values_before_after(quality_df)
    plot_readmission_distribution(full_df)
    plot_age_distribution(full_df)
    plot_admission_type_distribution(full_df)
    plot_length_of_stay_distribution(full_df)
    plot_confusion_matrices(clf_res)
    plot_model_comparison(metrics_df)
    plot_roc_curves(clf_res)
    plot_precision_recall_curves(clf_res)
    plot_elbow(eval_res)
    plot_silhouette(eval_res)
    plot_pca_clusters(df_clust, eval_res['best_k'])
    plot_cluster_profiles(summary_df)
    plot_top_association_rules(rules_df)
    plot_healthcare_utilization_charts(full_df)
    print("[VISUALIZATION] All 15 charts generated and saved under outputs/figures/")

if __name__ == "__main__":
    from src.preprocessing import preprocess_pipeline
    from src.classification import train_and_evaluate_models
    from src.clustering import run_kmeans_evaluation, generate_cluster_profiles
    from src.association_rules import mine_association_rules

    full_df, bin_df, qual_df = preprocess_pipeline()
    clf_res, metrics_df, _ = train_and_evaluate_models(bin_df)
    eval_res, best_k = run_kmeans_evaluation(bin_df)
    df_clust, summary_df = generate_cluster_profiles(bin_df, eval_res, best_k)
    _, rules_df = mine_association_rules(bin_df)

    generate_all_visualizations(full_df, bin_df, qual_df, clf_res, metrics_df, eval_res, df_clust, summary_df, rules_df)
