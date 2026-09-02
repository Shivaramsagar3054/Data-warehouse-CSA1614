"""
Module 6 — Clustering Algorithms & Visualization (CO4)
Implements K-Means patient segmentation, Elbow Method, Silhouette Evaluation,
PCA 2D Cluster Visualization, and Cluster Profiling.
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tables")
SUMMARY_CSV_PATH = os.path.join(TABLES_DIR, "cluster_summary.csv")

CLUSTER_FEATURES = [
    'age_num', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_outpatient', 'number_emergency',
    'number_inpatient', 'number_diagnoses'
]

def run_kmeans_evaluation(df: pd.DataFrame, k_range=range(2, 9)) -> (dict, int):
    """
    Evaluates K-Means clustering for K values from 2 to 8.
    Calculates Inertia (Elbow) and Silhouette Scores.
    Returns evaluation metrics dictionary and optimal K.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    available_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_raw = df[available_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    inertias = []
    silhouettes = []
    k_values = list(k_range)

    # Subsample for silhouette score calculation speed if dataset is large
    sample_size = min(3000, len(X_scaled))
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled[sample_indices]

    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(float(kmeans.inertia_))
        
        sample_labels = cluster_labels[sample_indices]
        score = silhouette_score(X_sample, sample_labels)
        silhouettes.append(float(score))

    # Select optimal K based on max Silhouette score (or elbow method)
    best_idx = int(np.argmax(silhouettes))
    best_k = k_values[best_idx]

    eval_res = {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k": best_k,
        "X_scaled": X_scaled,
        "available_cols": available_cols
    }

    print(f"[CLUSTERING] Evaluated K={k_values}. Optimal K selected: {best_k} (Silhouette: {silhouettes[best_idx]:.4f})")
    return eval_res, best_k

def generate_cluster_profiles(df: pd.DataFrame, eval_res: dict, selected_k: int = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Performs final K-Means segmentation with selected K, computes PCA 2D coordinates,
    and constructs a comprehensive Cluster Profile Summary table.
    """
    if selected_k is None:
        selected_k = eval_res["best_k"]

    X_scaled = eval_res["X_scaled"]

    kmeans = KMeans(n_clusters=selected_k, init='k-means++', n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered['cluster'] = clusters

    # Compute 2D PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df_clustered['pca_1'] = pca_coords[:, 0]
    df_clustered['pca_2'] = pca_coords[:, 1]

    # Build Cluster Summary Table
    profile_records = []
    total_patients = len(df_clustered)

    for c in range(selected_k):
        sub = df_clustered[df_clustered['cluster'] == c]
        c_count = len(sub)
        pct = round((c_count / total_patients) * 100, 2)
        
        avg_age = round(sub['age_num'].mean(), 1) if 'age_num' in sub.columns else 0
        avg_los = round(sub['time_in_hospital'].mean(), 2) if 'time_in_hospital' in sub.columns else 0
        avg_meds = round(sub['num_medications'].mean(), 2) if 'num_medications' in sub.columns else 0
        avg_labs = round(sub['num_lab_procedures'].mean(), 2) if 'num_lab_procedures' in sub.columns else 0
        avg_prior = round(sub['prior_visit_count'].mean(), 2) if 'prior_visit_count' in sub.columns else 0
        
        r_30_pct = round((sub['readmission_30d'] == 1).mean() * 100, 2) if 'readmission_30d' in sub.columns else 0

        # Dynamic evidence-based persona interpretation
        if avg_prior > 2.0 or avg_los > 6.0:
            interpretation = "High Utilization / Complex Case Patients"
        elif avg_age < 50.0:
            interpretation = "Younger / Low Acute Utilization Segment"
        elif avg_labs > 50.0:
            interpretation = "Intensive Diagnostics / High Lab Testing Segment"
        else:
            interpretation = "Moderate Utilization / Standard Inpatient Segment"

        profile_records.append({
            "Cluster": f"Cluster {c}",
            "Patient Count": c_count,
            "Percentage (%)": pct,
            "Avg Age (Years)": avg_age,
            "Avg Length of Stay (Days)": avg_los,
            "Avg Medications": avg_meds,
            "Avg Lab Procedures": avg_labs,
            "Avg Prior Visits": avg_prior,
            "30-Day Readmission Rate (%)": r_30_pct,
            "Analytical Segment Persona": interpretation
        })

    summary_df = pd.DataFrame(profile_records)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    print(f"[CLUSTERING] Generated profiles for K={selected_k}. Saved to {SUMMARY_CSV_PATH}")
    return df_clustered, summary_df

if __name__ == "__main__":
    from src.preprocessing import preprocess_pipeline
    _, bin_df, _ = preprocess_pipeline()
    eval_res, best_k = run_kmeans_evaluation(bin_df)
    df_clust, summary_df = generate_cluster_profiles(bin_df, eval_res, best_k)
    print("\nCluster Profile Table:")
    print(summary_df.to_string(index=False))
