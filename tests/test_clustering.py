"""
Unit tests for Module 6 — Clustering Algorithms (CO4).
"""

import pytest
import pandas as pd
from src.preprocessing import preprocess_pipeline
from src.clustering import run_kmeans_evaluation, generate_cluster_profiles

def test_kmeans_evaluation_and_profiling():
    """Verify K-Means evaluation, Elbow method, Silhouette score, and Profile table."""
    _, bin_df, _ = preprocess_pipeline()
    sample_df = bin_df.sample(n=min(1500, len(bin_df)), random_state=42)
    
    eval_res, best_k = run_kmeans_evaluation(sample_df, k_range=range(2, 5))
    
    assert 2 <= best_k <= 4
    assert len(eval_res["inertias"]) == 3
    assert len(eval_res["silhouettes"]) == 3
    
    df_clust, summary_df = generate_cluster_profiles(sample_df, eval_res, best_k)
    assert "cluster" in df_clust.columns
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == best_k
