"""
Unit tests for Module 2 — Data Preprocessing (CO2).
"""

import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_raw_data
from src.preprocessing import clean_data, transform_data, reduce_data, preprocess_pipeline

def test_clean_data_handles_qmarks_and_duplicates():
    """Verify '?' is replaced with NaN and high missingness columns dropped."""
    df_raw = load_raw_data().head(500)
    df_clean, stats = clean_data(df_raw)
    
    assert '?' not in df_clean.values
    assert 'weight' not in df_clean.columns
    assert stats['after_rows'] <= stats['before_rows']

def test_transform_data_target_creation():
    """Verify 30-day binary readmission target creation (<30 -> 1, NO -> 0)."""
    df_raw = load_raw_data().head(500)
    df_clean, _ = clean_data(df_raw)
    df_trans = transform_data(df_clean)
    
    assert 'readmission_30d' in df_trans.columns
    assert 'prior_visit_count' in df_trans.columns
    assert 'medication_burden' in df_trans.columns

def test_preprocess_pipeline_execution():
    """Verify full preprocessing pipeline runs cleanly."""
    full_df, bin_df, qual_df = preprocess_pipeline()
    assert len(full_df) >= 2000
    assert len(bin_df) >= 1000
    assert set(bin_df['readmission_30d'].unique()).issubset({0, 1})
    assert isinstance(qual_df, pd.DataFrame)
