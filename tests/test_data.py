"""
Unit tests for Module 1 — Data Acquisition and Validation.
"""

import os
import pytest
import pandas as pd
from src.data_loader import load_raw_data, validate_dataset, get_dataset_summary, RAW_CSV_PATH

def test_raw_dataset_exists_and_loads():
    """Verify raw dataset exists or can be downloaded and has >= 2000 rows."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2000

def test_validate_dataset_schema():
    """Verify schema validation on loaded raw dataset."""
    df = load_raw_data()
    summary = validate_dataset(df)
    assert summary["status"] == "VALID"
    assert summary["row_count"] >= 2000
    assert summary["column_count"] >= 20
    assert "duplicate_records" in summary

def test_get_dataset_summary():
    """Verify summary table generation."""
    df = load_raw_data()
    summary_df = get_dataset_summary(df)
    assert isinstance(summary_df, pd.DataFrame)
    assert "column" in summary_df.columns
    assert "missing_pct" in summary_df.columns
