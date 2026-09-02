"""
Unit tests for Module 7 — Association Rule Mining (CO5).
"""

import pytest
import pandas as pd
from src.preprocessing import preprocess_pipeline
from src.association_rules import create_healthcare_transactions, mine_association_rules

def test_association_rule_mining():
    """Verify healthcare transaction matrix creation and Apriori rule generation."""
    _, bin_df, _ = preprocess_pipeline()
    sample_df = bin_df.sample(n=min(2000, len(bin_df)), random_state=42)
    
    tx_matrix = create_healthcare_transactions(sample_df)
    assert isinstance(tx_matrix, pd.DataFrame)
    assert tx_matrix.shape[1] >= 5
    
    freq, rules_df = mine_association_rules(sample_df, min_support=0.02, min_confidence=0.1, min_lift=1.0)
    assert isinstance(rules_df, pd.DataFrame)
    if len(rules_df) > 0:
        assert "Support" in rules_df.columns
        assert "Confidence" in rules_df.columns
        assert "Lift" in rules_df.columns
