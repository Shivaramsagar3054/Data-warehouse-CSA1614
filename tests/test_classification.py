"""
Unit tests for Module 5 — Classification & Risk Prediction (CO3).
"""

import pytest
import pandas as pd
from src.preprocessing import preprocess_pipeline
from src.classification import train_and_evaluate_models, predict_patient_risk

def test_classification_training_and_metrics():
    """Verify classifiers train and return valid metrics."""
    _, bin_df, _ = preprocess_pipeline()
    # Test on subset for quick pytest execution speed
    sample_df = bin_df.sample(n=min(2000, len(bin_df)), random_state=42)
    
    results, metrics_df, best_payload = train_and_evaluate_models(sample_df)
    
    assert len(results) == 4
    assert isinstance(metrics_df, pd.DataFrame)
    assert set(metrics_df.columns) == {"Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"}
    assert best_payload["model_name"] in results

def test_patient_risk_inference():
    """Verify single patient risk prediction interface."""
    patient = {
        'age': '[60-70)',
        'age_num': 65,
        'gender': 'Female',
        'race': 'Caucasian',
        'admission_type_id': 1,
        'time_in_hospital': 4,
        'num_lab_procedures': 45,
        'num_procedures': 1,
        'num_medications': 15,
        'number_outpatient': 0,
        'number_emergency': 0,
        'number_inpatient': 1,
        'prior_visit_count': 1,
        'high_utilization_flag': 0,
        'diag_1_cat': 'Circulatory',
        'insulin': 'Steady'
    }
    res = predict_patient_risk(patient)
    assert "readmission_probability" in res
    assert res["risk_category"] in ["LOW RISK", "HIGH RISK"]
    assert "Academic decision-support" in res["disclaimer"]
