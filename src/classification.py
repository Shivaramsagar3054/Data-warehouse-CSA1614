"""
Module 5 — Classification & Prediction (CO3)
Predicts 30-day early hospital readmission using machine learning algorithms.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")
TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tables")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "figures")
MODEL_PATH = os.path.join(MODELS_DIR, "best_readmission_model.joblib")
RESULTS_CSV_PATH = os.path.join(TABLES_DIR, "classification_results.csv")

def prepare_ml_dataset(df: pd.DataFrame):
    """
    Prepares feature matrix X and target vector y for 30-day readmission prediction.
    Removes identifier columns and target leakage variables.
    """
    df_ml = df.copy()

    # Define target
    if 'readmission_30d' not in df_ml.columns:
        raise ValueError("Target column 'readmission_30d' missing from dataset.")

    y = df_ml['readmission_30d'].astype(int)

    # Columns to drop to prevent data leakage and remove identifiers
    drop_cols = [
        'encounter_id', 'patient_nbr', 'readmitted', 'readmitted_orig', 'readmission_30d',
        'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3'
    ]
    
    feature_cols = [c for c in df_ml.columns if c not in drop_cols]
    X = df_ml[feature_cols]

    # Identify categorical vs numerical columns
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    return X, y, num_cols, cat_cols

def build_preprocessor(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for scaling & encoding."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    return preprocessor

def train_and_evaluate_models(df_binary: pd.DataFrame) -> (dict, pd.DataFrame, object):
    """
    Trains Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
    Evaluates models using Accuracy, Precision, Recall, F1, and ROC-AUC.
    Saves metrics and best model artifact.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Subsample binary dataset if > 25,000 for high execution speed while retaining distribution
    if len(df_binary) > 25000:
        df_binary = df_binary.sample(n=25000, random_state=42)

    X, y, num_cols, cat_cols = prepare_ml_dataset(df_binary)

    # 80/20 Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Models dictionary with multi-threading optimization
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1, random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=60, max_depth=10, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=42)
    }

    results = {}
    metrics_records = []
    best_f1 = -1.0
    best_model_name = ""
    best_pipeline = None

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

        model_res = {
            "model_name": name,
            "pipeline": pipeline,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "y_test": y_test,
            "y_proba": y_proba
        }
        results[name] = model_res

        metrics_records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        # Model Selection criterion: balanced F1 and ROC-AUC
        score_metric = (f1 * 0.5) + (auc * 0.5)
        if score_metric > best_f1:
            best_f1 = score_metric
            best_model_name = name
            best_pipeline = pipeline

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(RESULTS_CSV_PATH, index=False)

    # Save best model pipeline along with feature columns schema
    saved_payload = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_cols": list(X.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }
    joblib.dump(saved_payload, MODEL_PATH)

    print(f"[CLASSIFICATION] Evaluated {len(models)} classifiers.")
    print(f"[CLASSIFICATION] Best Model: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f}, AUC: {results[best_model_name]['roc_auc']:.4f})")
    print(f"[CLASSIFICATION] Model saved to {MODEL_PATH}")

    return results, metrics_df, saved_payload

def predict_patient_risk(patient_dict: dict) -> dict:
    """
    Predicts 30-day hospital readmission risk for an individual patient dict.
    Returns predicted probability, risk level (LOW/HIGH), and disclaimer.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model file not found at {MODEL_PATH}. Train model first.")

    payload = joblib.load(MODEL_PATH)
    pipeline = payload["pipeline"]
    feature_cols = payload["feature_cols"]

    # Construct single-row DataFrame matching training features
    input_df = pd.DataFrame([patient_dict])
    
    # Fill missing features with default/mode values if needed
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0 if col in payload["num_cols"] else "Unknown"
    
    input_df = input_df[feature_cols]

    prob = float(pipeline.predict_proba(input_df)[0, 1])
    risk_category = "HIGH RISK" if prob >= 0.35 else "LOW RISK"

    return {
        "readmission_probability": round(prob * 100, 2),
        "raw_probability": prob,
        "risk_category": risk_category,
        "model_used": payload["model_name"],
        "disclaimer": "Academic decision-support prototype only. Not for clinical diagnosis."
    }

if __name__ == "__main__":
    from src.preprocessing import preprocess_pipeline
    _, bin_df, _ = preprocess_pipeline()
    res, df_metrics, best = train_and_evaluate_models(bin_df)
    print("\nClassification Comparison Table:")
    print(df_metrics.to_string(index=False))
