"""
Module 2 — Data Preprocessing (CO2)
Implements Data Cleaning, Data Integration, Data Transformation, and Data Reduction.
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, "diabetic_cleaned.csv")
REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tables", "data_quality_report.csv")

MEDICATION_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'gliclazide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citogliptin', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]

def map_icd9_to_category(code):
    """Maps ICD-9 code string to 9 high-level clinical categories."""
    if pd.isna(code) or str(code).strip() in ['?', '']:
        return 'Other'
    
    code_str = str(code).strip()
    
    # Handle Diabetes codes specifically (250.x)
    if code_str.startswith('250'):
        return 'Diabetes'
    
    # Try numeric evaluation
    try:
        val = float(code_str)
        if (390 <= val <= 459) or val == 785:
            return 'Circulatory'
        elif (460 <= val <= 519) or val == 786:
            return 'Respiratory'
        elif (520 <= val <= 579) or val == 787:
            return 'Digestive'
        elif (580 <= val <= 629) or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 800 <= val <= 999:
            return 'Injury'
        else:
            return 'Other'
    except ValueError:
        # Non-numeric codes (V-codes, E-codes, etc.)
        return 'Other'

def clean_data(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Data Cleaning: Replace '?' with NaN, drop high missingness columns,
    handle duplicate records and unknown values.
    """
    before_rows, before_cols = df.shape
    before_missing = int((df == '?').sum().sum() + df.isnull().sum().sum())
    before_duplicates = int(df.duplicated().sum())

    df_clean = df.copy()
    df_clean.replace('?', np.nan, inplace=True)

    # Drop columns with excessive missingness (>40%)
    high_missing_cols = ['weight', 'payer_code', 'medical_specialty']
    df_clean.drop(columns=[c for c in high_missing_cols if c in df_clean.columns], inplace=True)

    # Clean gender: drop 'Unknown/Invalid' (usually ~3 rows out of 101k)
    if 'gender' in df_clean.columns:
        df_clean = df_clean[df_clean['gender'] != 'Unknown/Invalid']

    # Impute categorical variables
    if 'race' in df_clean.columns:
        df_clean['race'].fillna('Unknown', inplace=True)

    # Remove exact duplicate rows if any
    df_clean.drop_duplicates(inplace=True)

    after_rows, after_cols = df_clean.shape
    after_missing = int(df_clean.isnull().sum().sum())
    after_duplicates = int(df_clean.duplicated().sum())

    clean_stats = {
        "before_rows": before_rows,
        "after_rows": after_rows,
        "before_cols": before_cols,
        "after_cols": after_cols,
        "before_missing": before_missing,
        "after_missing": after_missing,
        "before_duplicates": before_duplicates,
        "after_duplicates": after_duplicates
    }
    return df_clean, clean_stats

def integrate_logical_sources(df: pd.DataFrame) -> dict:
    """
    Data Integration: Derives 6 logical source tables from the UCI dataset to
    demonstrate heterogeneous source ETL architecture.
    """
    patient_source = df[['patient_nbr', 'race', 'gender', 'age']].drop_duplicates()
    
    encounter_cols = [c for c in ['encounter_id', 'patient_nbr', 'admission_type_id', 
                                  'discharge_disposition_id', 'admission_source_id', 
                                  'time_in_hospital'] if c in df.columns]
    encounter_source = df[encounter_cols]
    
    diag_cols = [c for c in ['encounter_id', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses'] if c in df.columns]
    diagnosis_source = df[diag_cols]
    
    lab_cols = [c for c in ['encounter_id', 'num_lab_procedures', 'max_glu_serum', 'A1Cresult'] if c in df.columns]
    laboratory_source = df[lab_cols]
    
    med_cols = [c for c in ['encounter_id', 'num_medications', 'change', 'diabetesMed'] + MEDICATION_COLUMNS if c in df.columns]
    medication_source = df[med_cols]
    
    admission_cols = [c for c in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id'] if c in df.columns]
    admission_source = df[admission_cols].drop_duplicates()

    sources = {
        "patient_source": patient_source,
        "encounter_source": encounter_source,
        "diagnosis_source": diagnosis_source,
        "laboratory_source": laboratory_source,
        "medication_source": medication_source,
        "admission_source": admission_source
    }
    return sources

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Transformation: Age mapping, Diagnosis categorisation, Medication burden,
    Utilization features, and 30-day binary readmission target creation.
    """
    df_trans = df.copy()

    # Age Group Transformation: '[0-10)' -> 5, etc.
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df_trans['age_num'] = df_trans['age'].map(age_map).fillna(55)

    # Diagnosis Grouping
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df_trans.columns:
            df_trans[f'{diag_col}_cat'] = df_trans[diag_col].apply(map_icd9_to_category)

    # Derived Healthcare Utilization Features
    df_trans['prior_visit_count'] = (
        df_trans['number_outpatient'].fillna(0) +
        df_trans['number_emergency'].fillna(0) +
        df_trans['number_inpatient'].fillna(0)
    )
    df_trans['high_utilization_flag'] = (df_trans['prior_visit_count'] >= 3).astype(int)

    # Medication Burden Calculation (count of prescribed/adjusted active drugs)
    active_meds = 0
    for med in MEDICATION_COLUMNS:
        if med in df_trans.columns:
            active_meds += df_trans[med].isin(['Up', 'Down', 'Steady']).astype(int)
    df_trans['medication_burden'] = active_meds

    # Readmission Target Creation
    # Binary Target for 30-day early readmission: '<30' -> 1, 'NO' -> 0, filter out '>30' for binary ML dataset
    df_trans['readmitted_orig'] = df_trans['readmitted']
    
    # Store flag before filtering
    df_trans['readmission_30d'] = np.where(df_trans['readmitted'] == '<30', 1, 
                                  np.where(df_trans['readmitted'] == 'NO', 0, np.nan))

    return df_trans

def reduce_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Data Reduction: Separates binary classification dataset (excluding '>30')
    and prepares final analytical feature matrix.
    """
    # Filter out '>30' for primary 30-day readmission classification dataset
    df_binary = df[df['readmission_30d'].notnull()].copy()
    df_binary['readmission_30d'] = df_binary['readmission_30d'].astype(int)

    return df, df_binary

def preprocess_pipeline(raw_df: pd.DataFrame = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Executes the complete CO2 data preprocessing pipeline.
    Returns: (cleaned_full_df, cleaned_binary_df, quality_report_df)
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    if raw_df is None:
        from src.data_loader import load_raw_data
        raw_df = load_raw_data()

    cleaned_df, stats = clean_data(raw_df)
    transformed_df = transform_data(cleaned_df)
    full_processed, binary_processed = reduce_data(transformed_df)

    # Save to disk
    full_processed.to_csv(PROCESSED_CSV_PATH, index=False)
    binary_path = os.path.join(PROCESSED_DIR, "diabetic_binary_30d.csv")
    binary_processed.to_csv(binary_path, index=False)

    # Generate Quality Report Table
    report_data = [
        {"metric": "Total Rows (Original Raw)", "value": stats['before_rows']},
        {"metric": "Total Rows (After Cleaning)", "value": stats['after_rows']},
        {"metric": "Total Rows (30-Day Binary ML Dataset)", "value": len(binary_processed)},
        {"metric": "Total Features (Original)", "value": stats['before_cols']},
        {"metric": "Total Features (After Transformation)", "value": full_processed.shape[1]},
        {"metric": "Missing Values (Before)", "value": stats['before_missing']},
        {"metric": "Missing Values (After)", "value": stats['after_missing']},
        {"metric": "Duplicate Records (Before)", "value": stats['before_duplicates']},
        {"metric": "Duplicate Records (After)", "value": stats['after_duplicates']},
        {"metric": "30-Day Readmission Count (<30)", "value": int((binary_processed['readmission_30d'] == 1).sum())},
        {"metric": "No Early Readmission Count (NO)", "value": int((binary_processed['readmission_30d'] == 0).sum())},
        {"metric": "30-Day Readmission Rate (%)", "value": round((binary_processed['readmission_30d'].mean() * 100), 2)}
    ]
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(REPORT_PATH, index=False)

    print(f"[PREPROCESSING COMPLETE] Cleaned dataset saved to {PROCESSED_CSV_PATH}")
    print(f"[PREPROCESSING COMPLETE] Quality report saved to {REPORT_PATH}")
    return full_processed, binary_processed, report_df

if __name__ == "__main__":
    from src.data_loader import load_raw_data
    raw = load_raw_data()
    full_df, bin_df, rep = preprocess_pipeline(raw)
    print("\nData Quality Report:")
    print(rep.to_string(index=False))
