"""
Module 1 — Data Acquisition: UCI Dataset 296 (Diabetes 130-US Hospitals 1999-2008)
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "diabetic_data.csv")
UCI_DATASET_ID = 296
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

EXPECTED_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "readmitted"
]

def ensure_directories():
    """Ensure raw, processed, and warehouse data directories exist."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    for subdir in ["data/raw", "data/processed", "data/warehouse", "outputs/figures", "outputs/tables", "outputs/models"]:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)

def download_uci_dataset() -> str:
    """
    Downloads UCI Dataset 296 ('Diabetes 130-US Hospitals for Years 1999-2008')
    using ucimlrepo or fallback HTTP download of official UCI zip archive.
    """
    ensure_directories()
    
    # Check if raw csv exists and has required columns
    if os.path.exists(RAW_CSV_PATH) and os.path.getsize(RAW_CSV_PATH) > 1000000:
        try:
            check_df = pd.read_csv(RAW_CSV_PATH, nrows=5, low_memory=False)
            if 'encounter_id' in check_df.columns and 'patient_nbr' in check_df.columns:
                print(f"[DATA LOADER] Raw dataset verified locally at: {RAW_CSV_PATH}")
                return RAW_CSV_PATH
            else:
                print(f"[DATA LOADER] Local CSV missing ID columns. Re-downloading dataset...")
        except Exception:
            pass

    print("[DATA LOADER] Attempting dataset acquisition via direct HTTP sources...")
    
    # Try Direct GitHub / UCI Mirrors
    MIRRORS = [
        "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
        "https://raw.githubusercontent.com/adrian-s-c/diabetes-readmission-dataset/main/diabetic_data.csv",
        "https://raw.githubusercontent.com/juliencohensolal/Diabetes-130-US-hospitals-for-years-1999-2008-Dataset/master/diabetic_data.csv"
    ]

    for url in MIRRORS:
        try:
            print(f"[DATA LOADER] Trying source: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read()

            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    for file_info in z.infolist():
                        if file_info.filename.endswith("diabetic_data.csv"):
                            with z.open(file_info) as extracted_file:
                                csv_bytes = extracted_file.read()
                                with open(RAW_CSV_PATH, "wb") as f_out:
                                    f_out.write(csv_bytes)
                            print(f"[DATA LOADER] Successfully extracted diabetic_data.csv to {RAW_CSV_PATH}")
                            return RAW_CSV_PATH
                        elif file_info.filename.endswith(".zip"):
                            with z.open(file_info) as nested_zip_file:
                                nested_bytes = nested_zip_file.read()
                                with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nz:
                                    for nfile in nz.infolist():
                                        if nfile.filename.endswith("diabetic_data.csv"):
                                            with nz.open(nfile) as f_in:
                                                csv_bytes = f_in.read()
                                                with open(RAW_CSV_PATH, "wb") as f_out:
                                                    f_out.write(csv_bytes)
                                            print(f"[DATA LOADER] Extracted diabetic_data.csv from nested zip to {RAW_CSV_PATH}")
                                            return RAW_CSV_PATH
            elif url.endswith(".csv"):
                if len(content) > 1000000:
                    with open(RAW_CSV_PATH, "wb") as f_out:
                        f_out.write(content)
                    print(f"[DATA LOADER] Successfully downloaded diabetic_data.csv directly to {RAW_CSV_PATH}")
                    return RAW_CSV_PATH
        except Exception as err:
            print(f"[DATA LOADER] Mirror {url} failed: {err}")

    # Fallback to ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        parts = []
        if hasattr(dataset.data, 'ids') and dataset.data.ids is not None:
            parts.append(dataset.data.ids)
        parts.append(dataset.data.features)
        parts.append(dataset.data.targets)
        df = pd.concat(parts, axis=1)
        if len(df) >= 2000:
            df.to_csv(RAW_CSV_PATH, index=False)
            print(f"[DATA LOADER] Successfully downloaded via ucimlrepo ({len(df)} rows). Saved to {RAW_CSV_PATH}")
            return RAW_CSV_PATH
    except Exception as e:
        print(f"[DATA LOADER] ucimlrepo fallback failed: {e}")

    raise RuntimeError("[DATA LOADER ERROR] Could not acquire UCI Dataset 296. Verification failed.")

def load_raw_data() -> pd.DataFrame:
    """Loads raw dataset from data/raw/diabetic_data.csv."""
    if not os.path.exists(RAW_CSV_PATH):
        download_uci_dataset()
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    return df

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validates dataset row count, column structure, missingness, and duplicates.
    Raises ValueError if validation fails.
    """
    rows, cols = df.shape
    if rows < 2000:
        raise ValueError(f"Dataset validation failed: expected >= 2,000 rows, found {rows}")

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset validation failed: missing required columns {missing_cols}")

    # Count '?' as missing values
    missing_qmark = (df == '?').sum().to_dict()
    missing_null = df.isnull().sum().to_dict()
    total_missing = {col: missing_qmark.get(col, 0) + missing_null.get(col, 0) for col in df.columns}

    duplicate_rows = df.duplicated().sum()

    summary = {
        "status": "VALID",
        "row_count": int(rows),
        "column_count": int(cols),
        "total_missing_values": int(sum(total_missing.values())),
        "missing_per_column": total_missing,
        "duplicate_records": int(duplicate_rows)
    }
    return summary

def get_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame summarizing data types, missing values, and unique counts."""
    records = []
    for col in df.columns:
        q_count = (df[col] == '?').sum() if df[col].dtype == object else 0
        null_count = df[col].isnull().sum()
        total_m = q_count + null_count
        records.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_qmark": int(q_count),
            "missing_null": int(null_count),
            "total_missing": int(total_m),
            "missing_pct": round((total_m / len(df)) * 100, 2),
            "unique_values": df[col].nunique()
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    path = download_uci_dataset()
    data = load_raw_data()
    val = validate_dataset(data)
    print("\nDataset Validation Summary:")
    print(f"Rows: {val['row_count']}, Cols: {val['column_count']}, Duplicates: {val['duplicate_records']}")
    summary_df = get_dataset_summary(data)
    print("\nTop Columns by Missingness:")
    print(summary_df.sort_values(by="missing_pct", ascending=False).head(10).to_string(index=False))
