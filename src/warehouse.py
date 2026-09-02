"""
Module 3 — Data Warehouse Architecture & Star Schema (CO1) & Module 4 — OLAP Queries
Implements SQLite Star Schema DDL, ETL population, and 12 analytical SQL queries.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

DW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "warehouse")
DB_PATH = os.path.join(DW_DIR, "hadss_dw.db")

def get_connection():
    """Establishes connection to the SQLite Data Warehouse database."""
    os.makedirs(DW_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_star_schema(conn: sqlite3.Connection):
    """
    Creates Star Schema DDL for HADSS Data Warehouse:
    Dimensions: DIM_PATIENT, DIM_TIME, DIM_HOSPITAL, DIM_DIAGNOSIS, DIM_MEDICATION, DIM_ADMISSION
    Fact Table: FACT_ENCOUNTER
    """
    cursor = conn.cursor()
    
    # Drop existing tables if re-initializing
    tables = ["FACT_ENCOUNTER", "DIM_PATIENT", "DIM_TIME", "DIM_HOSPITAL", 
              "DIM_DIAGNOSIS", "DIM_MEDICATION", "DIM_ADMISSION"]
    for t in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {t};")

    # DDL 1: DIM_PATIENT
    cursor.execute("""
    CREATE TABLE DIM_PATIENT (
        patient_key INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_nbr INTEGER NOT NULL,
        race TEXT,
        gender TEXT,
        age_group TEXT,
        age_num INTEGER
    );
    """)

    # DDL 2: DIM_TIME
    cursor.execute("""
    CREATE TABLE DIM_TIME (
        time_key INTEGER PRIMARY KEY AUTOINCREMENT,
        time_in_hospital INTEGER,
        length_of_stay_category TEXT
    );
    """)

    # DDL 3: DIM_HOSPITAL
    cursor.execute("""
    CREATE TABLE DIM_HOSPITAL (
        hospital_key INTEGER PRIMARY KEY AUTOINCREMENT,
        admission_type_id INTEGER,
        discharge_disposition_id INTEGER,
        admission_source_id INTEGER
    );
    """)

    # DDL 4: DIM_DIAGNOSIS
    cursor.execute("""
    CREATE TABLE DIM_DIAGNOSIS (
        diagnosis_key INTEGER PRIMARY KEY AUTOINCREMENT,
        diag_1 TEXT,
        diag_2 TEXT,
        diag_3 TEXT,
        diag_1_cat TEXT,
        diag_2_cat TEXT,
        diag_3_cat TEXT,
        number_diagnoses INTEGER
    );
    """)

    # DDL 5: DIM_MEDICATION
    cursor.execute("""
    CREATE TABLE DIM_MEDICATION (
        medication_key INTEGER PRIMARY KEY AUTOINCREMENT,
        num_medications INTEGER,
        medication_burden INTEGER,
        insulin TEXT,
        metformin TEXT,
        change TEXT,
        diabetesMed TEXT
    );
    """)

    # DDL 6: DIM_ADMISSION
    cursor.execute("""
    CREATE TABLE DIM_ADMISSION (
        admission_key INTEGER PRIMARY KEY AUTOINCREMENT,
        number_outpatient INTEGER,
        number_emergency INTEGER,
        number_inpatient INTEGER,
        prior_visit_count INTEGER,
        high_utilization_flag INTEGER
    );
    """)

    # DDL 7: FACT_ENCOUNTER
    cursor.execute("""
    CREATE TABLE FACT_ENCOUNTER (
        encounter_key INTEGER PRIMARY KEY AUTOINCREMENT,
        encounter_id INTEGER UNIQUE NOT NULL,
        patient_key INTEGER,
        time_key INTEGER,
        hospital_key INTEGER,
        diagnosis_key INTEGER,
        medication_key INTEGER,
        admission_key INTEGER,
        time_in_hospital INTEGER,
        num_lab_procedures INTEGER,
        num_procedures INTEGER,
        num_medications INTEGER,
        num_outpatient_visits INTEGER,
        num_emergency_visits INTEGER,
        num_inpatient_visits INTEGER,
        readmission_30d INTEGER,
        readmission_status TEXT,
        FOREIGN KEY (patient_key) REFERENCES DIM_PATIENT(patient_key),
        FOREIGN KEY (time_key) REFERENCES DIM_TIME(time_key),
        FOREIGN KEY (hospital_key) REFERENCES DIM_HOSPITAL(hospital_key),
        FOREIGN KEY (diagnosis_key) REFERENCES DIM_DIAGNOSIS(diagnosis_key),
        FOREIGN KEY (medication_key) REFERENCES DIM_MEDICATION(medication_key),
        FOREIGN KEY (admission_key) REFERENCES DIM_ADMISSION(admission_key)
    );
    """)

    conn.commit()
    print("[WAREHOUSE] Star Schema initialized successfully.")

def populate_warehouse(df: pd.DataFrame):
    """
    Fast ETL process to populate Data Warehouse dimensions and fact table from processed DataFrame.
    """
    conn = get_connection()
    init_star_schema(conn)

    print(f"[WAREHOUSE] Populating warehouse with {len(df)} encounter records...")

    # Prepare DIM_PATIENT
    dim_patient = pd.DataFrame({
        'patient_nbr': df['patient_nbr'].fillna(0).astype(int),
        'race': df['race'].fillna('Unknown').astype(str),
        'gender': df['gender'].fillna('Unknown').astype(str),
        'age_group': df['age'].fillna('Unknown').astype(str),
        'age_num': df['age_num'].fillna(55).astype(int)
    })
    dim_patient.to_sql('DIM_PATIENT', conn, if_exists='append', index=False)

    # Prepare DIM_TIME
    los_series = df['time_in_hospital'].fillna(1).astype(int)
    los_cat = np.where(los_series <= 3, "Short (1-3 days)",
               np.where(los_series <= 7, "Medium (4-7 days)", "Long (8+ days)"))
    dim_time = pd.DataFrame({
        'time_in_hospital': los_series,
        'length_of_stay_category': los_cat
    })
    dim_time.to_sql('DIM_TIME', conn, if_exists='append', index=False)

    # Prepare DIM_HOSPITAL
    dim_hospital = pd.DataFrame({
        'admission_type_id': df['admission_type_id'].fillna(0).astype(int),
        'discharge_disposition_id': df['discharge_disposition_id'].fillna(0).astype(int),
        'admission_source_id': df['admission_source_id'].fillna(0).astype(int)
    })
    dim_hospital.to_sql('DIM_HOSPITAL', conn, if_exists='append', index=False)

    # Prepare DIM_DIAGNOSIS
    dim_diag = pd.DataFrame({
        'diag_1': df['diag_1'].fillna('').astype(str),
        'diag_2': df['diag_2'].fillna('').astype(str),
        'diag_3': df['diag_3'].fillna('').astype(str),
        'diag_1_cat': df['diag_1_cat'].fillna('Other').astype(str) if 'diag_1_cat' in df.columns else 'Other',
        'diag_2_cat': df['diag_2_cat'].fillna('Other').astype(str) if 'diag_2_cat' in df.columns else 'Other',
        'diag_3_cat': df['diag_3_cat'].fillna('Other').astype(str) if 'diag_3_cat' in df.columns else 'Other',
        'number_diagnoses': df['number_diagnoses'].fillna(0).astype(int)
    })
    dim_diag.to_sql('DIM_DIAGNOSIS', conn, if_exists='append', index=False)

    # Prepare DIM_MEDICATION
    dim_med = pd.DataFrame({
        'num_medications': df['num_medications'].fillna(0).astype(int),
        'medication_burden': df['medication_burden'].fillna(0).astype(int) if 'medication_burden' in df.columns else 0,
        'insulin': df['insulin'].fillna('No').astype(str),
        'metformin': df['metformin'].fillna('No').astype(str),
        'change': df['change'].fillna('No').astype(str),
        'diabetesMed': df['diabetesMed'].fillna('No').astype(str)
    })
    dim_med.to_sql('DIM_MEDICATION', conn, if_exists='append', index=False)

    # Prepare DIM_ADMISSION
    dim_adm = pd.DataFrame({
        'number_outpatient': df['number_outpatient'].fillna(0).astype(int),
        'number_emergency': df['number_emergency'].fillna(0).astype(int),
        'number_inpatient': df['number_inpatient'].fillna(0).astype(int),
        'prior_visit_count': df['prior_visit_count'].fillna(0).astype(int) if 'prior_visit_count' in df.columns else 0,
        'high_utilization_flag': df['high_utilization_flag'].fillna(0).astype(int) if 'high_utilization_flag' in df.columns else 0
    })
    dim_adm.to_sql('DIM_ADMISSION', conn, if_exists='append', index=False)

    # Prepare FACT_ENCOUNTER (surrogate keys 1..N correspond exactly to row index + 1)
    keys = np.arange(1, len(df) + 1)
    r_orig = df['readmitted'].fillna('NO').astype(str)
    r_30 = (r_orig == '<30').astype(int)

    fact = pd.DataFrame({
        'encounter_id': df['encounter_id'].astype(int),
        'patient_key': keys,
        'time_key': keys,
        'hospital_key': keys,
        'diagnosis_key': keys,
        'medication_key': keys,
        'admission_key': keys,
        'time_in_hospital': los_series,
        'num_lab_procedures': df['num_lab_procedures'].fillna(0).astype(int),
        'num_procedures': df['num_procedures'].fillna(0).astype(int),
        'num_medications': df['num_medications'].fillna(0).astype(int),
        'num_outpatient_visits': df['number_outpatient'].fillna(0).astype(int),
        'num_emergency_visits': df['number_emergency'].fillna(0).astype(int),
        'num_inpatient_visits': df['number_inpatient'].fillna(0).astype(int),
        'readmission_30d': r_30,
        'readmission_status': r_orig
    })
    fact.to_sql('FACT_ENCOUNTER', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()
    print("[WAREHOUSE] ETL Population completed successfully.")

def run_olap_queries() -> dict:
    """
    Executes 12 OLAP analytical SQL queries against the Data Warehouse.
    Returns dictionary of DataFrames.
    """
    conn = get_connection()
    results = {}

    # Query 1: Total Encounters
    results['total_encounters'] = pd.read_sql_query("SELECT COUNT(*) AS total_encounters FROM FACT_ENCOUNTER", conn)

    # Query 2: Total Unique Patients
    results['total_patients'] = pd.read_sql_query("SELECT COUNT(DISTINCT patient_nbr) AS total_unique_patients FROM DIM_PATIENT", conn)

    # Query 3: Overall Readmission Rate
    results['readmission_rate'] = pd.read_sql_query("""
        SELECT readmission_status, COUNT(*) AS encounter_count,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM FACT_ENCOUNTER), 2) AS percentage
        FROM FACT_ENCOUNTER
        GROUP BY readmission_status
    """, conn)

    # Query 4: Readmissions by Age Group
    results['readmission_by_age'] = pd.read_sql_query("""
        SELECT p.age_group, 
               COUNT(f.encounter_key) AS total_encounters,
               SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) AS readmitted_30d_count,
               ROUND(SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(f.encounter_key), 2) AS readmission_rate_pct
        FROM FACT_ENCOUNTER f
        JOIN DIM_PATIENT p ON f.patient_key = p.patient_key
        GROUP BY p.age_group
        ORDER BY p.age_num
    """, conn)

    # Query 5: Readmissions by Admission Type
    results['readmission_by_admission_type'] = pd.read_sql_query("""
        SELECT h.admission_type_id, 
               COUNT(f.encounter_key) AS total_encounters,
               SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) AS readmitted_30d_count,
               ROUND(SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(f.encounter_key), 2) AS readmission_rate_pct
        FROM FACT_ENCOUNTER f
        JOIN DIM_HOSPITAL h ON f.hospital_key = h.hospital_key
        GROUP BY h.admission_type_id
    """, conn)

    # Query 6: Readmissions by Primary Diagnosis Category
    results['readmission_by_diagnosis'] = pd.read_sql_query("""
        SELECT d.diag_1_cat AS primary_diagnosis_category,
               COUNT(f.encounter_key) AS total_encounters,
               SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) AS readmitted_30d_count,
               ROUND(SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(f.encounter_key), 2) AS readmission_rate_pct
        FROM FACT_ENCOUNTER f
        JOIN DIM_DIAGNOSIS d ON f.diagnosis_key = d.diagnosis_key
        GROUP BY d.diag_1_cat
        ORDER BY total_encounters DESC
    """, conn)

    # Query 7: Average Length of Stay by Readmission Status
    results['avg_los_by_readmission'] = pd.read_sql_query("""
        SELECT readmission_status,
               ROUND(AVG(time_in_hospital), 2) AS avg_length_of_stay_days,
               MIN(time_in_hospital) AS min_days,
               MAX(time_in_hospital) AS max_days
        FROM FACT_ENCOUNTER
        GROUP BY readmission_status
    """, conn)

    # Query 8: Average Medications by Readmission Status
    results['avg_meds_by_readmission'] = pd.read_sql_query("""
        SELECT f.readmission_status,
               ROUND(AVG(f.num_medications), 2) AS avg_num_medications,
               ROUND(AVG(m.medication_burden), 2) AS avg_active_med_burden
        FROM FACT_ENCOUNTER f
        JOIN DIM_MEDICATION m ON f.medication_key = m.medication_key
        GROUP BY f.readmission_status
    """, conn)

    # Query 9: Admissions by Discharge Disposition
    results['discharge_disposition_summary'] = pd.read_sql_query("""
        SELECT h.discharge_disposition_id,
               COUNT(f.encounter_key) AS encounter_count,
               SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) AS readmitted_30d_count
        FROM FACT_ENCOUNTER f
        JOIN DIM_HOSPITAL h ON f.hospital_key = h.hospital_key
        GROUP BY h.discharge_disposition_id
        ORDER BY encounter_count DESC
        LIMIT 10
    """, conn)

    # Query 10: Emergency Utilization Summary
    results['emergency_utilization'] = pd.read_sql_query("""
        SELECT a.high_utilization_flag,
               COUNT(f.encounter_key) AS total_encounters,
               ROUND(AVG(f.num_emergency_visits), 2) AS avg_emergency_visits,
               SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) AS readmitted_30d_count
        FROM FACT_ENCOUNTER f
        JOIN DIM_ADMISSION a ON f.admission_key = a.admission_key
        GROUP BY a.high_utilization_flag
    """, conn)

    # Query 11: Inpatient Utilization Summary
    results['inpatient_utilization'] = pd.read_sql_query("""
        SELECT f.num_inpatient_visits,
               COUNT(f.encounter_key) AS encounter_count,
               ROUND(SUM(CASE WHEN f.readmission_status = '<30' THEN 1 ELSE 0 END) * 100.0 / COUNT(f.encounter_key), 2) AS readmission_rate_pct
        FROM FACT_ENCOUNTER f
        GROUP BY f.num_inpatient_visits
        HAVING encounter_count > 50
        ORDER BY f.num_inpatient_visits ASC
    """, conn)

    # Query 12: Laboratory Procedures by Length of Stay Category
    results['lab_by_los_category'] = pd.read_sql_query("""
        SELECT t.length_of_stay_category,
               COUNT(f.encounter_key) AS encounter_count,
               ROUND(AVG(f.num_lab_procedures), 2) AS avg_lab_procedures
        FROM FACT_ENCOUNTER f
        JOIN DIM_TIME t ON f.time_key = t.time_key
        GROUP BY t.length_of_stay_category
    """, conn)

    conn.close()
    return results

if __name__ == "__main__":
    from src.preprocessing import preprocess_pipeline
    full_df, _, _ = preprocess_pipeline()
    populate_warehouse(full_df)
    olap = run_olap_queries()
    print("\nOLAP Query 3 — Readmission Distribution:")
    print(olap['readmission_rate'].to_string(index=False))
    print("\nOLAP Query 4 — Readmissions by Age Group:")
    print(olap['readmission_by_age'].to_string(index=False))
