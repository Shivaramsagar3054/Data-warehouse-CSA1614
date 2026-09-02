"""
Module 7 — Association Rule Mining (CO5)
Discovers relationships and treatment patterns in healthcare encounters using Apriori.
"""

import os
import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori, association_rules

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "tables")
RULES_CSV_PATH = os.path.join(TABLES_DIR, "association_rules.csv")

def create_healthcare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms tabular healthcare features into a binary (0/1) transaction matrix
    of meaningful clinical events.
    """
    tx_df = pd.DataFrame(index=df.index)

    # 1. Primary Diagnosis Category
    if 'diag_1_cat' in df.columns:
        for cat in df['diag_1_cat'].dropna().unique():
            tx_df[f"Diag1_{cat}"] = (df['diag_1_cat'] == cat).astype(int)

    # 2. Key Medications
    meds = ['insulin', 'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']
    for m in meds:
        if m in df.columns:
            tx_df[f"Med_{m.capitalize()}_Prescribed"] = df[m].isin(['Up', 'Down', 'Steady']).astype(int)

    if 'diabetesMed' in df.columns:
        tx_df["DiabetesMed_Yes"] = (df['diabetesMed'] == 'Yes').astype(int)
    if 'change' in df.columns:
        tx_df["Med_Change_Yes"] = (df['change'] == 'Ch').astype(int)

    # 3. Lab Test Results
    if 'A1Cresult' in df.columns:
        tx_df["A1C_High (>7/8)"] = df['A1Cresult'].isin(['>7', '>8']).astype(int)
        tx_df["A1C_Normal"] = (df['A1Cresult'] == 'Norm').astype(int)

    if 'max_glu_serum' in df.columns:
        tx_df["Glucose_High (>200/300)"] = df['max_glu_serum'].isin(['>200', '>300']).astype(int)

    # 4. Clinical Utilization Flags
    if 'time_in_hospital' in df.columns:
        tx_df["Stay_Long (>=7 days)"] = (df['time_in_hospital'] >= 7).astype(int)
    if 'num_lab_procedures' in df.columns:
        tx_df["High_Lab_Procedures (>=50)"] = (df['num_lab_procedures'] >= 50).astype(int)
    if 'high_utilization_flag' in df.columns:
        tx_df["High_Prior_Utilization"] = (df['high_utilization_flag'] == 1).astype(int)
    if 'admission_type_id' in df.columns:
        tx_df["Admission_Emergency"] = (df['admission_type_id'] == 1).astype(int)

    # 5. Outcome
    if 'readmission_30d' in df.columns:
        tx_df["Readmitted_30d"] = (df['readmission_30d'] == 1).astype(int)

    return tx_df

def mine_association_rules(
    df: pd.DataFrame,
    min_support: float = 0.03,
    min_confidence: float = 0.2,
    min_lift: float = 1.1
) -> (pd.DataFrame, pd.DataFrame):
    """
    Mines frequent itemsets with Apriori and extracts association rules.
    Sorts by Lift, Confidence, Support.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)

    tx_matrix = create_healthcare_transactions(df)

    print(f"[ASSOCIATION RULES] Mining transactions with {tx_matrix.shape[0]} rows and {tx_matrix.shape[1]} clinical items...")

    # Frequent Itemsets via Apriori
    frequent_itemsets = apriori(tx_matrix, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        print("[ASSOCIATION RULES WARNING] No itemsets found with min_support threshold. Relaxing support to 0.01...")
        frequent_itemsets = apriori(tx_matrix, min_support=0.01, use_colnames=True)

    # Generate Rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        # Filter by min lift
        rules = rules[rules['lift'] >= min_lift].sort_values(by=['lift', 'confidence', 'support'], ascending=False)
    else:
        rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])

    # Format output table
    formatted_records = []
    for idx, r in rules.iterrows():
        ant = ", ".join(list(r['antecedents']))
        con = ", ".join(list(r['consequents']))
        formatted_records.append({
            "Antecedents": ant,
            "Consequents": con,
            "Support": round(r['support'], 4),
            "Confidence": round(r['confidence'], 4),
            "Lift": round(r['lift'], 4),
            "Clinical Pattern Observation": f"Encounters with '{ant}' frequently exhibit '{con}' (Lift: {r['lift']:.2f}x baseline). Note: Association does not imply causation."
        })

    formatted_df = pd.DataFrame(formatted_records)
    formatted_df.to_csv(RULES_CSV_PATH, index=False)

    print(f"[ASSOCIATION RULES] Extracted {len(formatted_df)} strong association rules. Saved to {RULES_CSV_PATH}")
    return frequent_itemsets, formatted_df

if __name__ == "__main__":
    from src.preprocessing import preprocess_pipeline
    _, bin_df, _ = preprocess_pipeline()
    freq, rules_df = mine_association_rules(bin_df)
    print("\nTop 10 Association Rules:")
    print(rules_df.head(10).to_string(index=False))
