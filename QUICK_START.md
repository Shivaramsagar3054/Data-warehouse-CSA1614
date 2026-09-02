# HADSS Project — Simple How-To-Run Guide

Follow these quick commands to run the **Smart Healthcare Analytics and Decision Support System (HADSS)** on your computer:

---

## 🚀 1. How to Launch the Web Dashboard (Easiest Way)

Open your terminal or PowerShell in this project folder and run:

```bash
streamlit run app.py
```

### What happens:
- It opens the web application automatically in your browser at: **`http://localhost:8501`**
- You can navigate through all **9 interactive pages**:
  - **Overview**: View total encounters, readmission rates, and charts.
  - **Data Quality**: View cleaning and feature transformation metrics.
  - **Data Warehouse**: Run live SQL queries against the SQLite Star Schema database.
  - **Classification**: View model performance metrics and ROC curves.
  - **Patient Risk Prediction**: Calculate readmission risk for individual patients in real-time.
  - **Clustering**: View K-Means patient segmentation and 2D PCA plots.
  - **Association Rules**: Filter clinical Apriori treatment rules.
  - **Insights & About**: View automated data-driven findings and SDG mappings.

---

## ⚙️ 2. How to Re-Run the Complete Data Pipeline

If you want to re-download the dataset, re-clean data, rebuild the database, re-train machine learning models, and re-generate all 15 figure charts, run:

```bash
python -m src.utils
```

### Outputs generated:
- **`outputs/figures/`**: 15 publication-grade charts.
- **`outputs/tables/`**: CSV tables of results.
- **`outputs/models/`**: Trained machine learning model artifact.
- **`data/warehouse/hadss_dw.db`**: SQLite Data Warehouse.

---

## 🧪 3. How to Run Automated Unit Tests

To verify that all code modules are working cleanly with 100% pass rate:

```bash
pytest
```

---

## 📋 Quick Command Summary

| Goal | Command |
| :--- | :--- |
| **Open Web App** | `streamlit run app.py` |
| **Re-run Pipeline** | `python -m src.utils` |
| **Run Tests** | `pytest` |
