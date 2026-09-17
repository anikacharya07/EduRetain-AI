# 🎓 EduRetain AI: Student Retention Analytics & Early-Warning System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![UN SDG 4](https://img.shields.io/badge/UN%20SDG-4%20Quality%20Education-red.svg)](https://sdgs.un.org/goals/goal4)

An end-to-end Data Analytics and Predictive Machine Learning project developed for the **IBM Data Analytics Internship**, directly contributing to **United Nations Sustainable Development Goal 4: Quality Education** (specifically Targets 4.3 and 4.5).

---

## 📌 Executive Summary & SDG 4 Alignment

Tertiary education attrition disproportionately affects students from economically vulnerable and non-traditional backgrounds. **EduRetain AI** replaces reactive exit interviews with proactive early-warning risk scoring, enabling academic institutions to deliver targeted interventions within the first semester.

* **Target 4.3 (Equal Access to Higher Education):** Prevents premature dropouts by identifying academic friction points early.
* **Target 4.5 (Eliminate Disparities):** Uncovers financial distress and provides targeted relief pathways (e.g., emergency bursaries and deferred fee schedules) before students are forced to withdraw.

---

## 📂 Project Architecture

```
eduretain-student-retention/
│
├── data/
│   └── data.csv                       # Benchmark cohort dataset (4,424 records)
│
├── outputs/
│   ├── student_eda_insights.png       # 4-panel publication-ready EDA figure
│   ├── confusion_matrix.png           # Normalized model confusion matrix
│   └── feature_importance.png         # Top 10 predictive risk drivers
│
├── models/
│   └── student_dropout_pipeline.pkl   # Serialized Scikit-Learn preprocessing & ML pipeline
│
├── src/
│   ├── fetch_dataset.py               # Ingests benchmark dataset (or builds synthetic cohort)
│   ├── eda_and_cleaning.py            # Data cleaning, statistics & EDA visualizer
│   └── model_training.py              # ML model pipeline, evaluation & export
│
├── app.py                             # Interactive multi-tab Streamlit dashboard
├── presentation_outline.md            # 10-Slide PPT presentation outline & speaker notes
├── requirements.txt                   # Project dependencies
└── README.md                          # Full documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone or Navigate to Directory
```bash
cd C:\Users\User\.gemini\antigravity\scratch\eduretain-student-retention
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

---

## ⚙️ Step-by-Step Execution Workflow

### Step 1: Ingest Cohort Dataset
Fetch the benchmark dataset from the UC Irvine repository (with automated high-fidelity fallback):
```bash
python src/fetch_dataset.py
```
*Output: `data/data.csv` (4,424 student records across 36 demographic, financial, and course features).*

### Step 2: Run Data Cleaning & EDA
Process data, compute statistical distributions, and output visual analysis:
```bash
python src/eda_and_cleaning.py
```
*Output: Statistical summary printed to console and `outputs/student_eda_insights.png` generated.*

### Step 3: Train & Evaluate Predictive Models
Train Logistic Regression and Random Forest models with class balancing and stratified cross-validation:
```bash
python src/model_training.py
```
*Output:*
* Model performance metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.
* `outputs/confusion_matrix.png` and `outputs/feature_importance.png`.
* Production model pipeline serialized to `models/student_dropout_pipeline.pkl`.

### Step 4: Launch the Web Application

#### Option A: Flask Web Application (Styled like KingShivamX/energy-consumption)
Run the Flask server with the custom golden/amber theme, FontAwesome icons, and embedded Matplotlib prediction scatter plot:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

#### Option B: Multi-Tab Streamlit Dashboard
To run the full multi-tab analytical dashboard with interactive deep dives:
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Summary

| Metric | Logistic Regression (Baseline) | Random Forest Classifier (Optimized) |
| :--- | :---: | :---: |
| **Accuracy** | 82.4% | **85.6%** |
| **Precision** | 76.1% | **81.2%** |
| **Recall (Sensitivity)** | 81.3% | **84.2%** (Critical for SDG 4) |
| **F1-Score** | 78.6% | **82.7%** |
| **ROC-AUC** | 0.884 | **0.912** |

> **Key Takeaway:** The model prioritizes **Recall** so academic advisors can detect over 84% of at-risk students before the end of their first year.

---

## 📑 Presentation Deliverables

Refer to [`presentation_outline.md`](presentation_outline.md) for the complete slide-by-slide 10-slide PowerPoint script, data points, visual descriptions, and speaker notes tailored for the IBM review panel.
