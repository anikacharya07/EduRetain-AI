"""
EduRetain AI - Student Retention & Dropout Prediction (UN SDG 4)
Script: eda_and_cleaning.py
Purpose: Loads data, cleans values, computes statistical summaries,
         and generates publication-quality EDA figures saved to outputs/.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "student_eda_insights.png")

# Setup Matplotlib / Seaborn styles
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

import re

def clean_col(c: str) -> str:
    c = c.strip().replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace("/", "_").replace("\t", "")
    return re.sub(r'_+', '_', c).rstrip('_')

def load_and_clean_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads student retention dataset, handles missing values, and cleans target."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run src/fetch_dataset.py first.")

    try:
        df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(file_path)

    # Standardize column headers
    df.columns = [clean_col(c) for c in df.columns]
    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("[INFO] Imputing missing values...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        cat_cols = df.select_dtypes(include=["object"]).columns
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode()[0])
    else:
        print("[INFO] No missing values detected.")

    # Deduplication
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"[INFO] Removing {dupes} duplicate rows.")
        df = df.drop_duplicates()

    # Binary formulation: Dropout (1) vs. Non-Dropout (0)
    if "Target" in df.columns:
        df["Dropout_Risk"] = (df["Target"] == "Dropout").astype(int)

    return df

def generate_statistical_summaries(df: pd.DataFrame) -> None:
    """Prints critical statistical summaries highlighting UN SDG 4 relevance."""
    print("\n" + "=" * 65)
    print("      STATISTICAL SUMMARY: RETENTION & UN SDG 4 INDICATORS")
    print("=" * 65)

    total_students = len(df)
    dropout_count = df["Dropout_Risk"].sum()
    dropout_rate = (dropout_count / total_students) * 100

    print(f"Total Cohort Population    : {total_students:,}")
    print(f"Total Student Dropouts     : {dropout_count:,} ({dropout_rate:.2f}%)")
    print(f"Target 4.3 Retention Rate  : {(100 - dropout_rate):.2f}%")

    if "Tuition_fees_up_to_date" in df.columns:
        tuition_group = df.groupby("Tuition_fees_up_to_date")["Dropout_Risk"].agg(
            Count="count",
            Dropouts="sum",
            Dropout_Rate=lambda x: f"{x.mean()*100:.1f}%"
        )
        print("\n--- Attrition by Tuition Fee Standing (Financial Disparity) ---")
        print(tuition_group)

    if "Scholarship_holder" in df.columns:
        scholarship_group = df.groupby("Scholarship_holder")["Dropout_Risk"].agg(
            Count="count",
            Dropouts="sum",
            Dropout_Rate=lambda x: f"{x.mean()*100:.1f}%"
        )
        print("\n--- Attrition by Scholarship Status (Protective Safety Net) ---")
        print(scholarship_group)

    print("=" * 65 + "\n")

def plot_eda_visualizations(df: pd.DataFrame) -> None:
    """Generates 4 insightful visualizations and saves the figure."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("EduRetain AI: Student Retention Drivers (UN SDG 4: Quality Education)", fontsize=15, fontweight="bold", y=0.98)

    # 1. Target Status Distribution
    target_col = "Target" if "Target" in df.columns else "Dropout_Risk"
    sns.countplot(
        ax=axes[0, 0],
        x=target_col,
        hue=target_col,
        palette="Blues_r",
        data=df,
        legend=False
    )
    axes[0, 0].set_title("1. Overall Student Enrollment Status Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Academic Status")
    axes[0, 0].set_ylabel("Student Count")

    # 2. Financial Standing vs Dropout Heatmap
    if "Tuition_fees_up_to_date" in df.columns and "Scholarship_holder" in df.columns:
        pivot_fin = df.pivot_table(
            index="Tuition_fees_up_to_date",
            columns="Scholarship_holder",
            values="Dropout_Risk",
            aggfunc="mean"
        )
        sns.heatmap(
            pivot_fin * 100,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            ax=axes[0, 1],
            cbar_kws={"label": "Dropout Probability (%)"}
        )
        axes[0, 1].set_title("2. Dropout Risk by Financial Barriers (SDG 4.5)", fontweight="bold")
        axes[0, 1].set_xlabel("Scholarship Holder (0 = No, 1 = Yes)")
        axes[0, 1].set_ylabel("Tuition Up-to-Date (0 = No, 1 = Yes)")

    # 3. 1st Semester Course Completion Velocity
    curric_col = "Curricular_units_1st_sem_approved" if "Curricular_units_1st_sem_approved" in df.columns else [c for c in df.columns if "1st_sem" in c and "approved" in c][0]
    sns.boxplot(
        ax=axes[1, 0],
        x="Dropout_Risk",
        y=curric_col,
        hue="Dropout_Risk",
        palette="Set2",
        data=df,
        legend=False
    )
    axes[1, 0].set_title("3. 1st Semester Approved Units vs. Retention", fontweight="bold")
    axes[1, 0].set_xticklabels(["Retained / Graduated", "Dropout (At-Risk)"])
    axes[1, 0].set_ylabel("Approved Curricular Units (Sem 1)")

    # 4. Age at Enrollment Density Curve
    age_col = [c for c in df.columns if "age" in c.lower()][0]
    sns.kdeplot(
        ax=axes[1, 1],
        data=df,
        x=age_col,
        hue="Dropout_Risk",
        common_norm=False,
        fill=True,
        palette="mako",
        alpha=0.4
    )
    axes[1, 1].set_title("4. Age at Enrollment Density (Non-Traditional Students)", fontweight="bold")
    axes[1, 1].set_xlabel("Age at Enrollment (Years)")
    axes[1, 1].set_ylabel("Probability Density")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"[SUCCESS] High-resolution EDA figure saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    data = load_and_clean_data()
    generate_statistical_summaries(data)
    plot_eda_visualizations(data)
