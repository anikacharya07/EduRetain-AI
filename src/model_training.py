"""
EduRetain AI - Student Retention & Dropout Prediction (UN SDG 4)
Script: model_training.py
Purpose: Trains Logistic Regression and Random Forest models, performs
         rigorous evaluation (Accuracy, Precision, Recall, F1, ROC-AUC),
         saves evaluation plots (Confusion Matrix, Feature Importance),
         and exports the serialized Scikit-Learn pipeline to models/.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PKL = os.path.join(MODEL_DIR, "student_dropout_pipeline.pkl")
CM_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
FI_PNG = os.path.join(OUTPUT_DIR, "feature_importance.png")

sns.set_theme(style="whitegrid")

def build_and_evaluate_models(data_path: str = DATA_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run fetch_dataset.py first.")

    try:
        df = pd.read_csv(data_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(data_path)

    df.columns = [c.strip().replace(" ", "_").replace("'", "") for c in df.columns]

    # Target formulation (1 = Dropout, 0 = Non-Dropout)
    if "Target" in df.columns:
        y = (df["Target"] == "Dropout").astype(int)
        X = df.drop(columns=["Target", "Dropout_Risk"], errors="ignore")
    elif "Dropout_Risk" in df.columns:
        y = df["Dropout_Risk"]
        X = df.drop(columns=["Dropout_Risk"])
    else:
        raise ValueError("Target column not identified in dataset.")

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"[INFO] Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"[INFO] Categorical Features ({len(cat_cols)}): {cat_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Models
    models = {
        "Logistic Regression (Baseline)": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Random Forest Classifier": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42)),
        ]),
    }

    best_pipeline = None
    best_model_name = ""
    best_f1 = 0.0
    results_summary = []

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall (Sensitivity)": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        print(f"\n{'='*25} {name} {'='*25}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}  <-- Crucial for UN SDG 4 early intervention")
        print(f"F1-Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Retained", "Dropout"]))

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipe
            best_model_name = name

    # Serialize Best Pipeline
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PKL)
    print(f"\n[SUCCESS] Exported top-performing pipeline ({best_model_name}) to: {MODEL_PKL}")

    # 1. Confusion Matrix Generation
    y_best_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_best_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Retained / Graduated", "Dropout (At-Risk)"],
        yticklabels=["Retained / Graduated", "Dropout (At-Risk)"],
        cbar_kws={"label": "Normalized Ratio"}
    )
    plt.title(f"Confusion Matrix: {best_model_name}\n(Prioritizing Retention Recall for SDG 4)", fontweight="bold")
    plt.xlabel("Predicted Outcome")
    plt.ylabel("Actual Outcome")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=300)
    print(f"[SUCCESS] Saved confusion matrix to: {CM_PNG}")

    # 2. Feature Importance Plot (from Random Forest)
    rf_pipe = models["Random Forest Classifier"]
    rf_clf = rf_pipe.named_steps["classifier"]
    
    # Extract feature names from preprocessor
    feature_names = num_cols.copy()
    if cat_cols:
        cat_encoder = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_names)

    importances = rf_clf.feature_importances_
    fi_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)

    plt.figure(figsize=(9, 5.5))
    sns.barplot(x=fi_series.values, y=fi_series.index, palette="mako")
    plt.title("Top 10 Risk Drivers (Random Forest Feature Importance)", fontweight="bold")
    plt.xlabel("Gini Importance Score")
    plt.ylabel("Predictive Feature")
    plt.tight_layout()
    plt.savefig(FI_PNG, dpi=300)
    print(f"[SUCCESS] Saved feature importance plot to: {FI_PNG}")

    return pd.DataFrame(results_summary)

if __name__ == "__main__":
    build_and_evaluate_models()
