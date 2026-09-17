"""
EduRetain AI - Student Retention & Dropout Prediction
Web Application Styled like KingShivamX/energy-consumption
Supports both execution methods:
1. Flask:     python app.py             (Opens http://127.0.0.1:5000)
2. Streamlit: streamlit run app.py     (Opens http://localhost:8501)
"""

import os
import io
import base64
import threading
import webbrowser
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_dropout_pipeline.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Load Dataset
df = pd.read_csv(DATA_PATH)
if "Target" in df.columns and "Dropout_Risk" not in df.columns:
    df["Dropout_Risk"] = (df["Target"] == "Dropout").astype(int)

# Load Scikit-Learn Pipeline
pipeline = None
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)


def generate_plot(approved_units, predicted_risk):
    """
    Generates an executive, publication-grade Matplotlib retention analysis plot.
    Features:
    - Distinct Retention Risk Zones (High Attrition, Moderate Risk, Stable Retention).
    - Historical Cohort Dropout Baseline Curve showing empirical dropout rate
      by First-Year Approved Units from the 4,424-student dataset.
    - Dynamic Predicted Student marker with custom crosshair reference lines.
    - Clean callout annotation card without deprecated/unsupported parameters.
    - Explicit white background preventing browser dark-mode distortion.
    """
    curric_col = [c for c in df.columns if "1st" in c.lower() and "approved" in c.lower()]
    curric_col_name = curric_col[0] if curric_col else "Curricular_units_1st_sem_approved"

    # Calculate empirical cohort retention baseline from historical records
    ct = df.groupby(curric_col_name)["Dropout_Risk"].agg(["count", "mean"]).reset_index()
    # Filter to statistically significant unit counts (at least 5 records and up to 20 units)
    ct = ct[ct["count"] >= 5]
    ct = ct[ct[curric_col_name] <= 20].sort_values(curric_col_name)

    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # 1. Shaded Risk Zones
    ax.axhspan(65, 100, color="#FEE2E2", alpha=0.55, label="High Risk Zone (>65%)", zorder=1)
    ax.axhspan(35, 65, color="#FEF3C7", alpha=0.55, label="Moderate Risk Zone (35-65%)", zorder=1)
    ax.axhspan(0, 35, color="#DCFCE7", alpha=0.55, label="Stable Retention Zone (<35%)", zorder=1)

    # 2. Historical Cohort Retention Trendline
    ax.plot(
        ct[curric_col_name],
        ct["mean"] * 100,
        color="#B45309",
        marker="o",
        linewidth=2.8,
        markersize=6.5,
        label="Cohort Historical Dropout Baseline (%)",
        zorder=3
    )
    ax.fill_between(ct[curric_col_name], ct["mean"] * 100, color="#F59E0B", alpha=0.15, zorder=2)

    # 3. Target Student Prediction Marker
    student_color = "#DC2626" if predicted_risk >= 65 else ("#D97706" if predicted_risk >= 35 else "#1E40AF")
    ax.scatter(
        [approved_units],
        [predicted_risk],
        color=student_color,
        s=220,
        zorder=6,
        edgecolors="#FFFFFF",
        linewidths=2.5,
        label="Predicted Student Outcome"
    )

    # 4. Reference Crosshair Lines
    ax.axhline(
        y=predicted_risk,
        color=student_color,
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label=f"Predicted Risk Level ({predicted_risk:.1f}%)"
    )
    ax.axvline(
        x=approved_units,
        color="#2563EB",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label=f"Approved Courses ({approved_units} Units)"
    )

    # 5. Smart Dynamic Callout Annotation (avoids edge clipping)
    x_offset = -4.5 if approved_units > 11 else 0.8
    y_offset = -14 if predicted_risk > 75 else 12
    ax.annotate(
        f"Student Outcome:\n{predicted_risk:.1f}% Risk | {approved_units} Units Passed",
        xy=(approved_units, predicted_risk),
        xytext=(approved_units + x_offset, predicted_risk + y_offset),
        arrowprops=dict(facecolor=student_color, edgecolor=student_color, arrowstyle="->", lw=1.8),
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec=student_color, lw=1.5),
        zorder=7
    )

    # Styling & Labels
    ax.set_xlabel("First Year Approved Curricular Units", fontsize=12, fontweight="bold", color="#111827", labelpad=8)
    ax.set_ylabel("Dropout Risk Probability (%)", fontsize=12, fontweight="bold", color="#111827", labelpad=8)
    ax.set_title("EduRetain AI: First-Year Course Velocity vs. Student Retention Risk", fontsize=13.5, fontweight="bold", pad=12, color="#0F172A")
    ax.set_xlim(-0.5, max(20.5, approved_units + 1.0))
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle=":", alpha=0.7, color="#CBD5E1", zorder=1)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

    img_data = io.BytesIO()
    plt.savefig(img_data, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    img_data.seek(0)
    plt.close(fig)

    return base64.b64encode(img_data.read()).decode("utf-8")


def predict_student(age, gender, tuition, scholarship, debtor, admission_grade, enrolled_units, approved_units, grade):
    """Calculates dropout risk probability, status, and intervention plan."""
    if pipeline is not None and hasattr(pipeline, "feature_names_in_"):
        expected_features = list(pipeline.feature_names_in_)
        base_series = df.drop(columns=["Target", "Dropout_Risk"], errors="ignore").median(numeric_only=True)
        input_dict = {col: [base_series.get(col, 0)] for col in expected_features}
        input_df = pd.DataFrame(input_dict)

        for col in expected_features:
            c_lower = col.lower()
            if "age" in c_lower and "enrollment" in c_lower: input_df[col] = age
            elif col == "Gender": input_df[col] = gender
            elif "scholarship" in c_lower: input_df[col] = scholarship
            elif col == "Debtor": input_df[col] = debtor
            elif "tuition" in c_lower: input_df[col] = tuition
            elif "admission" in c_lower: input_df[col] = admission_grade
            elif "1st" in c_lower and "enrolled" in c_lower: input_df[col] = enrolled_units
            elif "1st" in c_lower and "approved" in c_lower: input_df[col] = approved_units
            elif "1st" in c_lower and "grade" in c_lower: input_df[col] = grade
            elif "2nd" in c_lower and "enrolled" in c_lower: input_df[col] = enrolled_units
            elif "2nd" in c_lower and "approved" in c_lower: input_df[col] = approved_units
            elif "2nd" in c_lower and "grade" in c_lower: input_df[col] = grade

        prob = float(pipeline.predict_proba(input_df)[0][1]) * 100.0
    else:
        base_prob = 20.0
        if tuition == 0: base_prob += 45.0
        if debtor == 1: base_prob += 20.0
        if approved_units < (enrolled_units / 2): base_prob += 25.0
        if scholarship == 1: base_prob -= 15.0
        prob = float(np.clip(base_prob, 5.0, 95.0))

    if prob >= 65.0:
        risk_status = "HIGH ATTRITION RISK"
        intervention = "Initiate Emergency Bursary Review for Tuition Arrears & Mandatory 1-on-1 Tutoring"
    elif prob >= 35.0:
        risk_status = "MODERATE MONITORING"
        intervention = "Midterm Course Velocity Audit & Faculty Study Skills Workshop"
    else:
        risk_status = "STABLE RETENTION"
        intervention = "Standard Academic Progress & Career Development Mentorship"

    return prob, risk_status, intervention


# ================================================================
# CHECK EXECUTION ENVIRONMENT: STREAMLIT OR FLASK
# ================================================================
import streamlit as st

if st.runtime.exists():
    # STREAMLIT INTERFACE
    st.set_page_config(page_title="EduRetain AI", layout="centered")

    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, rgba(255, 191, 0, 0.9), rgba(255, 160, 0, 0.95));
            background-color: #FFBF00;
            color: #111111;
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
        }
        h1, h2, h3 {
            text-align: center;
            color: #111111 !important;
            font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
        }
        h1 { font-size: 2.3rem !important; }
        h3 { font-size: 1.5rem !important; margin-top: -10px; color: #382900 !important; }
        .stButton>button {
            background-color: #000000 !important;
            border: 2px solid #000000 !important;
            border-radius: 6px !important;
            color: #FFFFFF !important;
            width: 100% !important;
            height: 55px !important;
            font-size: 1.5rem !important;
            font-weight: bold !important;
            box-shadow: 4px 4px 8px rgba(0, 0, 0, 0.5) !important;
            font-family: inherit !important;
            transition: background-color 0.1s ease, color 0.1s ease !important;
        }
        .stButton>button p, .stButton>button span {
            color: #FFFFFF !important;
            font-weight: bold !important;
        }
        .stButton>button:hover {
            background-color: #222222 !important;
            color: #FFFFFF !important;
        }
        .stButton>button:hover p, .stButton>button:hover span {
            color: #FFFFFF !important;
        }
        .stButton>button:hover:active,
        .stButton>button:active,
        .stButton>button:focus:active,
        .stButton>button:focus {
            background-color: #FFBF00 !important;
            color: #000000 !important;
            border-color: #000000 !important;
            box-shadow: inset 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
        }
        .stButton>button:hover:active p, .stButton>button:hover:active span,
        .stButton>button:active p, .stButton>button:active span,
        .stButton>button:focus:active p, .stButton>button:focus:active span,
        .stButton>button:focus p, .stButton>button:focus span {
            color: #000000 !important;
        }
        .result-box {
            margin-top: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>EduRetain AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Enter The Student Values Correctly</h3>", unsafe_allow_html=True)

    with st.form("energy_style_form"):
        age = st.number_input("Enter Age at Enrollment:", min_value=17, max_value=65, value=20)
        st.caption("Enter student's age when enrolling in university (e.g., 18 to 45)")

        gender = st.selectbox("Select Gender:", options=[0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
        st.caption("Select Student Gender (0 for Female, 1 for Male)")

        tuition = st.selectbox("Tuition Fees Status (0 or 1):", options=[1, 0], format_func=lambda x: "1 - Up-to-Date (Paid)" if x == 1 else "0 - Overdue (In Arrears)")
        st.caption("Enter 1 if student's tuition fees are paid and up-to-date, else 0 if overdue")

        scholarship = st.selectbox("Scholarship Holder (0 or 1):", options=[0, 1], format_func=lambda x: "0 - No Scholarship" if x == 0 else "1 - Scholarship Recipient")
        st.caption("Enter 1 if student receives an academic/financial scholarship, else 0")

        debtor = st.selectbox("Declared Financial Debtor (0 or 1):", options=[0, 1], format_func=lambda x: "0 - No Unsettled Debt" if x == 0 else "1 - Declared Debtor")
        st.caption("Enter 1 if student has outstanding institutional debts, else 0")

        admission_grade = st.number_input("Admission Entrance Grade (0 to 200):", min_value=0.0, max_value=200.0, value=127.5, step=0.5)
        st.caption("Enter university entrance exam grade score (scale 0 to 200)")

        enrolled_units = st.number_input("First Year Enrolled Units:", min_value=0, max_value=20, value=6)
        st.caption("Total number of curricular courses enrolled in First Year (0 to 20)")

        approved_units = st.number_input("First Year Approved Units:", min_value=0, max_value=20, value=5)
        st.caption("Total number of curricular courses successfully passed in First Year (0 to 20)")

        grade = st.number_input("First Year Average Grade (0 to 20):", min_value=0.0, max_value=20.0, value=13.2, step=0.1)
        st.caption("Average grade scored in First Year (European scale 0 to 20)")

        submitted = st.form_submit_button("Predict Student Dropout Risk")

    if submitted:
        prob, risk_status, intervention = predict_student(age, gender, tuition, scholarship, debtor, admission_grade, enrolled_units, approved_units, grade)
        plot_b64 = generate_plot(approved_units, prob)

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown("<h2>Prediction Results:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.3rem;'>Input Age at Enrollment: {age} years</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.3rem;'>Tuition Fees Up-to-Date: {'1 - Yes (Paid)' if tuition == 1 else '0 - No (Overdue)'}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.3rem;'>Scholarship Holder: {'1 - Yes' if scholarship == 1 else '0 - No'}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.3rem;'>First Year Course Velocity: {approved_units} Approved / {enrolled_units} Enrolled</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.3rem;'>First Year Average Grade: {grade} / 20</p>", unsafe_allow_html=True)

        color_hex = "#C02626" if prob >= 65 else ("#C26100" if prob >= 35 else "#12892B")
        st.markdown(f"<h2 style='color: {color_hex} !important; text-decoration: underline;'>Predicted Student Dropout Risk: {prob:.2f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: {color_hex} !important;'>Retention Status: {risk_status}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: #1E3A8A !important; font-size: 1.5rem;'>Recommended Action: {intervention}</h2>", unsafe_allow_html=True)

        st.markdown("<h3>Matplotlib Graph Of Collected And Predicted Data</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center;'><img src='data:image/png;base64,{plot_b64}' style='border: 5px solid black; border-radius: 10px; max-width: 100%;'></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # FLASK INTERFACE
    from flask import Flask, render_template, request

    flask_app = Flask(__name__)

    @flask_app.route("/", methods=["GET"])
    def home():
        return render_template("index.html")

    @flask_app.route("/", methods=["POST"])
    def predict():
        try:
            age = int(request.form.get("age", 20))
            gender = int(request.form.get("gender", 0))
            tuition = int(request.form.get("tuition", 1))
            scholarship = int(request.form.get("scholarship", 0))
            debtor = int(request.form.get("debtor", 0))
            admission_grade = float(request.form.get("admission_grade", 127.5))
            enrolled_units = int(request.form.get("enrolled_units", 6))
            approved_units = int(request.form.get("approved_units", 5))
            grade = float(request.form.get("grade", 13.0))

            prob, risk_status, intervention = predict_student(age, gender, tuition, scholarship, debtor, admission_grade, enrolled_units, approved_units, grade)
            img_base64 = generate_plot(approved_units, prob)

            return render_template(
                "index.html",
                age=age,
                gender=gender,
                tuition=tuition,
                scholarship=scholarship,
                debtor=debtor,
                admission_grade=admission_grade,
                enrolled_units=enrolled_units,
                approved_units=approved_units,
                grade=grade,
                predicted_risk=round(prob, 2),
                risk_status=risk_status,
                intervention_action=intervention,
                plot=img_base64
            )
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return render_template("index.html", error=str(e))

    if __name__ == "__main__":
        print("[INFO] Starting Server on http://127.0.0.1:5000 ...")
        # Automatically launch web browser after 1 second so user is not left at terminal
        threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000/")).start()
        flask_app.run(debug=True, use_reloader=False, port=5000)
