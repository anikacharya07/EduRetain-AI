"""
EduRetain AI - Student Retention Analytics (UN SDG 4)
Script: generate_presentation.py
Purpose: Generates a professional 10-slide PowerPoint (.pptx) presentation
         for the IBM Data Analytics Internship final review panel.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PPTX = os.path.join(BASE_DIR, "EduRetain_AI_Presentation.pptx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

EDA_IMG = os.path.join(OUTPUT_DIR, "student_eda_insights.png")
CM_IMG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
FI_IMG = os.path.join(OUTPUT_DIR, "feature_importance.png")

# Brand Colors (IBM Deep Blue & UN SDG 4 Crimson/Red accents)
COLOR_PRIMARY = RGBColor(15, 23, 42)      # Slate 900
COLOR_ACCENT = RGBColor(30, 58, 138)      # Blue 900
COLOR_HIGHLIGHT = RGBColor(220, 38, 38)   # Red 600 (SDG 4)
COLOR_TEXT = RGBColor(51, 65, 85)         # Slate 700
COLOR_MUTED = RGBColor(100, 116, 139)     # Slate 500

def create_deck():
    prs = Presentation()
    # 16:9 widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="IBM DATA ANALYTICS INTERNSHIP | UN SDG 4: QUALITY EDUCATION"):
        # Header category
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_HIGHLIGHT

        # Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_ACCENT

    # SLIDE 1: Title & Executive Summary
    slide1 = prs.slides.add_slide(blank_layout)
    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "EduRetain AI: Early-Warning Student Retention Analytics"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    p_sub = tf1.add_paragraph()
    p_sub.text = "Predicting Student Dropout Risk Aligned with UN SDG 4 (Quality Education)"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = COLOR_HIGHLIGHT
    p_sub.space_before = Pt(12)

    p_meta = tf1.add_paragraph()
    p_meta.text = "Author: Data Analytics Intern | IBM Data Analytics Internship Capstone Project"
    p_meta.font.size = Pt(13)
    p_meta.font.color.rgb = COLOR_MUTED
    p_meta.space_before = Pt(36)

    # SLIDE 2: Problem Statement & SDG 4
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Global Tertiary Education Attrition Challenge")
    tb2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        "The Crisis: Globally, 25% to 35% of higher education students drop out before obtaining a qualification.",
        "UN SDG Target 4.3: Ensure equal access for all women and men to affordable and quality technical, vocational, and tertiary education.",
        "UN SDG Target 4.5: Eliminate gender and socio-economic disparities in education to ensure access for the vulnerable.",
        "Core Bottleneck: Universities operate on reactive exit interviews rather than proactive early intervention.",
        "Solution Mandate: Construct an early-warning ML diagnostic engine to intervene in the first 6 weeks of enrollment."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(14)

    # SLIDE 3: Dataset Architecture
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Dataset Architecture & Preprocessing Pipeline")
    tb3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    bullets3 = [
        "Benchmark Source: Polytechnic Institute of Portalegre & UC Irvine Repository (4,424 records, 36 features).",
        "Demographic Spectrum: Age at enrollment, gender, nationality, special educational accommodations.",
        "Socio-Economic Variables: Scholarship status, debtor records, tuition fee standing (up-to-date vs overdue).",
        "Academic Velocity: Enrolled vs. evaluated vs. approved units and grade averages across Semesters 1 and 2.",
        "Preprocessing Pipeline: Scikit-Learn ColumnTransformer using StandardScaler for continuous features and OneHotEncoder for categorical features with stratified 80/20 train/test partitioning."
    ]
    for b in bullets3:
        p = tf3.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(14)

    # SLIDE 4: EDA Academic Factors
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Exploratory Insights: Academic Velocity as an Early Signal")
    tb4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.0))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    bullets4 = [
        "First-Semester Course Velocity: Single strongest academic predictor of retention.",
        "The Critical Tipping Point: Students completing <50% of enrolled units in Semester 1 show an 81% probability of eventual departure.",
        "Grade Point Dynamics: Grade averages below 11.5/20 mark a steep rise in withdrawal likelihood.",
        "Intervention Window: Early course check-ins at Week 6 provide the greatest preservation of retention."
    ]
    for b in bullets4:
        p = tf4.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(12)
    if os.path.exists(EDA_IMG):
        slide4.shapes.add_picture(EDA_IMG, Inches(7.0), Inches(1.6), width=Inches(5.6))

    # SLIDE 5: EDA Financial Factors
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Exploratory Insights: Financial Distress as an Attrition Multiplier")
    tb5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    bullets5 = [
        "Tuition Fee Arrears: Students with overdue fees exhibit an 86.6% dropout rate vs 24.7% for accounts in good standing.",
        "The Scholarship Safety Buffer: Scholarship holders experience an attrition rate of only 12.2% vs 38.7% for non-holders.",
        "Non-Traditional Learner Vulnerability: Students enrolling above age 24 demonstrate elevated attrition due to work-study friction.",
        "UN SDG 4.5 Mandate: Financial distress requires automated institutional bursary triggers, not punitive academic suspensions."
    ]
    for b in bullets5:
        p = tf5.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(14)

    # SLIDE 6: Modeling Strategy
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Predictive Modeling Strategy & Algorithmic Design")
    tb6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    bullets6 = [
        "Problem Framing: Binary Classification (1 = At-Risk Dropout, 0 = Retained / Graduated).",
        "Social Metric Prioritization: In educational equity, Recall (Sensitivity) outweighs Precision to minimize False Negatives (unassisted dropouts).",
        "Model 1 (Baseline): Balanced Logistic Regression for transparent policy coefficients and odds ratio calculation.",
        "Model 2 (Optimized Ensemble): Random Forest Classifier (200 trees, balanced weights) capturing non-linear feature interactions.",
        "Validation Strategy: Stratified 5-fold cross-validation preventing target imbalance bias."
    ]
    for b in bullets6:
        p = tf6.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(14)

    # SLIDE 7: Evaluation Results
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Comparative Performance & Confusion Matrix Analysis")
    tb7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.0))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    bullets7 = [
        "Random Forest Benchmark: Accuracy: 88.36% | ROC-AUC: 0.9328 | F1-Score: 0.8151.",
        "Logistic Regression Baseline: Accuracy: 87.34% | ROC-AUC: 0.9272 | Recall: 83.10%.",
        "False Negative Containment: Both models successfully identify >80% of at-risk students.",
        "Operational Production Decision: Serialized top Random Forest pipeline to models/student_dropout_pipeline.pkl."
    ]
    for b in bullets7:
        p = tf7.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(12)
    if os.path.exists(CM_IMG):
        slide7.shapes.add_picture(CM_IMG, Inches(7.2), Inches(1.6), width=Inches(5.3))

    # SLIDE 8: Feature Importance
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Key Risk Drivers: What Matters Most in Retention?")
    tb8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(5.8), Inches(5.0))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    bullets8 = [
        "1. Curricular Units 2nd Sem Approved: Immediate academic persistence indicator.",
        "2. Curricular Units 1st Sem Approved: Early velocity benchmark.",
        "3. Tuition Fees Up-to-Date: Decisive socio-economic solvency gate.",
        "4. Curricular Units 1st Sem Grade: Academic mastery score.",
        "5. Age at Enrollment: Marker for non-traditional student responsibilities."
    ]
    for b in bullets8:
        p = tf8.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(12)
    if os.path.exists(FI_IMG):
        slide8.shapes.add_picture(FI_IMG, Inches(6.8), Inches(1.6), width=Inches(5.8))

    # SLIDE 9: Streamlit Dashboard
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Operational Solution: EduRetain AI Web Dashboard")
    tb9 = slide9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    bullets9 = [
        "Counselor-Centric Design: Built with Streamlit (app.py) for instantaneous advisor adoption.",
        "Executive Dashboard Tab: Live tracking of cohort attrition rates and SDG 4 target health.",
        "Interactive EDA Explorer: Instant drill-downs into financial and demographic cohorts.",
        "Real-Time Student Screener: Input student indicators to output risk percentage and risk badge (High / Medium / Low).",
        "Automated Intervention Protocol: Generates personalized academic action plans (Emergency Bursaries, 1-on-1 Tutoring, Peer Mentoring)."
    ]
    for b in bullets9:
        p = tf9.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(14)

    # SLIDE 10: Conclusion & Recommendations
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Policy Recommendations & UN SDG 4 Impact")
    tb10 = slide10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.5), Inches(5.0))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    bullets10 = [
        "1. Automated Bursary Flags: Trigger emergency financial aid consultations when tuition falls behind, avoiding punitive deregistration.",
        "2. Week-6 Midterm Check-Ins: Implement mandatory advising for any student completing <50% of first-semester modules.",
        "3. Flexible Adult Learning Tracks: Provide hybrid and evening structures for students aged 25+ to reconcile work and studies.",
        "Conclusion: EduRetain AI transforms institutional response from passive exit metrics into proactive educational preservation, fulfilling the promise of UN SDG 4."
    ]
    for b in bullets10:
        p = tf10.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(15)
        p.font.color.rgb = COLOR_TEXT
        p.space_after = Pt(14)

    prs.save(OUTPUT_PPTX)
    print(f"[SUCCESS] PowerPoint presentation saved to: {OUTPUT_PPTX}")

if __name__ == "__main__":
    create_deck()
