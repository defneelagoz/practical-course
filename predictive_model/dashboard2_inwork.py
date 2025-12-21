import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import subprocess

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Note: explainability imports removed

# --- Mappings ---
application_mode_map = {
    1: '1st phase - general contingent',
    2: 'Ordinance No. 612/93',
    3: '1st phase - special contingent (Azores Island)',
    4: 'Holders of other higher courses',
    5: 'Ordinance No. 854-B/99',
    6: 'International student (bachelor)',
    7: '1st phase - special contingent (Madeira Island)',
    8: '2nd phase - general contingent',
    9: '3rd phase - general contingent',
    10: 'Ordinance No. 533-A/99 (Different Plan)',
    11: 'Ordinance No. 533-A/99 (Other Institution)',
    12: 'Over 23 years old',
    13: 'Transfer',
    14: 'Change of course',
    15: 'Technological specialization diploma holders',
    16: 'Change of institution/course',
    17: 'Short cycle diploma holders',
    18: 'Change of institution/course (International)'
}

course_map = {
    1: 'Biofuel Production Technologies',
    2: 'Animation and Multimedia Design', 
    3: 'Social Service (evening attendance)',
    4: 'Agronomy',
    5: 'Communication Design',
    6: 'Veterinary Nursing',
    7: 'Informatics Engineering',
    8: 'Equinculture',
    9: 'Management',
    10: 'Social Service',
    11: 'Tourism',
    12: 'Nursing',
    13: 'Oral Hygiene',
    14: 'Advertising and Marketing Management',
    15: 'Journalism and Communication',
    16: 'Basic Education',
    17: 'Management (evening attendance)'
}

# Page Config
st.set_page_config(page_title="Student Success Dashboard (In-Work)", layout="wide")

# Title
st.title("🎓 Student Success Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Load Data & Model
@st.cache_resource
def load_resources():
    # Paths - Using the REAL 5-Fold Model
    model_path = os.path.join(os.path.dirname(__file__), "gam", "gam_model_student2_real5fold.joblib")
    data_path = os.path.join(os.path.dirname(__file__), "gam", "student_data_2.csv")
    
    # Load
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run `predictive_model/gam/gam_real_5fold.py` first.")
        return None, None
        
    try:
        model_artifact = joblib.load(model_path)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None

    if not isinstance(model_artifact, dict):
        st.error("Model file format unexpected.")
        return None, None

    # Load CSV
    if not os.path.exists(data_path):
        st.error(f"Data not found at {data_path}")
        return None, None

    df = pd.read_csv(data_path, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
    
    # Clean columns
    clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
    df.columns = clean_cols
    
    # Robust Numeric Cleaning
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            try:
                series = df[col].astype(str).str.replace(',', '.', regex=False)
                temp = pd.to_numeric(series, errors='coerce')
                valid_count = temp.notna().sum()
                if valid_count > 0.5 * len(temp):
                    df[col] = temp
            except:
                pass

    numeric_cols_temp = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols_temp:
        mask_outliers = df[c] > 1e10 
        if mask_outliers.any():
            df.loc[mask_outliers, c] = np.nan
            
    return model_artifact, df

model_artifact, df = load_resources()

if model_artifact is not None and df is not None:
    model = model_artifact["model"]
    preprocessor = model_artifact["preprocess"]
    label_encoder = model_artifact["label_encoder"]

    # Target Logic
    target_col = "Target"
    known_targets = ["Output", "Target", "Status", "Outcome"]
    for t in known_targets:
        if t in df.columns:
            target_col = t
            break
            
    if target_col not in df.columns:
         st.error(f"Could not find target column. Available columns: {list(df.columns)}")
         st.stop()
    
    X = df.drop(columns=[target_col])
    
    # --- Sidebar Filters ---
    st.sidebar.subheader("Filter Students")
    
    # Thresholds
    high_risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.05)
    low_risk_threshold = st.sidebar.slider("Safe Threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Student Selector (ONLY Existing Students)
    student_index = st.sidebar.selectbox("Select Student Index", X.index)
    student_data = X.loc[[student_index]]
    
    # Predict
    classes = None
    probs = None

    try:
        student_data_pre = preprocessor.transform(student_data).astype(float)
        probs = model.predict_proba(student_data_pre)
        classes = label_encoder.classes_
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()
    
    if classes is None or probs is None:
        st.error("Model prediction failed. Check logs.")
        st.stop()

    dropout_idx = list(classes).index("Dropout") if "Dropout" in classes else 0
    
    if probs.ndim == 1:
        p_1 = probs[0]
        dropout_prob = p_1 if classes[1] == "Dropout" else (1 - p_1)
    else:
        dropout_prob = probs[0][dropout_idx]
    
    # Display Status
    st.sidebar.markdown("### Prediction Status")
    if dropout_prob >= high_risk_threshold:
        st.sidebar.error(f"🚨 HIGH RISK ({dropout_prob:.1%})")
    elif dropout_prob <= low_risk_threshold:
        st.sidebar.success(f"✅ SAFE ({dropout_prob:.1%})")
    else:
        st.sidebar.warning(f"⚠️ MONITOR ({dropout_prob:.1%})")
        
    # --- Main Content ---
    st.subheader("Student Profile")

    # --- Representative Data Check ---
    sel_course = student_data["Course"].iloc[0] if "Course" in student_data.columns else None
    
    app_mode_col = "Application_mode" if "Application_mode" in student_data.columns else ("Application mode" if "Application mode" in student_data.columns else None)
    sel_app_mode = student_data[app_mode_col].iloc[0] if app_mode_col else None

    # Check counts
    if sel_course is not None:
        course_count = X[X["Course"] == sel_course].shape[0]
        if course_count < 50:
            st.warning(f"⚠️ Low Sample Size: This Course has only {course_count} students. Predictions may be less reliable.")

    if sel_app_mode is not None:
        app_count = X[X[app_mode_col] == sel_app_mode].shape[0]
        if app_count < 50:
            st.warning(f"⚠️ Low Sample Size: This Application Mode has only {app_count} students. Predictions may be less reliable.")

    # Create a display copy to show readable labels
    display_data = student_data.copy()
    if "Application_mode" in display_data.columns:
            display_data["Application_mode"] = display_data["Application_mode"].apply(
                lambda x: application_mode_map.get(int(x), x) if str(x).replace('.','').isdigit() else x
            )
    elif "Application mode" in display_data.columns:
            display_data["Application mode"] = display_data["Application mode"].apply(
                lambda x: application_mode_map.get(int(x), x) if str(x).replace('.','').isdigit() else x
            )
    
    if "Course" in display_data.columns:
            display_data["Course"] = display_data["Course"].apply(
                lambda x: course_map.get(int(x), x) if str(x).replace('.','').isdigit() else x
            )

    st.dataframe(display_data)

else:
    st.warning("Please ensure the model is trained and data is available.")

if __name__ == "__main__":
    print("WARNING: You are running this script directly with Python.")
    print("Please run this app using: streamlit run predictive_model/dashboard2_inwork.py")
