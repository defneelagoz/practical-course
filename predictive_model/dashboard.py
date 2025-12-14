import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainability import calculate_shap_values, generate_genai_explanation

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
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}

# Page Config
st.set_page_config(page_title="Student Success Dashboard", layout="wide")

# Title
st.title("🎓 Student Success & Dropout Risk Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Load Data & Model
@st.cache_resource
def load_resources():
    # Paths - Adjusted for GAM model
    # Expecting dashboard.py to be in predictive_model/, so gam/ is a subdir
    model_path = os.path.join(os.path.dirname(__file__), "gam", "gam_model_student2.joblib")
    data_path = os.path.join(os.path.dirname(__file__), "gam", "student_data_2.csv")
    
    # Load
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run the GAM training script first.")
        return None, None
        
    # The GAM model is saved as a dictionary: {"model": gam, "preprocess": pre, "label_encoder": label_enc}
    try:
        model_artifact = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

    if not isinstance(model_artifact, dict):
        st.error("Model file format unexpected. Expected a dictionary from gam_fix.py.")
        return None, None

    # Load CSV with robust handling similar to gam_fix.py
    if not os.path.exists(data_path):
        st.error(f"Data not found at {data_path}")
        return None, None

    df = pd.read_csv(data_path, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
    
    # Clean columns
    clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
    df.columns = clean_cols
    
    # --- Robust Numeric Cleaning (Mirroring gam_fix.py) ---
    # The dashboard needs to clean raw data exactly like the training script
    # otherwise the Imputer will fail on string values like '1,34286E+16'
    
    # 1. Identify and fix string-numeric columns
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            try:
                # Replace comma with dot and coerce to numeric
                series = df[col].astype(str).str.replace(',', '.', regex=False)
                temp = pd.to_numeric(series, errors='coerce')
                
                # Check if it looks numeric (heuristic: >50% valid)
                valid_count = temp.notna().sum()
                if valid_count > 0.5 * len(temp):
                    df[col] = temp
            except:
                pass

    # 2. Clean extreme outliers (>1e10)
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
    # We don't perform extensive cleaning here because the preprocessor in the artifact handles it
    # BUT we need to ensure formatting matches what preprocessor expects (e.g. comma decimals if raw)
    # pd.read_csv handled decimal=',' above, so dtypes should be mostly correct.
    
    # --- Sidebar Filters ---
    st.sidebar.subheader("Filter Students")
    
    # Thresholds
    high_risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.05)
    low_risk_threshold = st.sidebar.slider("Safe Threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Student Selector
    student_index = st.sidebar.selectbox("Select Student Index", X.index)
    
    # Get selected student data
    student_data = X.loc[[student_index]]
    
    # Predict
    # GAM needs preprocessed data
    classes = None  # Safe initialization
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

    # Find index of "Dropout" class
    dropout_idx = list(classes).index("Dropout") if "Dropout" in classes else 0
    # Handle prob shape (n_samples, n_classes) or (n_samples,) if binary
    if probs.ndim == 1:
        # Binary case for pygam often returns just P(y=1)
        # We need to check label encoder to see which is 1
        # Typically 1 is the second class in classes_
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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Student Profile")
        
        # Create a display copy to show readable labels
        display_data = student_data.copy()
        if "Application_mode" in display_data.columns:
             # Check if numeric to map, handle strings if already mapped
             display_data["Application_mode"] = display_data["Application_mode"].apply(
                 lambda x: application_mode_map.get(int(x), x) if str(x).replace('.','').isdigit() else x
             )
        # Fallback if column name has spaces
        elif "Application mode" in display_data.columns:
             display_data["Application mode"] = display_data["Application mode"].apply(
                 lambda x: application_mode_map.get(int(x), x) if str(x).replace('.','').isdigit() else x
             )
        
        # Course Map
        if "Course" in display_data.columns:
             display_data["Course"] = display_data["Course"].apply(
                 lambda x: course_map.get(int(x), x) if str(x).replace('.','').isdigit() else x
             )

        st.dataframe(display_data)
        
        st.subheader("Model Explainability (SHAP)")
        with st.spinner("Calculating SHAP values..."):
            # Pass the dictionary artifact to calculate_shap_values so it can handle the logic
            # Use a sample of X for background
            explainer, shap_vals, X_transformed, feature_names = calculate_shap_values(
                model_artifact, 
                X.sample(min(100, len(X)), random_state=42), 
                student_data
            )
            
            # SHAP values for the specific class (Dropout)
            if isinstance(shap_vals, list):
                # Multi-class
                # If pygam returns list, check length
                if len(shap_vals) > dropout_idx:
                    sv = shap_vals[dropout_idx]
                else:
                    sv = shap_vals[0] # Fallback
            else:
                # Binary / single array
                sv = shap_vals
                
            # Force Plot / Bar Plot logic
            st.markdown("**Why did the model make this prediction?**")
            
            # Ensure sv is 1D array of feature impacts
            impact_values = np.array(sv)
            if impact_values.ndim > 1:
                impact_values = impact_values.flatten()
            
            # Check lengths
            # For GAMs/OneHot, feature_names might be huge.
            if len(feature_names) != len(impact_values):
                # Re-align - this can happen if variable inputs
                # Try to use just the top ones or handle mismatched shapes gracefully
                st.warning(f"Shape mismatch (feats={len(feature_names)}, impact={len(impact_values)}). Showing raw impacts.")
                feature_names = [f"Feature {i}" for i in range(len(impact_values))]

            # Create a DataFrame for plotting
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": impact_values
            })
            shap_df["AbsImpact"] = shap_df["Impact"].abs()
            shap_df = shap_df.sort_values("AbsImpact", ascending=False).head(10)
            
            fig, ax = plt.subplots()
            colors = ['red' if x > 0 else 'blue' for x in shap_df['Impact']]
            ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors)
            ax.set_xlabel("Impact on Dropout Risk (SHAP Value)")
            st.pyplot(fig)
            
    with col2:
        st.subheader("🤖 GenAI Insight")
        
        # Prepare data for GenAI
        top_features = list(zip(shap_df['Feature'], shap_df['Impact']))
        
        explanation = generate_genai_explanation(student_index, dropout_prob, top_features)
        
        st.info(explanation)
        
        st.markdown("### Counselor Actions")
        action = st.selectbox("Recommended Action", ["None", "Send Email", "Schedule Meeting", "Refer to Tutor"])
        
        if st.button("Confirm Action"):
            st.success(f"Action '{action}' recorded for Student {student_index}.")

else:
    st.warning("Please ensure the model is trained and data is available.")

if __name__ == "__main__":
    print("WARNING: You are running this script directly with Python.")
    print("Please run this app using: streamlit run predictive_model/dashboard.py")
