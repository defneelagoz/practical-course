import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import sys
import subprocess

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
st.set_page_config(page_title="Student Success Dashboard (v2 - 5Fold)", layout="wide")

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
    model_path = os.path.join(os.path.dirname(__file__), "gam", "gam_model_student2_real5fold.joblib")
    data_path = os.path.join(os.path.dirname(__file__), "gam", "student_data_2.csv")
    
    # Load
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run `predictive_model/gam/gam_real_5fold.py` first.")
        return None, None
        
    # The GAM model is saved as a dictionary: {"model": gam, "preprocess": pre, "label_encoder": label_enc}
    try:
        model_artifact = joblib.load(model_path)
    except Exception as e:
        # Auto-Recovery Logic
        st.warning(f"Model load failed ({e}). Attempting to auto-fix for your device...")
        
        try:
            # Locate the fix script
            fix_script = os.path.join(os.path.dirname(__file__), "gam", "gam_real_5fold.py")
            if not os.path.exists(fix_script):
                 st.error(f"Cannot find fix script at {fix_script}")
                 return None, None
            
            # Run it
            st.info("Retraining model with gam_real_5fold.py... This may take a moment.")
            process = subprocess.run([sys.executable, fix_script], capture_output=True, text=True)
            if process.returncode != 0:
                st.error(f"Auto-fix failed:\n{process.stderr}")
                return None, None
                
            st.success("Model optimized! Reloading...")
            model_artifact = joblib.load(model_path)
            
        except Exception as e2:
             st.error(f"Fatal error loading model: {e2}")
             return None, None

    if not isinstance(model_artifact, dict):
        st.error("Model file format unexpected. Expected a dictionary from gam_real_5fold.py.")
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

def clean_feature_name(name):
    """Cleans technical feature names (e.g. cat__Course_12) into readable text."""
    # Remove Sklearn/PyGAM prefixes
    name = name.replace("cat__", "").replace("num__", "")
    
    # Check for One-Hot suffix (usually _<number>)
    # But be careful with names that naturally end in numbers like "sem_1"
    # Usually one-hot encoding appends an underscore and the value.
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0].replace("_", " ")
        val = parts[1]
        
        # Optional: Try to map specific columns if known
        if "Course" in base:
            val = course_map.get(int(val), val)
        elif "Application mode" in base:
             val = application_mode_map.get(int(val), val)
             
        return f"{base}: {val}"
        
    return name.replace("_", " ")

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
    
    # Data Source Selection
    data_source = st.sidebar.radio("Data Source", ["Select Existing Student", "Simulate New Student"])

    if data_source == "Select Existing Student":
        # Student Selector
        student_index = st.sidebar.selectbox("Select Student Index", X.index)
        # Get selected student data
        student_data = X.loc[[student_index]]
    
    else:
        st.sidebar.markdown("### Simulate Student")
        student_index = "Simulated_User"
        
        # --- INPUT FORM ---
        # 1. Application Mode (Reverse Map for UI)
        # Create reverse map: "Name": ID
        app_mode_rev = {v: k for k, v in application_mode_map.items()}
        # Add any missing values from X that might not be in the map (just in case)
        # valid_app_modes = sorted(list(app_mode_rev.keys()))
        s_app_mode_label = st.sidebar.selectbox("Application Mode", options=list(app_mode_rev.keys()))
        s_app_mode = app_mode_rev[s_app_mode_label]

        # 2. Course
        course_rev = {v: k for k, v in course_map.items()}
        # Add generic options for ids 1-17 if map is incomplete, but we fixed it.
        s_course_label = st.sidebar.selectbox("Course", options=list(course_rev.keys()))
        s_course = course_rev[s_course_label]

        # 3. Tuition Fees
        s_tuition = st.sidebar.selectbox("Tuition fees up to date?", ["Yes", "No"])
        s_tuition_val = 1 if s_tuition == "Yes" else 0

        # 4. Age
        s_age = st.sidebar.number_input("Age at enrollment", min_value=17, max_value=70, value=20)

        # 5. Grades & Units (1st Sem)
        st.sidebar.markdown("#### 1st Semester")
        s_u1_enrolled = st.sidebar.number_input("Units Enrolled (1st)", 0, 20, 5)
        s_u1_approved = st.sidebar.number_input("Units Approved (1st)", 0, 20, 5)
        s_u1_grade = st.sidebar.number_input("Grade Avg (1st)", 0.0, 20.0, 14.0)

        # 6. Grades & Units (2nd Sem)
        st.sidebar.markdown("#### 2nd Semester")
        s_u2_enrolled = st.sidebar.number_input("Units Enrolled (2nd)", 0, 20, 5)
        s_u2_approved = st.sidebar.number_input("Units Approved (2nd)", 0, 20, 5)
        s_u2_grade = st.sidebar.number_input("Grade Avg (2nd)", 0.0, 20.0, 14.0)

        # Construct DataFrame with EXACT columns as X
        # We need to match the Training Data Columns exactly
        # Columns: ['Application mode', 'Course', 'Tuition fees up to date', 'Age at enrollment', 
        #           'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 
        #           'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)']
        
        sim_data = {
            'Application_mode': [s_app_mode],
            'Course': [s_course],
            'Tuition_fees_up_to_date': [s_tuition_val],
            'Age_at_enrollment': [s_age],
            'Curricular_units_1st_sem_(enrolled)': [s_u1_enrolled],
            'Curricular_units_1st_sem_(approved)': [s_u1_approved],
            'Curricular_units_1st_sem_(grade)': [s_u1_grade],
            'Curricular_units_2nd_sem_(enrolled)': [s_u2_enrolled],
            'Curricular_units_2nd_sem_(approved)': [s_u2_approved],
            'Curricular_units_2nd_sem_(grade)': [s_u2_grade]
        }
        student_data = pd.DataFrame(sim_data) # This will be used by the rest of the app
    
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

        # --- Representative Data Check ---
        # Get raw values for the selected student
        sel_course = student_data["Course"].iloc[0] if "Course" in student_data.columns else None
        
        # Handle App Mode name variation
        app_mode_col = "Application_mode" if "Application_mode" in student_data.columns else ("Application mode" if "Application mode" in student_data.columns else None)
        sel_app_mode = student_data[app_mode_col].iloc[0] if app_mode_col else None

        # Check counts in the FULL dataset (X)
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
                # Multi-class output from KernelExplainer
                if len(shap_vals) > dropout_idx:
                    sv = shap_vals[dropout_idx]
                else:
                    sv = shap_vals[0] # Fallback
            else:
                # Binary / single array
                # For PyGAM binary, this usually explains P(y=1) (e.g., Graduate)
                # If 'Dropout' is Class 0, we need to INVERT the SHAP values 
                # to explain P(Dropout).
                sv = shap_vals
                if probs.ndim == 1 and classes[1] != "Dropout":
                     # Model explains "Graduate". We want "Dropout".
                     # Invert the impact.
                     sv = -1 * sv
                
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

            # Retrieve the input values for this student
            # X_transformed is returned by calculate_shap_values
            # Ideally X_transformed corresponds to the X_test we passed (student_data)
            # which is 1 row.
            
            student_inputs = X_transformed[0] if X_transformed.ndim > 1 else X_transformed
            
            # Filter Logic:
            # We want to HIDE features that are:
            # 1. Categorical (Course, App Mode)
            # 2. AND have a value of 0 (False)
            
            filtered_data = []
            
            for i, name in enumerate(feature_names):
                clean_name = clean_feature_name(name)
                val = student_inputs[i] if i < len(student_inputs) else 0
                impact = impact_values[i]
                
                # Check criteria
                is_categorical_feature = ("Course" in clean_name or "Application mode" in clean_name or "Tuition" in clean_name)
                is_inactive = (abs(val) < 0.01) # effectively 0
                has_no_impact = (abs(impact) < 0.001) # effectively 0 impact

                if (is_categorical_feature and is_inactive) or has_no_impact:
                    continue # SKIP IT
                
                filtered_data.append({"Feature": clean_name, "Impact": impact})
            
            # Create a DataFrame for plotting
            shap_df = pd.DataFrame(filtered_data)
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
    print("Please run this app using: streamlit run predictive_model/dashboard2.py")
