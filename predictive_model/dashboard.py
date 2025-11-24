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
    # Paths
    model_path = os.path.join("predictive_model", "baseline_model.joblib")
    data_path = os.path.join("predictive_model", "student_data.csv")
    
    # Load
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please run the training script first.")
        return None, None
        
    model = joblib.load(model_path)
    df = pd.read_csv(data_path, sep=";", engine="python", encoding="utf-8-sig") # Adjust sep if needed
    
    # Clean columns like in the training script
    clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
    df.columns = clean_cols
    
    return model, df

model, df = load_resources()

if model is not None and df is not None:
    # Preprocessing for SHAP (Need X_train and X_test split logic or just use a sample)
    # For simplicity in this dashboard, we'll just use the whole dataset to select a student
    # In a real app, you'd have a separate test set or new data
    
    target_col = "Target" # Default
    
    # Explicitly check for known target names
    known_targets = ["Output", "Target", "Status", "Outcome"]
    for t in known_targets:
        if t in df.columns:
            target_col = t
            break
            
    if target_col not in df.columns:
         st.error(f"Could not find target column. Available columns: {list(df.columns)}")
         st.stop()
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
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
    # The model pipeline handles preprocessing
    probs = model.predict_proba(student_data)
    classes = model.named_steps['clf'].classes_
    
    # Find index of "Dropout" class
    dropout_idx = list(classes).index("Dropout") if "Dropout" in classes else 0
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
        st.dataframe(student_data)
        
        st.subheader("Model Explainability (SHAP)")
        with st.spinner("Calculating SHAP values..."):
            # Calculate SHAP for this student
            # Note: Passing a small sample of X as 'train' background for speed if needed
            # For better accuracy, we should pass the actual X_train used during training
            explainer, shap_vals, X_transformed, feature_names = calculate_shap_values(model, X.sample(100, random_state=42), student_data)
            
            # SHAP values for the specific class (Dropout)
            # TreeExplainer returns list of arrays for each class, or single array if binary
            # KernelExplainer returns list
            
            if isinstance(shap_vals, list):
                # Multi-class
                sv = shap_vals[dropout_idx]
            else:
                # Binary
                sv = shap_vals
                
            # Force Plot
            st.markdown("**Why did the model make this prediction?**")
            
            # Ensure sv is 1D array of feature impacts
            # sv might be (1, n_features) or just (n_features,)
            impact_values = np.array(sv)
            if impact_values.ndim > 1:
                impact_values = impact_values.flatten()
            
            # DEBUG: Check lengths
            if len(feature_names) != len(impact_values):
                msg = f"Shape mismatch! Features: {len(feature_names)}, Impacts: {len(impact_values)}"
                st.error(msg)
                print(msg, file=sys.stderr) # Print to console so I can see it
                st.write("Feature Names (first 5):", feature_names[:5])
                st.write("Impact Values shape:", impact_values.shape)
                st.stop()

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
