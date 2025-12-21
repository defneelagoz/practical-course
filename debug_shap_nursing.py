import pandas as pd
import numpy as np
import joblib
import shap
import os
import sys

# Load Model
model_path = "predictive_model/gam/gam_model_student2_real5fold.joblib"
data_path = "predictive_model/gam/student_data_2.csv"

if not os.path.exists(model_path):
    print("Model not found.")
    sys.exit()

artifact = joblib.load(model_path)
model = artifact["model"]
preprocessor = artifact["preprocess"]

# Load Data
df = pd.read_csv(data_path, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
df.columns = clean_cols

# Clean numerics exactly as before
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

# Prepare X
target_col = "Output" # or Target
if target_col not in df.columns:
    target_col = "Target"
X = df.drop(columns=[target_col])

# Select Student 305
student_row = X.loc[[305]]
print("\nStudent 305 Course:", student_row["Course"].values[0])
print("Student 305 App Mode:", student_row["Application_mode"].values[0])

# Transform
X_pre = preprocessor.transform(X).astype(float)
student_pre = preprocessor.transform(student_row).astype(float)

# Background for SHAP
background = shap.kmeans(X_pre, 10)
explainer = shap.KernelExplainer(model.predict_proba, background)

# Calculate SHAP
print("\nCalculating SHAP...")
shap_vals = explainer.shap_values(student_pre)

# Get Feature Names
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = [f"feat_{i}" for i in range(X_pre.shape[1])]

# Find 'Course_Nursing' feature (Course 12)
# Note: OneHot might name it "cat__Course_12.0" or similar
nursing_feats = [f for f in feature_names if "Course" in f and "12" in f]
print("\nNursing Features:", nursing_feats)

# Print SHAP for Nursing
print("\n--- SHAP Values for Student 305 ---")
if isinstance(shap_vals, list):
    # prob for class 1 (Dropout?) 
    # Check classes
    classes = artifact["label_encoder"].classes_
    dropout_idx = list(classes).index("Dropout")
    sv = shap_vals[dropout_idx][0]
else:
    sv = shap_vals[0]

for f in nursing_feats:
    idx = list(feature_names).index(f)
    val = student_pre[0][idx]
    impact = sv[idx]
    print(f"Feature: {f}")
    print(f"  Input Value: {val} (Should be 0 for non-nursing)")
    print(f"  SHAP Impact: {impact} (Positive=Bad, Negative=Good)")
    
    # Check expected value (Baseline)
    # The expected value is explainer.expected_value[dropout_idx]
    
print(f"\nBaseline Expected Value: {explainer.expected_value[dropout_idx]}")
