import pandas as pd
import numpy as np
import joblib
import shap
import os
import sys

# Load Model
model_path = "predictive_model/gam/gam_model_student2_real5fold.joblib"
if not os.path.exists(model_path):
    print("Model not found.")
    sys.exit()

artifact = joblib.load(model_path)
model = artifact["model"]
le = artifact["label_encoder"]
preprocessor = artifact["preprocess"]

print(f"Classes: {le.classes_}")

# Load Data
data_path = "predictive_model/gam/student_data_2.csv"
df = pd.read_csv(data_path, sep=";", decimal=",", engine="python", encoding="utf-8-sig")
clean_cols = [c.replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]
df.columns = clean_cols

# Clean numerics
for col in df.columns:
    if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
        try:
             series = df[col].astype(str).str.replace(',', '.', regex=False)
             temp = pd.to_numeric(series, errors='coerce')
             if temp.notna().sum() > 0.5 * len(temp): df[col] = temp
        except: pass
numeric_cols_temp = df.select_dtypes(include=[np.number]).columns
for c in numeric_cols_temp:
    mask_outliers = df[c] > 1e10 
    if mask_outliers.any(): df.loc[mask_outliers, c] = np.nan

X = df.drop(columns=["Output"] if "Output" in df.columns else ["Target"])

# Student 2
s2_row = X.loc[[2]]
print("\n--- Student 2 Data ---")
print(s2_row.T)

s2_pre = preprocessor.transform(s2_row).astype(float)
prob = model.predict_proba(s2_pre)
print(f"\nProbabilities: {prob}")

# Calculate SHAP
background = shap.kmeans(preprocessor.transform(X).astype(float), 10)
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_vals = explainer.shap_values(s2_pre, nsamples=50)

print("\n--- SHAP Raw Values (KernelExplainer) ---")
# Check structure
if isinstance(shap_vals, list):
    print(f"Returned list of length {len(shap_vals)}")
    # Assuming index 1 is Graduate? Or Index 0 Dropput?
    # Classes are [Dropout, Graduate] usually.
    # index 0 = Dropout.
    sv0 = shap_vals[0][0]
else:
    print("Returned single array")
    sv0 = shap_vals[0]

feature_names = preprocessor.get_feature_names_out()

target_feats = ["grade", "approved", "enrolled"]
for i, f in enumerate(feature_names):
    if any(t in f.lower() for t in target_feats) or "tuition" in f.lower():
        print(f"{f}: Val={s2_pre[0][i]:.2f}, SHAP={sv0[i]:.4f}")

