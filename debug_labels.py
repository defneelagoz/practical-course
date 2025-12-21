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

print("\n--- Label Encoder Classes ---")
print(f"Classes: {le.classes_}")
dropout_idx = list(le.classes_).index("Dropout")
print(f"Dropout Index: {dropout_idx}")

# Check Student 2 SHAP again
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
preprocessor = artifact["preprocess"]
X_pre = preprocessor.transform(X).astype(float)

# Student 2
s2_row = X.loc[[2]]
s2_pre = preprocessor.transform(s2_row).astype(float)
prob = model.predict_proba(s2_pre)
print(f"\nStudent 2 Probabilities: {prob}")
print(f"Prob for Dropout (Class {dropout_idx}): {prob[0][dropout_idx] if prob.ndim > 1 else '?'} (If >0.5, usually Risk)" if prob.ndim > 1 else prob)

# SHAP
background = shap.kmeans(X_pre, 10)
explainer = shap.KernelExplainer(model.predict_proba, background)
shap_vals = explainer.shap_values(s2_pre, nsamples=50) # fast approximation

print("\n--- SHAP Check ---")
if isinstance(shap_vals, list):
    sv_dropout = shap_vals[dropout_idx][0]
else:
    sv_dropout = shap_vals[0]

# Check Tuition Feature
# It's likely one-hot. Let's find columns with "Tuition"
feats = preprocessor.get_feature_names_out()
tuition_feats = [(i, f) for i, f in enumerate(feats) if "Tuition" in f]

for i, f in tuition_feats:
    val = s2_pre[0][i]
    impact = sv_dropout[i]
    if val > 0: # Active feature for this student
        print(f"Active Feature: {f}")
        print(f"  Value: {val}")
        print(f"  Impact on Dropout: {impact} (Expect Positive/Red for Tuition=0)")
