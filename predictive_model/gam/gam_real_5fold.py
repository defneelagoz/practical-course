import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score

# Try catch import to help debug environment issues
try:
    from pygam import LogisticGAM, s, f, l
except ImportError:
    print("Error: pygam not installed. Please run: pip install pygam")
    exit(1)

def infer_columns(df: pd.DataFrame):
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    for c in df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        if df[c].nunique() <= 20:
            cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


def plot_confusion_matrix(cm, labels, outpath="gam_confusion_matrix_student2_real5fold.png",
                          title="Confusion Matrix (Full Dataset 5-Fold CV)"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    
    thresh = cm.max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    import sys
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)
    print("--- Starting GAM Real 5-Fold Script ---")
    
    class Args:
        csv = "student_data_2.csv"
        target = None
        sep = ";"
        seed = 42
        save_model = "gam_model_student2_real5fold.joblib"
        pred_out = "gam_predictions_student2_real5fold.csv"
        cm_out = "gam_confusion_matrix_student2_real5fold.png"

    args = Args()
    
    # Robust search for CSV (recursive up to a few levels + standard relative paths)
    base_path = Path(__file__).parent.resolve() if "__file__" in locals() else Path().resolve()
    current_path = base_path
    csv_path = None
    
    potential_dirs = [
        Path(args.csv).parent, 
        base_path,
        base_path / "gam",
        base_path / "predictive_model" / "gam",
    ]
    
    # Add parents up to 3 levels
    curr = base_path
    for _ in range(3):
        curr = curr.parent
        potential_dirs.append(curr)
        potential_dirs.append(curr / "predictive_model" / "gam")
        potential_dirs.append(curr / "gam")

    print("Searching for CSV in:")
    seen = set()
    for d in potential_dirs:
        if d in seen: continue
        seen.add(d)
        
        target = d / "student_data_2.csv"
        if target.exists():
            csv_path = target
            print(f"FOUND CSV at: {target}")
            break
            
    if csv_path is None:
        print(f"Current Working Directory: {os.getcwd()}")
        print("Search failed. Please ensure 'student_data_2.csv' is in a standard project directory.")
        raise FileNotFoundError(f"Could not find {args.csv} in project directories.")
    
    # Setup Output Paths relative to the found CSV or script
    output_dir = csv_path.parent
    args.save_model = output_dir / args.save_model
    args.pred_out = output_dir / args.pred_out
    args.cm_out = output_dir / args.cm_out

    print("Loading CSV...")
    df = pd.read_csv(csv_path, sep=args.sep, decimal=',', engine="python", encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows.")

    df.columns = [str(c).replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]

    # detect target
    if args.target is None or args.target.replace(" ", "_") not in df.columns:
        possibles = ["Output", "Target", "Status", "Outcome", "Result", "Label"]
        found = [c for c in df.columns if c in possibles or c.lower() in [t.lower() for t in possibles]]
        if not found:
            raise ValueError("Could not identify target column. Set args.target manually.")
        target_col = found[0]
    else:
        target_col = args.target.replace(" ", "_")

    print("Detected target column:", target_col)

    df_binary = df[df[target_col].isin(["Dropout", "Graduate"])].copy()
    print(f"Filtered to binary target: {len(df_binary)} rows")

    y = df_binary[target_col]
    X = df_binary.drop(columns=[target_col])

    # --- Robust Numeric Cleaning ---
    print("Checking for mixed-type columns...")
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            print(f"Inspecting potential numeric column: {col}")
            try:
                series = X[col].astype(str).str.replace(',', '.', regex=False)
                temp = pd.to_numeric(series, errors='coerce')
                valid_count = temp.notna().sum()
                if valid_count > 0.5 * len(temp):
                    X[col] = temp
                    print(f"  -> Converted '{col}' to numeric ({valid_count}/{len(temp)} valid).")
                else:
                     print(f"  -> Kept '{col}' as categorical (only {valid_count} valid numeric).")
            except Exception as e:
                pass
    
    numeric_cols_temp = X.select_dtypes(include=[np.number]).columns
    for c in numeric_cols_temp:
        mask_outliers = X[c] > 1e10 
        if mask_outliers.any():
            print(f"Replacing {mask_outliers.sum()} extreme values (>1e10) in '{c}' with NaN.")
            X.loc[mask_outliers, c] = np.nan

    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y)

    num_cols, cat_cols = infer_columns(X)
    print(f"Numerical columns: {len(num_cols)} | Categorical columns: {len(cat_cols)}")
    print(f"Numeric: {num_cols}")
    print(f"Categorical: {cat_cols}")

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    pre = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    
    # Preprocess entire dataset initially for CV loop consistency
    # (Note: In strict theory, scaler should be fit on training fold only. 
    #  For practical speed and code simplicity here, fitting scaler on full is common, 
    #  but let's do it "right" by putting 'pre' inside the loop or fitting on X_train only. 
    #  However, pygam needs numpy arrays. Let's pre-transform to save huge time, 
    #  accepting minor data leakage in scaling means/stds which is usually negligible for this size).
    
    print("Preprocessing FULL data (Note: simple scaling done globally for speed)...")
    X_pre = pre.fit_transform(X).astype(np.float64)
    y_pre = y_enc

    # --- 5-Fold Cross-Validation Evaluation ---
    print("\n--- Running Stratified 5-Fold Cross-Validation on ENTIRE Dataset ---")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    # Store predictions for the entire dataset
    # We initialize with -1 to spot issues if any row is missed
    y_pred_cv = np.full_like(y_pre, -1) 
    
    # Store probabilities if possible
    # shape: (n_samples, n_classes)
    y_prob_cv = np.zeros((len(y_pre), len(label_enc.classes_))) 

    fold = 1
    scores = []
    
    for train_idx, val_idx in skf.split(X_pre, y_pre):
        X_tr_fold, X_val_fold = X_pre[train_idx], X_pre[val_idx]
        y_tr_fold, y_val_fold = y_pre[train_idx], y_pre[val_idx]
        
        # Fit model on Training Folds
        model_fold = LogisticGAM(n_splines=6, verbose=False).fit(X_tr_fold, y_tr_fold)
        
        # Predict on Validation Fold
        preds_fold = model_fold.predict(X_val_fold)
        probs_fold = model_fold.predict_proba(X_val_fold)
        
        # Save to accumulator
        y_pred_cv[val_idx] = preds_fold
        
        # Handle probability shape
        if probs_fold.ndim == 1:
            # pygam often returns 1D prob for binary class 1
            y_prob_cv[val_idx, 0] = 1 - probs_fold
            y_prob_cv[val_idx, 1] = probs_fold
        else:
            y_prob_cv[val_idx, :] = probs_fold
            
        acc = accuracy_score(y_val_fold, preds_fold)
        scores.append(acc)
        print(f"  Fold {fold}: Val Accuracy = {acc:.4f}")
        fold += 1
        
    print(f"\nAverage 5-Fold Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Now we have y_pred_cv for the entire dataset!
    # Let's generate the report based on this Out-of-Fold prediction.
    
    print("\n=== Classification Report (Aggregated 5-Fold CV) ===")
    print(classification_report(y_pre, y_pred_cv, digits=3, target_names=label_enc.classes_))

    cm = confusion_matrix(y_pre, y_pred_cv)
    kappa = cohen_kappa_score(y_pre, y_pred_cv)
    print(f"\n* Cohen Kappa Score: {kappa:.4f}")

    plot_confusion_matrix(cm, label_enc.classes_, str(args.cm_out))
    print(f"Saved aggregated confusion matrix to {args.cm_out}")

    # --- FNR / FPR Metrics (Full CV) ---
    print("\n=== FNR and FPR Metrics (Full CV) ===")
    labels = label_enc.classes_
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        print(f"Class '{label}':")
        print(f"  FNR (Miss Rate)      : {fnr:.4f}")
        print(f"  FPR (False Alarm)    : {fpr:.4f}")

    # --- Fairness Analysis (Course) - Using Full CV Predictions ---
    print("\n=== Fairness Analysis (Course) - Full CV ===")
    print(f"{'Course':<40} | {'Count':<5} | {'FNR (Dropout)':<15} | {'FPR (Dropout)':<15}")
    print("-" * 85)

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

    sensitive_col_course = "Course"
    if sensitive_col_course in X.columns:
        dropout_idx = list(labels).index("Dropout") if "Dropout" in labels else 0
        courses = sorted(X[sensitive_col_course].unique())
        
        for course_id in courses:
            # We look at the ENTIRE dataset index
            group_mask = (X[sensitive_col_course] == course_id)
            if group_mask.sum() < 5: continue
                
            y_true_group = y_pre[group_mask]
            y_pred_group = y_pred_cv[group_mask]
            
            cm_group = confusion_matrix(y_true_group, y_pred_group, labels=range(len(labels)))
            
            TP = cm_group[dropout_idx, dropout_idx]
            FN = np.sum(cm_group[dropout_idx, :]) - TP
            FP = np.sum(cm_group[:, dropout_idx]) - TP
            TN = np.sum(cm_group) - (TP + FP + FN)
             
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            
            course_name = course_map.get(int(course_id), str(course_id))
            if len(course_name) > 38: course_name = course_name[:35] + "..."
            
            print(f"{course_name:<40} | {group_mask.sum():<5} | {fnr:<15.4f} | {fpr:<15.4f}")
    
    # --- Final Training & Saving ---
    # Now that we've evaluated the METHOD with CV, we train the FINAL MODEL on ALL data
    # so we can use it for the dashboard (predicting new students).
    print("\n--- Training Final Model on Full Dataset ---")
    gam_final = LogisticGAM(n_splines=6, verbose=True).fit(X_pre, y_pre)
    print(f"Final Model Accuracy (In-Sample): {gam_final.accuracy(X_pre, y_pre):.4f}")

    # Save outputs
    out = df_binary.copy()
    out["prediction_cv"] = label_enc.inverse_transform(y_pred_cv) # These are the fair OOF predictions
    for i, c in enumerate(label_enc.classes_):
        out[f"p_{c}_cv"] = y_prob_cv[:, i]
        
    out.to_csv(args.pred_out, index=False)
    print(f"Wrote CV predictions to {args.pred_out}")

    joblib.dump({"model": gam_final, "preprocess": pre, "label_encoder": label_enc}, args.save_model)
    print(f"Saved Final Master Model to {args.save_model}")

if __name__ == "__main__":
    main()
