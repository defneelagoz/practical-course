import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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


def plot_confusion_matrix(cm, labels, outpath="gam_confusion_matrix_student2_5fold.png",
                          title="Confusion Matrix – GAM (student_data_2) 5-Fold"):
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
    print("--- Starting GAM Fix Script using 5-Fold CV ---")
    
    class Args:
        csv = "student_data_2.csv"
        target = None
        sep = ";"
        test_size = 0.2
        seed = 42
        save_model = "gam_model_student2_5fold.joblib"
        pred_out = "gam_predictions_student2_5fold.csv"
        cm_out = "gam_confusion_matrix_student2_5fold.png"

    args = Args()
    
    # Robust search for CSV (recursive up to a few levels + standard relative paths)
    base_path = Path(__file__).parent.resolve() if "__file__" in locals() else Path().resolve()
    current_path = base_path
    csv_path = None
    
    # Search locations:
    # 1. Explicit arg path
    # 2. Base path (script dir)
    # 3. Base path / gam
    # 4. Climbing up parent directories
    
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
        # print(f"  Checking: {target}") # Debug only if needed
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
    # Use decimal=',' for European format
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
    # Identify likely numeric columns that might have been read as objects due to formatting issues
    print("Checking for mixed-type columns...")
    for col in X.columns:
        # Check if it's currently object/category
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            print(f"Inspecting potential numeric column: {col}")
            # Try to convert to numeric, coercing errors
            # CRITICAL FIX: explicitly replace ',' with '.' for string parsing if pd.read_csv missed it
            try:
                # We work on a copy to avoid SettingWithCopy warnings if any
                series = X[col].astype(str).str.replace(',', '.', regex=False)
                # Force coerce
                temp = pd.to_numeric(series, errors='coerce')
                
                # Heuristic: If >50% are valid numbers
                valid_count = temp.notna().sum()
                if valid_count > 0.5 * len(temp):
                    X[col] = temp
                    print(f"  -> Converted '{col}' to numeric ({valid_count}/{len(temp)} valid).")
                else:
                     print(f"  -> Kept '{col}' as categorical (only {valid_count} valid numeric).")
            except Exception as e:
                print(f"  -> Failed to convert '{col}': {e}")
                pass
    
    # Clean extreme outliers 
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

    # Use a pipeline for numeric columns to impute THEN scale
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.seed, stratify=y_enc
    )

    print("Preprocessing data...")
    X_train_pre = pre.fit_transform(X_train).astype(np.float64)
    X_test_pre = pre.transform(X_test).astype(np.float64)
    print(f"Training Data Shape: {X_train_pre.shape}")

    # --- Step 1: Simple Fit (Sanity Check) ---
    print("\n--- Step 1: Running Simple Fit (Sanity Check) ---")
    t0 = time.time()
    try:
        # Reduced splines for safety and speed
        print("Fitting on component subset (first 100 rows)...")
        gam_subset = LogisticGAM(n_splines=5).fit(X_train_pre[:100], y_train[:100])
        print("Subset fit successful.")

        print("Fitting on full training set (no grid search)...")
        gam_simple = LogisticGAM(n_splines=6, verbose=True).fit(X_train_pre, y_train)
        print(f"Simple fit successful in {time.time() - t0:.2f} seconds.")
        print(f"Simple Fit Accuracy: {gam_simple.accuracy(X_test_pre, y_test):.4f}")
    except Exception as e:
        print(f"WARNING: Simple fit failed! Error: {e}") 
        import traceback
        traceback.print_exc()
    
    gam = gam_simple
    
    # --- Explicit 5-Fold Validation Step ---
    print("\n--- Explicit 5-Fold Cross-Validation on Training Set ---")
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_scores = []
        
        fold = 1
        for train_idx, val_idx in skf.split(X_train_pre, y_train):
            X_tr_fold, X_val_fold = X_train_pre[train_idx], X_train_pre[val_idx]
            y_tr_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            # Fit new instance
            gam_fold = LogisticGAM(n_splines=6, verbose=False).fit(X_tr_fold, y_tr_fold)
            acc = gam_fold.accuracy(X_val_fold, y_val_fold)
            cv_scores.append(acc)
            print(f"  Fold {fold}: Accuracy = {acc:.4f}")
            fold += 1
            
        print(f"5-Fold CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    except Exception as e:
         print(f"WARNING: 5-Fold CV failed! Error: {e}")
         import traceback
         traceback.print_exc()

    y_pred = gam.predict(X_test_pre)
    print("\n=== Classification Report (test set) ===")
    print(classification_report(y_test, y_pred, digits=3, target_names=label_enc.classes_))

    cm = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"\n* Cohen Kappa Score: {kappa:.4f}")

    plot_confusion_matrix(cm, label_enc.classes_, str(args.cm_out))
    print(f"Saved confusion matrix to {args.cm_out}")

    # --- FNR / FPR Metrics ---
    print("\n=== FNR and FPR Metrics ===")
    labels = label_enc.classes_
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        # Avoid division by zero
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        print(f"Class '{label}':")
        print(f"  FNR (Miss Rate)      : {fnr:.4f}")
        print(f"  FPR (False Alarm)    : {fpr:.4f}")

    # --- Fairness Analysis (Application_mode) ---
    print("\n=== Fairness Analysis (Application_mode) ===")
    print(f"{'App Mode':<10} | {'Count':<5} | {'FNR (Dropout)':<15} | {'FPR (Dropout)':<15}")
    print("-" * 55)

    sensitive_col = "Application_mode"
    if sensitive_col in X_test.columns:
        dropout_idx = list(labels).index("Dropout") if "Dropout" in labels else 0
        
        # Get unique modes present in the test set
        modes = sorted(X_test[sensitive_col].unique())
        
        for mode in modes:
            group_mask = (X_test[sensitive_col] == mode)
            if group_mask.sum() < 5: 
                continue 
            
            y_true_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]
            
            cm_group = confusion_matrix(y_true_group, y_pred_group, labels=range(len(labels)))
            
            TP = cm_group[dropout_idx, dropout_idx]
            FN = np.sum(cm_group[dropout_idx, :]) - TP
            FP = np.sum(cm_group[:, dropout_idx]) - TP
            TN = np.sum(cm_group) - (TP + FP + FN)
             
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            
            print(f"{str(mode):<10} | {group_mask.sum():<5} | {fnr:<15.4f} | {fpr:<15.4f}")
    else:
        print(f"Warning: '{sensitive_col}' not found in X_test features.")

    # --- Fairness Analysis (Course) ---
    print("\n=== Fairness Analysis (Course) ===")
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
    if sensitive_col_course in X_test.columns:
        dropout_idx = list(labels).index("Dropout") if "Dropout" in labels else 0
        
        courses = sorted(X_test[sensitive_col_course].unique())
        
        for course_id in courses:
            group_mask = (X_test[sensitive_col_course] == course_id)
            if group_mask.sum() < 5: 
                continue 
                
            y_true_group = y_test[group_mask]
            y_pred_group = y_pred[group_mask]
            
            cm_group = confusion_matrix(y_true_group, y_pred_group, labels=range(len(labels)))
            
            TP = cm_group[dropout_idx, dropout_idx]
            FN = np.sum(cm_group[dropout_idx, :]) - TP
            FP = np.sum(cm_group[:, dropout_idx]) - TP
            TN = np.sum(cm_group) - (TP + FP + FN)
             
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            
            course_name = course_map.get(int(course_id), str(course_id))
            if len(course_name) > 38:
                course_name = course_name[:35] + "..."
            
            print(f"{course_name:<40} | {group_mask.sum():<5} | {fnr:<15.4f} | {fpr:<15.4f}")
    else:
        print(f"Warning: '{sensitive_col_course}' not found in X_test features.")

    # full dataset
    X_full_pre = pre.transform(X).astype(np.float64)
    probs = gam.predict_proba(X_full_pre)

    out = df_binary.copy()
    out["prediction"] = label_enc.inverse_transform(gam.predict(X_full_pre))
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    for i, c in enumerate(label_enc.classes_):
        out[f"p_{c}"] = probs[:, i]
    out.to_csv(args.pred_out, index=False)
    print(f"Wrote predictions to {args.pred_out}")

    joblib.dump({"model": gam, "preprocess": pre, "label_encoder": label_enc}, args.save_model)
    print(f"Saved GAM model to {args.save_model}")

if __name__ == "__main__":
    main()
