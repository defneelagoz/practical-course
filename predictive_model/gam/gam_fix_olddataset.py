import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

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


def plot_confusion_matrix(cm, labels, outpath="gam_confusion_matrix_student.png",
                          title="Confusion Matrix – GAM (student_data)"):
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
    sys.stdout.reconfigure(line_buffering=True)
    print("--- Starting GAM Fix Script using sys.stdout.reconfigure ---")
    
    class Args:
        csv = "student_data.csv"  # ← CHANGED
        target = None
        sep = ";"
        test_size = 0.2
        seed = 42
        save_model = "gam_model_student.joblib"            # ← CHANGED
        pred_out = "gam_predictions_student.csv"           # ← CHANGED
        cm_out = "gam_confusion_matrix_student.png"        # ← CHANGED

    args = Args()
    
    base_path = Path(__file__).parent.resolve() if "__file__" in locals() else Path().resolve()
    csv_path = None
    
    potential_dirs = [
        Path(args.csv).parent, 
        base_path,
        base_path / "gam",
        base_path / "predictive_model" / "gam",
    ]
    
    curr = base_path
    for _ in range(3):
        curr = curr.parent
        potential_dirs.append(curr)
        potential_dirs.append(curr / "predictive_model" / "gam")
        potential_dirs.append(curr / "gam")

    print("Searching for CSV in:")
    seen = set()
    for d in potential_dirs:
        if d in seen: 
            continue
        seen.add(d)
        
        target = d / "student_data.csv"  # ← CHANGED
        if target.exists():
            csv_path = target
            print(f"FOUND CSV at: {target}")
            break
            
    if csv_path is None:
        print(f"Current Working Directory: {os.getcwd()}")
        print("Search failed. Please ensure 'student_data.csv' is in a standard project directory.")
        raise FileNotFoundError(f"Could not find {args.csv} in project directories.")
    
    output_dir = csv_path.parent
    args.save_model = output_dir / "gam_model_student.joblib"           # ← CHANGED
    args.pred_out = output_dir / "gam_predictions_student.csv"          # ← CHANGED
    args.cm_out = output_dir / "gam_confusion_matrix_student.png"       # ← CHANGED

    print("Loading CSV...")
    df = pd.read_csv(csv_path, sep=args.sep, decimal=',', engine="python", encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows.")

    df.columns = [str(c).replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]

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
                print(f"  -> Failed to convert '{col}': {e}")
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=args.test_size, random_state=args.seed, stratify=y_enc
    )

    print("Preprocessing data...")
    X_train_pre = pre.fit_transform(X_train).astype(np.float64)
    X_test_pre = pre.transform(X_test).astype(np.float64)
    print(f"Training Data Shape: {X_train_pre.shape}")

    print("\n--- Step 1: Running Simple Fit (Sanity Check) ---")
    t0 = time.time()
    try:
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

    print("\n--- Step 2: Grid Search (SKIPPED) ---")
    gam = gam_simple

    print("\nRunning explicit 3-Fold Cross-Validation on the training set (Verification)...")
    cv_scores = cross_val_score(gam_simple, X_train_pre, y_train, cv=3, scoring='accuracy')
    print(f"3-Fold CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    y_pred = gam.predict(X_test_pre)
    print("\n=== Classification Report (test set) ===")
    print(classification_report(y_test, y_pred, digits=3, target_names=label_enc.classes_))

    cm = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"\n⭐ Cohen Kappa Score: {kappa:.4f}")

    plot_confusion_matrix(cm, label_enc.classes_, str(args.cm_out))
    print(f"🖼️ Saved confusion matrix to {args.cm_out}")

    X_full_pre = pre.transform(X).astype(np.float64)
    probs = gam.predict_proba(X_full_pre)

    out = df_binary.copy()
    out["prediction"] = label_enc.inverse_transform(gam.predict(X_full_pre))
    if probs.ndim == 1:
        probs = np.column_stack([1 - probs, probs])
    for i, c in enumerate(label_enc.classes_):
        out[f"p_{c}"] = probs[:, i]
    out.to_csv(args.pred_out, index=False)
    print(f"✅ Wrote predictions to {args.pred_out}")

    joblib.dump({"model": gam, "preprocess": pre, "label_encoder": label_enc}, args.save_model)
    print(f"💾 Saved GAM model to {args.save_model}")

if __name__ == "__main__":
    main()
