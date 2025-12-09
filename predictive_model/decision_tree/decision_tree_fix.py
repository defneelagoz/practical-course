import os
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, make_scorer

def infer_columns(df: pd.DataFrame):
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    for c in df.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        if df[c].nunique() <= 20:
            cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols

def plot_confusion_matrix(cm, labels, outpath="decision_tree_confusion_matrix_student2.png", title="Confusion Matrix"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def simple_fnr_fpr(cm, labels):
    """
    Calculates and returns per-class FNR and FPR.
    cm: Confusion Matrix (true_label on rows, predicted on cols)
    labels: List of class names
    """
    tn = []
    fp = []
    fn = []
    tp = []
    
    # Calculate for each class vs All
    # This is a 'macro' boolean approach
    metric_res = {}
    
    for i, label in enumerate(labels):
        # Treat class i as Positive, others as Negative
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)
        
        # Rates
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        metric_res[label] = {"FNR": fnr, "FPR": fpr}
        
    return metric_res

def main():
    parser = argparse.ArgumentParser(description="DT Fix with FNR/FPR")
    parser.add_argument("--csv", default="student_data_2.csv")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sep", default=";")
    args = parser.parse_args()

    # --- Robust Path Search (Copied from gam_fix.py) ---
    base_path = Path(__file__).parent.resolve() if "__file__" in locals() else Path().resolve()
    potential_dirs = [
        Path(args.csv).parent, 
        base_path, 
        base_path / "decision_tree",
        base_path.parent, 
        base_path.parent / "gam", # It often lives here
        base_path.parent.parent,
        Path("c:/Users/wwwut/practical-course/predictive_model/gam") # Fallback absolute
    ]
    csv_path = None
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
        # Debug info
        print(f"Search failed. checked: {[str(d) for d in potential_dirs]}")
        raise FileNotFoundError("Could not find student_data_2.csv in common locations.")
    
    output_dir = csv_path.parent / "decision_tree"
    output_dir.mkdir(exist_ok=True, parents=True) # Ensure dir exists

    print("Loading CSV...")
    df = pd.read_csv(csv_path, sep=args.sep, decimal=',', engine="python", encoding="utf-8-sig")
    
    # Clean columns
    df.columns = [str(c).replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]

    # Target Detection
    target_col = "Target"
    possibles = ["Output", "Target", "Status", "Outcome"]
    for p in possibles:
        if p in df.columns: target_col = p; break
    
    # --- Robust Data Cleaning (Copied from gam_fix.py) ---
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print("Cleaning numeric columns...")
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            try:
                series = X[col].astype(str).str.replace(',', '.', regex=False)
                temp = pd.to_numeric(series, errors='coerce')
                valid_count = temp.notna().sum()
                if valid_count > 0.5 * len(temp):
                    X[col] = temp
            except: pass
            
    # Remove extreme values
    num_cols_temp = X.select_dtypes(include=[np.number]).columns
    for c in num_cols_temp:
        mask = X[c] > 1e10
        if mask.any(): X.loc[mask, c] = np.nan
        
    # --- Pipeline & Training ---
    num_cols, cat_cols = infer_columns(X)
    
    pre = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop"
    )
    
    clf = DecisionTreeClassifier(random_state=args.seed, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    
    print("Fitting model...")
    pipe.fit(X_train, y_train)
    
    # --- Evaluation ---
    y_pred = pipe.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"Cohen Kappa: {kappa:.4f}")
    
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Save Outputs
    plot_path = output_dir / "decision_tree_confusion_matrix_student2.png"
    plot_confusion_matrix(cm, labels, str(plot_path), "Confusion Matrix - Decision Tree")
    print(f"Saved confusion matrix to {plot_path}")
    
    # Save Model
    mod_path = output_dir / "decision_tree_model_student2.joblib"
    joblib.dump(pipe, mod_path)
    print(f"Saved model to {mod_path}")
    
    # --- FNR / FPR Metrics ---
    print("\n=== FNR and FPR Metrics ===")
    metrics = simple_fnr_fpr(cm, labels)
    for cls, res in metrics.items():
        print(f"Class '{cls}':")
        print(f"  FNR (Miss Rate)      : {res['FNR']:.4f}")
        print(f"  FPR (False Alarm)    : {res['FPR']:.4f}")

if __name__ == "__main__":
    main()
