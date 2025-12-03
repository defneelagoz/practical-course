# %%
import os, argparse, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path




# %%
from sklearn.linear_model import LogisticRegression

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)


# %%
# Current notebook folder
base_path = Path().resolve()
print("Notebook folder:", base_path)

# If notebook is already inside predictive_model
csv_path = base_path.parent / "student_data.csv"

# Load CSV
df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8-sig")
print(f"Loaded {len(df)} rows from {csv_path}")


# %%
def infer_columns(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    # düşük kart sayılı tamsayıları kategorik say (örn. dönem kodu)
    for c in X.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        if X[c].nunique() <= 20:
            cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols

# %%
def plot_confusion_matrix(cm, labels, outpath="confusion_matrix.png", title="Confusion Matrix – Baseline"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath); plt.close(fig)


# %%
def main():
    parser = argparse.ArgumentParser(description="Baseline student outcome model (single-file).")

    parser.add_argument("--csv", default=os.path.join("predictive_model", "student_data.csv"))
    parser.add_argument("--target", default="Target")
    parser.add_argument("--sep", default=";")
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_model", default=os.path.join("predictive_model", "baseline_model.joblib"))
    parser.add_argument("--pred_out", default=os.path.join("predictive_model", "predictions.csv"))
    parser.add_argument("--cm_out", default=os.path.join("predictive_model", "confusion_matrix.png"))

    args = parser.parse_args(args=[])

    base_path = Path().resolve()
    args.pred_out = base_path / "predictions.csv"
    args.save_model = base_path / "baseline_model.joblib"
    args.cm_out = base_path / "confusion_matrix.png"

    # ===== 1) Load CSV =====
    csv_path = base_path.parent / "student_data.csv"
    print("Loading CSV from:", csv_path)

    df = pd.read_csv(csv_path, sep=args.sep, engine="python", encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows")

    df.columns = [str(c).replace("\ufeff", "").strip().replace(" ", "_") for c in df.columns]

    # Target detection
    cand_target = args.target.replace(" ", "_")
    if cand_target in df.columns:
        target_col = cand_target
    else:
        aliases = ["Target", "target", "Status", "Outcome", "Result", "Label"]
        aliases = [a.replace(" ", "_") for a in aliases]
        found = [c for c in df.columns if c in aliases or c.lower() in [a.lower() for a in aliases]]
        if found:
            target_col = found[0]
        else:
            KNOWN = {"Dropout", "Graduate", "Enrolled"}
            found = []
            for c in df.columns:
                vals = set(df[c].astype(str).str.strip().unique())
                if len(vals - KNOWN - {""}) <= 0 and len(vals & KNOWN) >= 2:
                    found.append(c)
            if not found:
                raise ValueError("Target column not found.")
            target_col = found[0]

    # ===== 2) Split =====
    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ===== 3) Preprocess + Model =====
    num_cols, cat_cols = infer_columns(df, target_col)

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=args.seed
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

    # ===== 4) Fit =====
    pipe.fit(X_train, y_train)

    # ===== 5) Evaluate =====
    y_pred = pipe.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=3))

    # === ⭐ KAPPA SCORE EKLENDİ ⭐ ===
    kappa = cohen_kappa_score(y_test, y_pred)
    print(f"\n⭐ Cohen Kappa Score: {kappa:.4f}")

    # Confusion Matrix
    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plot_confusion_matrix(cm, labels_sorted, outpath=args.cm_out)
    print(f"🖼️ Confusion matrix saved to: {args.cm_out}")

    # ===== 6) Predict on full CSV =====
    proba = pipe.predict_proba(X)
    out = df.copy()
    out["prediction"] = pipe.predict(X)
    classes = pipe.named_steps["clf"].classes_

    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]

    out.to_csv(args.pred_out, index=False)
    print(f"📁 Predictions saved to: {args.pred_out}")

    # ===== 7) Save Model =====
    joblib.dump(pipe, args.save_model)
    print(f"💾 Model saved to: {args.save_model}")


main()




