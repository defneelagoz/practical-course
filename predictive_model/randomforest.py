import os, argparse, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier      # ← CHANGED
from sklearn.metrics import classification_report, confusion_matrix

def infer_columns(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    for c in X.select_dtypes(include=["int64", "int32", "int16", "int8"]).columns:
        if X[c].nunique() <= 20:
            cat_cols.append(c)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols

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

def main():
    parser = argparse.ArgumentParser(description="Baseline student outcome model (single-file).")
    parser.add_argument("--csv", default=os.path.join("predictive_model", "student_data.csv"),
                        help="CSV yolu (default: predictive_model/student_data.csv)")
    parser.add_argument("--target", default="Target", help="Hedef sütun adı (default: Target)")
    parser.add_argument("--sep", default=";", help="CSV ayraç (default: ;)")
    parser.add_argument("--test_size", type=float, default=0.20, help="Test oranı (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgele tohum (default: 42)")
    parser.add_argument("--save_model", default=os.path.join("predictive_model", "baseline_model.joblib"),
                        help="Model kaydetme yolu")
    parser.add_argument("--pred_out", default=os.path.join("predictive_model", "predictions.csv"),
                        help="Tahmin çıktısı CSV yolu")
    parser.add_argument("--cm_out", default=os.path.join("predictive_model", "confusion_matrix.png"),
                        help="Karışıklık matrisi görsel yolu")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, sep=args.sep, engine="python", encoding="utf-8-sig")

    clean_cols = []
    for c in df.columns:
        c2 = str(c).replace("\ufeff", "").strip().replace(" ", "_")
        clean_cols.append(c2)
    df.columns = clean_cols

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
                raise ValueError(
                    f"Couldn't find the target column. Columns: {list(df.columns)[:15]} ... "
                    "Pass --target CorrectName or open the CSV to check the exact header."
                )
            target_col = found[0]

    y = df[target_col].astype(str)
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    num_cols, cat_cols = infer_columns(df, target_col)
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )

    # ------------------------ ONLY CHANGE HERE ------------------------
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=args.seed
    )
    # ------------------------------------------------------------------

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("=== Classification Report (test set) ===")
    print(classification_report(y_test, y_pred, digits=3))

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plot_confusion_matrix(cm, labels_sorted, outpath=args.cm_out)
    print(f"🖼️ Saved confusion matrix to {args.cm_out}")

    proba = pipe.predict_proba(X)
    out = df.copy()
    out["prediction"] = pipe.predict(X)
    classes = list(pipe.named_steps["clf"].classes_)

    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]

    out.to_csv(args.pred_out, index=False)
    print(f"✅ Wrote predictions to {args.pred_out}")

    if args.save_model:
        joblib.dump(pipe, args.save_model)
        print(f"💾 Saved model to {args.save_model}")

if __name__ == "__main__":
    main()
