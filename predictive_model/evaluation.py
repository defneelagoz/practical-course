# evaluation.py
from sklearn.metrics import confusion_matrix

def eval_model(y_true, y_pred, group_column=None, X=None):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)

    results = {
        "confusion_matrix": cm,
        "FPR": FPR,
        "FNR": FNR
    }

    # Optional fairness check
    if group_column is not None and X is not None:
        fairness = {}
        for g in X[group_column].unique():
            idx = X[group_column] == g
            cm_g = confusion_matrix(y_true[idx], y_pred[idx]).ravel()
            tng, fpg, fng, tpg = cm_g
            fairness[g] = {
                "FPR": fpg / (fpg + tng) if (fpg+tng)>0 else None,
                "FNR": fng / (fng + tpg) if (fng+tpg)>0 else None
            }
        results["fairness"] = fairness

    return results
