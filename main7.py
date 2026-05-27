from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


OUT_DIR = Path("outputs")

FMR_OPERATING_POINTS = [1e-3, 1e-4]


def compute_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    tnr = tn / max(tn + fp, 1)

    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "TPR": tpr,
        "FPR": fpr,
        "FNR": fnr,
        "TNR": tnr,
        "TP":  int(tp),
        "FP":  int(fp),
        "TN":  int(tn),
        "FN":  int(fn)
    }


def compute_eer(y_true, scores):
    """Equal Error Rate: point where FMR == FNMR."""
    thresholds = np.linspace(min(scores), max(scores), 1000)
    best_eer = 1.0
    best_t = thresholds[0]

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fmr  = fp / max(fp + tn, 1)
        fnmr = fn / max(fn + tp, 1)
        if abs(fmr - fnmr) < abs(best_eer - 0.5) + 1:
            best_eer = abs(fmr - fnmr)
            best_t   = t

    pred = (scores >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    eer_value = (fp / max(fp + tn, 1) + fn / max(fn + tp, 1)) / 2

    return float(eer_value), float(best_t)


def fnmr_at_fmr(y_true, scores, target_fmr: float):
    """FNMR at a fixed FMR operating point."""
    thresholds = sorted(np.unique(scores), reverse=True)

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        fmr = fp / max(fp + tn, 1)
        if fmr <= target_fmr:
            fnmr = fn / max(fn + tp, 1)
            return float(fnmr), float(t)

    return 1.0, float(thresholds[-1])


def compute_operating_points(y_true, scores) -> dict:
    y_true = np.array(y_true)
    scores = np.array(scores)

    eer, eer_t = compute_eer(y_true, scores)
    result = {"EER": eer, "EER_threshold": eer_t}

    for fmr_level in FMR_OPERATING_POINTS:
        fnmr, t = fnmr_at_fmr(y_true, scores, fmr_level)
        label = f"1e{int(np.log10(fmr_level))}"
        result[f"FNMR@FMR={label}"] = fnmr
        result[f"threshold@FMR={label}"] = t

    return result


def evaluate(df: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    Evaluates performance across all 8 BFW demographic groups using:
    - Standard classification metrics
    - EER and FNMR at Frontex operating points
    """
    results = []
    overall = compute_metrics(df["label"], df["pred"])
    overall["group"]    = "overall"
    overall.update(compute_operating_points(df["label"], df["score"]))
    results.append(overall)

    for group, part in df.groupby("group"):
        m = compute_metrics(part["label"], part["pred"])
        m["group"] = group
        m.update(compute_operating_points(part["label"], part["score"]))
        results.append(m)

    results_df = pd.DataFrame(results)

    cols = [
        "group",
        "EER",
        "FNMR@FMR=1e-3", "FNMR@FMR=1e-4",
        "accuracy", "balanced_accuracy",
        "TPR", "FPR", "FNR", "TNR",
        "TP", "FP", "TN", "FN"
    ]
    cols = [c for c in cols if c in results_df.columns]

    print(f"\n===== {title} =====")
    print(results_df[cols].to_string(index=False))

    return results_df[cols]


# ============================================================
# MAIN
# ============================================================

def main():
    # Load test pairs
    test_path = OUT_DIR / "test_pairs.csv"

    # Load thresholds
    global_threshold_path = OUT_DIR / "global_threshold.csv"
    group_thresholds_path = OUT_DIR / "group_thresholds.csv"
    
    print("Loading test pairs and thresholds...")
    test_df = pd.read_csv(test_path)
    global_threshold = pd.read_csv(global_threshold_path)["threshold"].iloc[0]
    group_thresholds = pd.read_csv(group_thresholds_path).set_index("group")["threshold"].to_dict()


    # BASELINE: GLOBAL THRESHOLD
    baseline_test = test_df.copy()
    baseline_test["pred"] = (baseline_test["score"] >= global_threshold).astype(int)

    baseline_results = evaluate(baseline_test, "BASELINE — GLOBAL THRESHOLD")
    baseline_results.to_csv(OUT_DIR / "baseline_results.csv", index=False)

    # MITIGATION: PER-GROUP THRESHOLDS
    mitigated_test = test_df.copy()
    mitigated_test["pred"] = [
        int(score >= group_thresholds[group])
        for score, group in zip(mitigated_test["score"], mitigated_test["group"])
    ]

    mitigated_results = evaluate(mitigated_test, "MITIGATION — GROUP THRESHOLDS")
    mitigated_results.to_csv(OUT_DIR / "mitigated_results.csv", index=False)

    print("\nSaved:")
    print("  outputs/baseline_results.csv")
    print("  outputs/mitigated_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()