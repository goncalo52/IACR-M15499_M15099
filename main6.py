from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


OUT_DIR = Path("outputs")


#THRESHOLD SELECTION

def balanced_accuracy_from_predictions(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)

    return 0.5 * (tpr + tnr)


def find_best_threshold(df: pd.DataFrame, n_thresholds: int = 1000) -> float:
    """
    Finds the threshold that maximises balanced accuracy.
    Uses n_thresholds evenly spaced candidates instead of every unique score
    to avoid being slow on large datasets like BFW.
    """
    scores = df["score"].values
    labels = df["label"].values

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    best_t     = thresholds[0]
    best_score = -1

    for t in thresholds:
        pred  = (scores >= t).astype(int)
        score = balanced_accuracy_from_predictions(labels, pred)

        if score > best_score:
            best_score = score
            best_t     = t

    return float(best_t)


def main():
    val_path = OUT_DIR / "val_pairs.csv"

    if not val_path.exists():
        raise FileNotFoundError("val_pairs.csv not found — run Step 4 first.")

    print("Loading validation pairs...")
    val_df = pd.read_csv(val_path)

    # Global threshold (one for all groups)
    global_threshold = find_best_threshold(val_df)
    print(f"\nGlobal threshold: {global_threshold:.6f}")

    # Per-group thresholds
    group_thresholds = {
        group: find_best_threshold(part)
        for group, part in val_df.groupby("group")
    }

    group_thresholds_df = pd.DataFrame([
        {"group": g, "threshold": t}
        for g, t in group_thresholds.items()
    ])

    print("\nGroup thresholds:")
    print(group_thresholds_df.to_string(index=False))

    # Save both
    pd.DataFrame([{"threshold": global_threshold}]).to_csv(
        OUT_DIR / "global_threshold.csv", index=False
    )
    group_thresholds_df.to_csv(
        OUT_DIR / "group_thresholds.csv", index=False
    )

    print(f"\nSaved:")
    print(f"  outputs/global_threshold.csv")
    print(f"  outputs/group_thresholds.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()