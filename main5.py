from pathlib import Path
import pandas as pd

OUT_DIR = Path("outputs")

# Folds 1-4 = validation (threshold selection)
# Fold 5    = test (final evaluation)
VAL_FOLDS = [1, 2, 3, 4]
TEST_FOLD  = 5



def split_val_test(scored_df: pd.DataFrame):
    """
    Uses BFW pre-defined folds instead of a random split.
    Avoids identity leakage: the same person cannot appear
    on both sides of the split.
    """
    val  = scored_df[scored_df["fold"].isin(VAL_FOLDS)].copy()
    test = scored_df[scored_df["fold"] == TEST_FOLD].copy()

    val.to_csv(OUT_DIR  / "val_pairs.csv",  index=False)
    test.to_csv(OUT_DIR / "test_pairs.csv", index=False)

    print("\nValidation pairs (folds 1-4):")
    print(val.groupby(["group", "label"]).size().to_string())

    print("\nTest pairs (fold 5):")
    print(test.groupby(["group", "label"]).size().to_string())

    return val, test


def main():
    scored_path = OUT_DIR / "scored_pairs.csv"

    if not scored_path.exists():
        raise FileNotFoundError("scored_pairs.csv not found — run Step 3 first.")

    print("Loading scored pairs...")
    scored_df = pd.read_csv(scored_path)

    val_df, test_df = split_val_test(scored_df)

    print(f"\nValidation pairs: {len(val_df):,}")
    print(f"Test pairs:       {len(test_df):,}")
    print("\nDone.")


if __name__ == "__main__":
    main()