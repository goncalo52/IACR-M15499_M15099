import os
from pathlib import Path
import pandas as pd

# --- Configuration ---
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BFW_ROOT = Path(os.environ.get("BFW_ROOT", "BFW_ROOT"))
CSV_PATH = BFW_ROOT / "bfw-datatable.csv"
IMAGES_DIR = BFW_ROOT / "Users/jrobby/bfw/bfw-cropped-aligned"


VALID_GROUPS = {
    "asian_females", "asian_males", "black_females", "black_males",
    "indian_females", "indian_males", "white_females", "white_males"
}

def main():
    # Grab all JPEGs and filter by valid groups in the path
    images = [
        p for p in IMAGES_DIR.rglob("*.jpg") 
        if len(p.parts) >= 3 and p.parts[-3].lower() in VALID_GROUPS
    ]
    
    inventory = pd.DataFrame({
        "image_path": [str(p) for p in images],
        "group": [p.parts[-3].lower() for p in images],
        "person_id": [f"{p.parts[-3].lower()}/{p.parts[-2]}" for p in images]
    })
    inventory.to_csv(OUT_DIR / "image_metadata.csv", index=False)
    print(f"Found {len(inventory):,} valid images.\n")

    # 2. Load and Clean Pairs CSV
    df = pd.read_csv(CSV_PATH)

    # Drop genuine pairs with mismatched attributes
    df = df[~((df["label"] == 1) & (df["att1"] != df["att2"]))]

    # Build absolute paths and group info
    df["img1"] = df["p1"].apply(lambda p: str(IMAGES_DIR / p))
    df["img2"] = df["p2"].apply(lambda p: str(IMAGES_DIR / p))
    df["group"] = df["att1"].str.strip().str.lower()
    
    # Extract person identities
    df["person1"] = df["p1"].apply(lambda p: "/".join(Path(p).parts[-3:-1]))
    df["person2"] = df["p2"].apply(lambda p: "/".join(Path(p).parts[-3:-1]))

    # Filter out missing files by checking against our inventory
    valid_paths = set(inventory["image_path"])
    df = df[df["img1"].isin(valid_paths) & df["img2"].isin(valid_paths)]

    
    df.to_csv(OUT_DIR / "all_pairs.csv", index=False)


    print(f"Total valid pairs: {len(df):,}")
    print(f"Genuine (1):       {(df['label'] == 1).sum():,}")
    print(f"Impostor (0):      {(df['label'] == 0).sum():,}\n")
    print("Pairs per group & label:")
    print(df.groupby(["group", "label"]).size().to_string())

if __name__ == "__main__":
    main()