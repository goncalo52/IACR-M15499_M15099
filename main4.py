import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OUT_DIR = Path("outputs")
EMB_DIR = Path("data/embeddings")


def get_emb_path(image_path: str) -> Path:
    """Mirrors the get_output_paths logic from Step 2."""
    p = Path(image_path)
    rel_path = Path(*p.parts[-3:]).with_suffix(".npy")
    return EMB_DIR / rel_path


def cosine_score(img1: str, img2: str) -> float:
    emb1 = np.load(get_emb_path(img1))
    emb2 = np.load(get_emb_path(img2))
    return float(np.dot(emb1, emb2))


def score_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    scores = []

    for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Scoring pairs"):
        scores.append(cosine_score(row.img1, row.img2))

    scored = pairs_df.copy()
    scored["score"] = scores


    scored.to_csv(OUT_DIR / "scored_pairs.csv", index=False)

    print("\nScored pairs:")
    print(scored.groupby(["group", "label"]).size().to_string())

    return scored


def main():
    pairs_path = OUT_DIR / "all_pairs.csv"
    pairs_df = pd.read_csv(pairs_path)
    scored_df = score_pairs(pairs_df)

    print(f"\nTotal scored pairs: {len(scored_df):,}")
    print(f"\nScored pairs saved to: {OUT_DIR / 'scored_pairs.csv'}")


if __name__ == "__main__":
    main()