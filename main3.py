import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("outputs")
EMB_DIR = Path("data/embeddings")

# Aligned crops produced by main2.py
ALIGNED_CSV = Path("output/bfw-aligned.csv")
CROPS_ROOT  = Path("output/data")

for d in [OUT_DIR, EMB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Model only (MTCNN already ran in main2.py) ---
print(f"Using device: {DEVICE}")
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def get_emb_path(crop_path: Path) -> Path:
    """Maps a crop path to its embedding path, mirroring the directory structure."""
    rel = crop_path.relative_to(CROPS_ROOT)
    emb_path = EMB_DIR / rel.with_suffix(".npy")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    return emb_path


def load_crop_as_tensor(crop_path: Path) -> torch.Tensor:
    """Loads an already-aligned crop and standardises it for the model."""
    img = Image.open(crop_path).convert("RGB").resize((160, 160))
    arr = np.asarray(img).astype(np.float32)
    tensor = torch.tensor(arr).permute(2, 0, 1)  # HWC → CHW
    return fixed_image_standardization(tensor)


@torch.no_grad()
def extract_embedding(crop_path: Path) -> np.ndarray:
    """Generates and L2-normalises an embedding from a pre-aligned crop."""
    face = load_crop_as_tensor(crop_path).unsqueeze(0).to(DEVICE)
    emb = model(face).cpu().numpy()[0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb


def main():
    # Load the aligned CSV written by main2.py
    if not ALIGNED_CSV.exists():
        raise FileNotFoundError(
            f"{ALIGNED_CSV} not found — run main2.py first to generate aligned crops."
        )

    df = pd.read_csv(ALIGNED_CSV)

    # Collect unique crop paths from both columns
    unique_crops = pd.unique(
        df[["p1_aligned", "p2_aligned"]].values.ravel()
    )
    unique_crops = [Path(p) for p in unique_crops if isinstance(p, str)]

    print(f"Total unique aligned crops: {len(unique_crops):,}")
    missing = [p for p in unique_crops if not p.exists()]
    if missing:
        print(f"  Warning: {len(missing):,} crop files not found on disk — they will be skipped.")

    for crop_path in tqdm(unique_crops, desc="Extracting embeddings"):
        if not crop_path.exists():
            continue

        emb_path = get_emb_path(crop_path)

        # Resume: skip already-processed embeddings
        if emb_path.exists():
            continue

        try:
            emb = extract_embedding(crop_path)
            np.save(emb_path, emb)
        except Exception as e:
            print(f"Skipping {crop_path}: {e}")

    print(f"\nEmbeddings saved in: {EMB_DIR}")


if __name__ == "__main__":
    main()