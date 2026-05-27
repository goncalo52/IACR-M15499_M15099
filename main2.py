import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import torch
from facenet_pytorch import MTCNN


BFW_ROOT   = Path(os.environ.get("BFW_ROOT", "BFW_ROOT"))
OUT_DIR    = Path("output/")
CSV_PATH   = BFW_ROOT / "bfw-datatable.csv"
TARGET_SIZE = 160         
IMAGES_DIR  = Path("/home/gon/Desktop/Inteligencia/projeto/BFW_ROOT/Users/jrobby/bfw/bfw-cropped-aligned/")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (MTCNN)
def build_detector():
    mtcnn = MTCNN(
        image_size=TARGET_SIZE,
        margin=14,               # margem à volta da face (pixels)
        min_face_size=20,        # faces mais pequenas são ignoradas
        thresholds=[0.6, 0.7, 0.7],  # limiares das 3 etapas do MTCNN
        factor=0.709,
        keep_all=False,          # manter só a face maior (como o teu max() anterior)
        device=DEVICE,
        post_process=True,       # normalização para [-1, 1]
    )
    return mtcnn


def align_face(detector: MTCNN, img_path: Path):
    """
    Deteta, alinha e faz crop da face.
    Retorna imagem PIL 160x160 ou None se nenhuma face for detetada.
    """
    img = Image.open(img_path).convert("RGB")
    
    # Versão que devolve PIL (sem post_process) para guardar em disco
    detector_save = MTCNN(
        image_size=TARGET_SIZE,
        margin=14,
        keep_all=False,
        device=DEVICE,
        post_process=False,   
    )
    
    face_tensor, prob = detector_save(img, return_prob=True)
    
    if face_tensor is None or prob is None or prob < 0.9:
        return None
    
    # Converter tensor [C, H, W] uint8 → PIL Image
    face_np = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(face_np)


def collect_images(df: pd.DataFrame):
    """Recolhe todos os caminhos de imagem únicos do CSV."""
    paths = pd.concat([df["p1"], df["p2"]]).unique()
    return [IMAGES_DIR / p for p in paths]


def process(detector: MTCNN, image_paths: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    n_fallback = 0

    for img_path in tqdm(image_paths, desc="Aligning (MTCNN)"):
        rel      = img_path.relative_to(IMAGES_DIR)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Saltar se já processado
        if out_path.exists():
            manifest[str(rel)] = str(out_path)
            continue

        aligned = align_face(detector, img_path)

        if aligned is None:
            # Fallback: resize simples sem deteção (igual ao teu script original)
            aligned = Image.open(img_path).convert("RGB").resize(
                (TARGET_SIZE, TARGET_SIZE), Image.BILINEAR
            )
            n_fallback += 1

        aligned.save(out_path)
        manifest[str(rel)] = str(out_path)

    print(f"Fallback (sem face detetada): {n_fallback}/{len(image_paths)}")
    return manifest



def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"Pairs: {len(df)}")

    image_paths = collect_images(df)
    print(f"Unique images: {len(image_paths)}")

    print(f"Loading MTCNN detector (device={DEVICE})...")
    detector = build_detector()

    print("Processing images...")
    manifest = process(detector, image_paths, OUT_DIR / "data")

    # Guardar manifest
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Atualizar CSV com caminhos alinhados
    df["p1_aligned"] = df["p1"].map(manifest)
    df["p2_aligned"] = df["p2"].map(manifest)
    out_csv = OUT_DIR / "bfw-aligned.csv"
    df.to_csv(out_csv, index=False)

    print("Done!")
    print("Manifest:", manifest_path)
    print("CSV:", out_csv)


if __name__ == "__main__":
    main()