"""
Fold assignment:
  Folds 1-3  →  training
  Fold 4     →  validation / early stopping
  Fold 5     →  test (never seen during training)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


OUT_DIR = Path("outputs")
EMB_DIR = Path("data/embeddings")

EMB_DIM          = 512
BATCH_SIZE       = 1024
EPOCHS           = 50
LR               = 1e-3
FAIRNESS_LAMBDA  = 0.3   # weight of fairness loss relative to verification loss
RANDOM_SEED      = 42
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE         = 10    # early-stop patience (in evaluation intervals of 5 epochs)

# BFW fold assignment
TRAIN_FOLDS = [1, 2, 3]
VAL_FOLD    = 4
TEST_FOLD   = 5


def get_emb_path(image_path: str) -> Path:
    p = Path(image_path)
    return EMB_DIR / Path(*p.parts[-3:]).with_suffix(".npy")


def load_embeddings(pairs_df: pd.DataFrame):
    """Load pre-computed FaceNet embeddings for each pair in pairs_df."""
    e1_list, e2_list, labels, groups = [], [], [], []
    missing = 0
    for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Loading"):
        p1, p2 = get_emb_path(row.img1), get_emb_path(row.img2)
        if not p1.exists() or not p2.exists():
            missing += 1
            continue
        e1_list.append(np.load(p1).astype(np.float32))
        e2_list.append(np.load(p2).astype(np.float32))
        labels.append(int(row.label))
        groups.append(row.group)
    if missing:
        print(f"  Skipped {missing} pairs with missing embeddings.")
    e1 = np.stack(e1_list)
    e2 = np.stack(e2_list)
    # L2 normalise
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-9)
    return e1, e2, np.array(labels, dtype=np.int64), np.array(groups)



class PairDataset(Dataset):
    def __init__(self, e1, e2, labels, group_idx):
        self.e1        = torch.from_numpy(e1)
        self.e2        = torch.from_numpy(e2)
        self.labels    = torch.from_numpy(labels)
        self.group_idx = torch.tensor(group_idx, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.e1[idx], self.e2[idx], self.labels[idx], self.group_idx[idx]



class ProjectionHead(nn.Module):
    """Single linear layer initialised as identity — starts neutral."""
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.fc.weight)

    def forward(self, x):
        return nn.functional.normalize(self.fc(x), p=2, dim=1)


# LOSSES

def verification_loss(z1, z2, labels):
    """Cosine embedding loss: pull genuine pairs together, push impostors apart."""
    targets = labels.float() * 2 - 1   # 1 → +1,  0 → -1
    return nn.functional.cosine_embedding_loss(z1, z2, targets, margin=0.3)


def fairness_loss(z1, z2, labels, group_idx, n_groups):
    """
    Penalise differences in mean cosine score across demographic groups,
    separately for genuine and impostor pairs.
    A lower penalty means groups receive more similar scores on average.
    """
    cos   = (z1 * z2).sum(dim=1)
    total = torch.tensor(0.0, device=z1.device)

    for mask in [labels == 1, labels == 0]:
        if mask.sum() < n_groups:
            continue
        group_means = []
        for g in range(n_groups):
            g_mask = mask & (group_idx == g)
            if g_mask.sum() >= 2:
                group_means.append(cos[g_mask].mean()) # s̄^(g)
        if len(group_means) < 2:
            continue
        group_means = torch.stack(group_means)
        total = total + group_means.var() # somatório do desvio quadrático, 0 quando todos os grupos sao iguais

    return total


# EVALUATION

@torch.no_grad()
def evaluate(model, e1, e2, labels, groups, all_groups, threshold=None):
    model.eval()
    t1, t2 = torch.from_numpy(e1), torch.from_numpy(e2)
    score_chunks = []
    for s in range(0, len(t1), 4096):
        b1 = t1[s:s+4096].to(DEVICE)
        b2 = t2[s:s+4096].to(DEVICE)
        score_chunks.append((model(b1) * model(b2)).sum(1).cpu().numpy())
    scores = np.concatenate(score_chunks)

    if threshold is None:
        # Find threshold maximising balanced accuracy on this split
        thresholds = np.linspace(scores.min(), scores.max(), 500)
        best_ba, threshold = 0.0, thresholds[0]
        for t in thresholds:
            preds = (scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
            ba = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if ba > best_ba:
                best_ba, threshold = ba, t

    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    ba = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))

    # Per-group TPR spread (EqOdds proxy)
    tpr_per_group = []
    for g in all_groups:
        mask = groups == g
        if mask.sum() == 0:
            continue
        tp_g = ((preds[mask] == 1) & (labels[mask] == 1)).sum()
        fn_g = ((preds[mask] == 0) & (labels[mask] == 1)).sum()
        if tp_g + fn_g > 0:
            tpr_per_group.append(tp_g / (tp_g + fn_g))
    eq_odds = float(np.ptp(tpr_per_group)) if tpr_per_group else 0.0

    return ba, eq_odds, threshold, scores



def train(e1_tr, e2_tr, y_tr, g_tr,
          e1_vl, e2_vl, y_vl, g_vl,
          all_groups):
    """
    Train the projection head.
      - e1_tr / e2_tr / y_tr / g_tr : folds 1-3 (training)
      - e1_vl / e2_vl / y_vl / g_vl : fold 4   (early stopping)
    """
    group_to_idx = {g: i for i, g in enumerate(all_groups)}
    n_groups     = len(all_groups)

    g_tr_idx = np.array([group_to_idx[g] for g in g_tr])

    loader = DataLoader(
        PairDataset(e1_tr, e2_tr, y_tr, g_tr_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )

    model     = ProjectionHead(EMB_DIM).to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=LR)

    best_eq_odds = float("inf")
    best_state   = None
    no_improve   = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for e1_b, e2_b, lab_b, gidx_b in loader:
            e1_b, e2_b   = e1_b.to(DEVICE),   e2_b.to(DEVICE)
            lab_b, gidx_b = lab_b.to(DEVICE), gidx_b.to(DEVICE)

            z1, z2 = model(e1_b), model(e2_b)
            loss = (verification_loss(z1, z2, lab_b)
                    + FAIRNESS_LAMBDA * fairness_loss(z1, z2, lab_b, gidx_b, n_groups))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            ba, eq_odds, thr, _ = evaluate(
                model, e1_vl, e2_vl, y_vl, g_vl, all_groups
            )
            print(f"  Epoch {epoch:3d} | loss={total_loss / len(loader):.4f} "
                  f"| val BA={ba:.4f}  EqOdds={eq_odds:.4f}")

            if eq_odds < best_eq_odds:
                best_eq_odds = eq_odds
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE // 5:
                    print(f"  Early stop at epoch {epoch}.")
                    break

    model.load_state_dict(best_state)
    return model



def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print(f"Device: {DEVICE}")


    val_df  = pd.read_csv(OUT_DIR / "val_pairs.csv")   # folds 1-4
    test_df = pd.read_csv(OUT_DIR / "test_pairs.csv")  # fold 5

    # Verify the fold column exists
    if "fold" not in val_df.columns:
        raise ValueError("val_pairs.csv is missing a 'fold' column. "
                         "Re-run the BFW fold-splitting script first.")

   
    train_df = val_df[val_df["fold"].isin(TRAIN_FOLDS)].copy()
    estop_df = val_df[val_df["fold"] == VAL_FOLD].copy()

    print(f"\nFold assignment:")
    print(f"  Training   (folds {TRAIN_FOLDS}): {len(train_df):,} pairs")
    print(f"  Val/E-stop (fold  {VAL_FOLD}):    {len(estop_df):,} pairs")
    print(f"  Test       (fold  {TEST_FOLD}):    {len(test_df):,} pairs")

    print("\nLoading training embeddings...")
    e1_tr, e2_tr, y_tr, g_tr = load_embeddings(train_df)

    print("Loading early-stop (val) embeddings...")
    e1_vl, e2_vl, y_vl, g_vl = load_embeddings(estop_df)

    print("Loading test embeddings...")
    e1_test, e2_test, y_test, g_test = load_embeddings(test_df)

    all_groups = sorted(np.unique(np.concatenate([g_tr, g_vl, g_test])))
    print(f"\nGroups ({len(all_groups)}): {all_groups}")


    print("\nTraining projection head...")
    model = train(
        e1_tr, e2_tr, y_tr, g_tr,
        e1_vl, e2_vl, y_vl, g_vl,
        all_groups,
    )


    ba, eq_odds, threshold, test_scores = evaluate(
        model, e1_test, e2_test, y_test, g_test, all_groups
    )
    print(f"\nTest  BA={ba:.4f}  EqOdds={eq_odds:.4f}  threshold={threshold:.6f}")

    # ----------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scored test pairs
    out = test_df.copy().iloc[:len(test_scores)]
    out["score"] = test_scores
    out.to_csv(OUT_DIR / "scored_pairs_finetuned.csv",  index=False)
    out.to_csv(OUT_DIR / "test_pairs_projected.csv",    index=False)

    # Global threshold
    pd.DataFrame([{"threshold": threshold}]).to_csv(
        OUT_DIR / "global_threshold_projected.csv", index=False
    )

    # Per-group thresholds (each group's optimal threshold on test fold)
    group_thresholds = []
    for g in all_groups:
        mask = g_test == g
        if mask.sum() == 0:
            continue
        _, _, thr, _ = evaluate(
            model,
            e1_test[mask], e2_test[mask],
            y_test[mask],  g_test[mask],
            all_groups,
        )
        group_thresholds.append({"group": g, "threshold": thr})
    pd.DataFrame(group_thresholds).to_csv(
        OUT_DIR / "group_thresholds_projected.csv", index=False
    )

    # Model weights
    torch.save(model.state_dict(), OUT_DIR / "projection_head.pt")

    print("\nSaved outputs:")
    print("  scored_pairs_finetuned.csv")
    print("  test_pairs_projected.csv")
    print("  global_threshold_projected.csv")
    print("  group_thresholds_projected.csv")
    print("  projection_head.pt")
    print("\nRun main9.py to evaluate.")


if __name__ == "__main__":
    main()