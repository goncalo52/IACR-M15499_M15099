from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix


OUT_DIR   = Path("outputs")
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COND_COLORS = {
    "Baseline":   "steelblue",
    "Post-hoc":   "darkorange",
    "Fine-tuned": "seagreen",
}
COND_MARKERS = {"Baseline": "o", "Post-hoc": "s", "Fine-tuned": "^"}
COND_LS      = {"Baseline": "-", "Post-hoc": "--", "Fine-tuned": ":"}

GROUPS = [
    "asian_females", "asian_males",
    "black_females", "black_males",
    "indian_females", "indian_males",
    "white_females", "white_males",
]


def compute_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    tnr = tn / max(tn + fp, 1)
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "TPR": tpr, "FPR": fpr,
        "FNR": fnr, "TNR": tnr,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def ba_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))


def find_best_threshold(df, n=1000):
    scores, labels = df["score"].values, df["label"].values
    thresholds = np.linspace(scores.min(), scores.max(), n)
    best_t, best_s = thresholds[0], -1.0
    for t in thresholds:
        s = ba_score(labels, (scores >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)


def evaluate_scored(scored_df, threshold):
    df = scored_df.copy()
    df["pred"] = (df["score"] >= threshold).astype(int)
    rows = []
    m = compute_metrics(df["label"], df["pred"])
    m["group"] = "overall"
    rows.append(m)
    for group, part in df.groupby("group"):
        m = compute_metrics(part["label"], part["pred"])
        m["group"] = group
        rows.append(m)
    return pd.DataFrame(rows)


def fairness_gaps(results_df):
    g = results_df[results_df["group"] != "overall"].copy()
    rows = []
    for metric in ["accuracy", "balanced_accuracy", "TPR", "FPR", "FNR", "TNR"]:
        gap   = g[metric].max() - g[metric].min()
        worst = g.loc[g[metric].idxmax(), "group"]
        best  = g.loc[g[metric].idxmin(), "group"]
        rows.append({"metric": metric, "gap": round(gap, 6),
                     "worst": worst, "best": best})
    return pd.DataFrame(rows)


def compute_additional_fairness(results_df):
    g = results_df[results_df["group"] != "overall"].copy()
    rows = []
    max_tpr, min_tpr = g["TPR"].max(), g["TPR"].min()
    di = min_tpr / max_tpr if max_tpr > 0 else 1.0
    rows.append({"metric": "Disparate Impact (TPR)",      "value": round(di, 4)})
    rows.append({"metric": "Equal Opportunity Gap (TPR)", "value": round(max_tpr - min_tpr, 4)})
    fpr_gap = g["FPR"].max() - g["FPR"].min()
    rows.append({"metric": "Equalized Odds Gap",          "value": round(max(max_tpr - min_tpr, fpr_gap), 4)})
    for metric in ["accuracy", "TPR", "FPR", "FNR"]:
        rows.append({"metric": f"STD {metric}", "value": round(g[metric].std(), 4)})
    return pd.DataFrame(rows)



def plot_fairness_gaps(gaps: dict):
    metrics = gaps["Baseline"]["metric"].tolist()
    conds   = list(gaps.keys())
    y       = np.arange(len(metrics))
    h       = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, cond in enumerate(conds):
        vals   = gaps[cond].set_index("metric").loc[metrics, "gap"].values
        offset = (i - len(conds) / 2 + 0.5) * h
        bars   = ax.barh(y + offset, vals, h,
                         label=cond, color=COND_COLORS[cond],
                         alpha=0.85, edgecolor="black", lw=0.4)
        for bar in bars:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.3f}", va="center", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=10)
    ax.set_xlabel("Max − Min gap across groups  (lower = more equitable)", fontsize=10)
    ax.set_title("Fairness Gaps: Baseline vs Post-hoc vs Fine-tuned\n"
                 "Each bar is the worst-case disparity across 8 demographic groups", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_fairness_gaps.png", dpi=150)
    plt.close()
    print("  Saved: comparison_fairness_gaps.png")



def plot_std_comparison(results: dict):
    metrics = ["accuracy", "TPR", "FPR", "FNR"]
    conds   = list(results.keys())
    x       = np.arange(len(metrics))
    w       = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cond in enumerate(conds):
        g    = results[cond][results[cond]["group"] != "overall"]
        stds = [g[m].std() for m in metrics]
        off  = (i - len(conds) / 2 + 0.5) * w
        bars = ax.bar(x + off, stds, w, label=cond,
                      color=COND_COLORS[cond], alpha=0.85,
                      edgecolor="black", lw=0.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Standard deviation across 8 groups  (lower = more equitable)", fontsize=10)
    ax.set_title("Inter-Group Consistency: Baseline vs Post-hoc vs Fine-tuned", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_std.png", dpi=150)
    plt.close()
    print("  Saved: comparison_std.png")



def plot_roc_per_group(scored_sets: dict):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, group in enumerate(GROUPS):
        ax = axes[i]
        for cond, scored_df in scored_sets.items():
            part = scored_df[scored_df["group"] == group]
            fpr_arr, tpr_arr, _ = roc_curve(part["label"], part["score"])
            roc_auc = auc(fpr_arr, tpr_arr)
            ax.plot(fpr_arr, tpr_arr,
                    color=COND_COLORS[cond], ls=COND_LS[cond], lw=2,
                    label=f"{cond} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(group.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("ROC Curves per Demographic Group — All Three Conditions\n"
                 "Overlapping curves = similar separability; gaps = condition differences",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_roc.png", dpi=150)
    plt.close()
    print("  Saved: comparison_roc.png")


def plot_metric_heatmap(results: dict):
    conds = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, cmap, title in zip(
        axes,
        ["TPR", "FPR"],
        ["YlGn", "YlOrRd"],
        ["TPR per group  (higher = better)", "FPR per group  (lower = better)"],
    ):
        # Build matrix: rows = groups, cols = conditions
        data = np.array([
            [results[c].set_index("group").loc[g, metric] for c in conds]
            for g in GROUPS
        ])

        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=data.min() - 0.01, vmax=data.max() + 0.01)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=10)
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels([g.replace("_", " ") for g in GROUPS], fontsize=9)
        ax.set_title(title, fontsize=11)

        # Annotate cells
        for r in range(len(GROUPS)):
            for c in range(len(conds)):
                ax.text(c, r, f"{data[r, c]:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="black")

    fig.suptitle("Per-Group TPR and FPR Heatmap — All Conditions\n"
                 "Reveals which groups benefit or regress under each intervention",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_heatmap.png", dpi=150)
    plt.close()
    print("  Saved: comparison_heatmap.png")




def plot_delta_vs_baseline(results: dict):

    base  = results["Baseline"].set_index("group")
    conds = ["Post-hoc", "Fine-tuned"]
    x     = np.arange(len(GROUPS))
    w     = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    for ax, metric, better in zip(axes, ["TPR", "FPR"], ["↑ better", "↓ better"]):
        # Subtle background: green = improvement side, red = regression side
        if metric == "TPR":
            ax.axhspan(0, 0.06,  color="honeydew",  zorder=0)
            ax.axhspan(-0.06, 0, color="mistyrose",  zorder=0)
        else:
            ax.axhspan(-0.06, 0, color="honeydew",  zorder=0)
            ax.axhspan(0, 0.06,  color="mistyrose",  zorder=0)

        for i, cond in enumerate(conds):
            df     = results[cond].set_index("group")
            deltas = np.array([df.loc[g, metric] - base.loc[g, metric] for g in GROUPS])
            off    = (i - 0.5) * w
            bars   = ax.bar(x + off, deltas, w,
                            color=COND_COLORS[cond],
                            edgecolor="black", lw=0.4,
                            label=cond, alpha=0.85, zorder=3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + (0.001 if h >= 0 else -0.003),
                        f"{h:+.3f}", ha="center",
                        va="bottom" if h >= 0 else "top", fontsize=6.5)

        ax.axhline(0, color="black", lw=1, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_", "\n") for g in GROUPS], fontsize=8)
        ax.set_ylabel(f"Δ {metric} vs Baseline  ({better})", fontsize=10)
        ax.set_title(f"Δ {metric} per Group vs Baseline\n"
                     f"Green background = improvement side", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, zorder=1)

    fig.suptitle("Δ TPR and Δ FPR vs Baseline — Post-hoc vs Fine-tuned",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison_delta.png", dpi=150)
    plt.close()
    print("  Saved: comparison_delta.png")



def write_summary_report(results, gaps, additional):
    path = OUT_DIR / "threeway_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("THREE-WAY COMPARISON SUMMARY\n")
        f.write("Baseline | Post-hoc | Fine-tuned\n")
        f.write("=" * 70 + "\n\n")
        for cond in results:
            f.write(f"{'=' * 70}\n{cond.upper()}\n{'=' * 70}\n")
            f.write(results[cond].to_string(index=False))
            f.write("\n\nFairness gaps:\n")
            f.write(gaps[cond].to_string(index=False))
            f.write("\n\nAdditional fairness metrics:\n")
            f.write(additional[cond].to_string(index=False))
            f.write("\n\n")
        merged = gaps["Baseline"][["metric", "gap"]].rename(
            columns={"gap": "gap_baseline"})
        for cond in ["Post-hoc", "Fine-tuned"]:
            merged = merged.merge(
                gaps[cond][["metric", "gap"]].rename(
                    columns={"gap": f"gap_{cond.lower().replace('-','_')}"}),
                on="metric")
        f.write("=" * 70 + "\nFAIRNESS GAP SIDE-BY-SIDE\n" + "=" * 70 + "\n")
        f.write(merged.to_string(index=False))
        f.write("\n")
    print(f"  Saved: threeway_summary.txt")



def main():
    for fname in ["baseline_results.csv", "mitigated_results.csv",
                  "scored_pairs_finetuned.csv", "test_pairs.csv"]:
        if not (OUT_DIR / fname).exists():
            raise FileNotFoundError(f"{fname} not found — run previous steps first.")

    print("Loading results...")
    baseline_results  = pd.read_csv(OUT_DIR / "baseline_results.csv")
    mitigated_results = pd.read_csv(OUT_DIR / "mitigated_results.csv")
    finetuned_scored  = pd.read_csv(OUT_DIR / "scored_pairs_finetuned.csv")
    test_pairs        = pd.read_csv(OUT_DIR / "test_pairs.csv")

    print("Evaluating fine-tuned scores...")
    ft_threshold      = find_best_threshold(finetuned_scored)
    print(f"Fine-tuned threshold: {ft_threshold:.6f}")
    finetuned_results = evaluate_scored(finetuned_scored, ft_threshold)
    finetuned_results.to_csv(OUT_DIR / "finetuned_results.csv", index=False)

    results = {
        "Baseline":   baseline_results,
        "Post-hoc":   mitigated_results,
        "Fine-tuned": finetuned_results,
    }
    scored_sets = {
        "Baseline":   test_pairs,
        "Post-hoc":   test_pairs,
        "Fine-tuned": finetuned_scored,
    }

    # Console output
    print("\n--- RAW PERFORMANCE METRICS PER GROUP ---")
    for cond, res_df in results.items():
        print(f"\n{cond.upper()} RAW RESULTS:")
        print(res_df.to_string(index=False))

    gaps       = {cond: fairness_gaps(df)               for cond, df in results.items()}
    additional = {cond: compute_additional_fairness(df)  for cond, df in results.items()}

    print("\n--- FAIRNESS GAP COMPARISON ---")
    for cond, gap_df in gaps.items():
        print(f"\n{cond}:")
        print(gap_df.to_string(index=False))

    print("\n--- ADDITIONAL FAIRNESS METRICS ---")
    for cond, add_df in additional.items():
        print(f"\n{cond}:")
        print(add_df.to_string(index=False))

    # Comparative plots only
    print("\nGenerating comparative plots...")
    plot_fairness_gaps(gaps)            # gaps side by side
    plot_std_comparison(results)        # STD across groups
    plot_roc_per_group(scored_sets)     # ROC all conditions per group
    plot_metric_heatmap(results)        # heatmap groups × conditions
    plot_delta_vs_baseline(results)     # Δ vs baseline per group

    write_summary_report(results, gaps, additional)
    print("\nDone.")


if __name__ == "__main__":
    main()