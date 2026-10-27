"""
Evaluation of OOD scores against the petrographic ground truth.

This module is a LIBRARY, not a command-line script: it is used by
run_ood_cv.py, which passes it the per-sample report built by pooling the
folds of the cross-validation.

Ground-truth CSV, columns (sample_id, true_class), sep=';':
  - true_class = fabric number (1-20, same numbering as training) for samples
    that petrographic analysis referred to a known fabric;
  - true_class = "NOVEL" for samples that are genuinely new (loners).
    "NEW", "LONER" and the Italian "NUOVA"/"NUOVO" are also accepted.

Two independent evaluations:
  1. evaluate_classification - on known-fabric samples, does the predicted
     class match the true one?
  2. evaluate_ood_detection - on samples with known ground truth (known +
     novel), is the assigned status correct? With so few loners a single AUC
     would be noisy, so a dot plot shows on which side of the threshold each
     sample falls.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

from ood_utils import STATUS_NEW

NOVEL_SENTINEL = "NOVEL"
NOVEL_LABELS = {"NOVEL", "NEW", "LONER", "NUOVA", "NUOVO"}


def load_ground_truth(path: str, sep: str = ";") -> pd.DataFrame:
    gt = pd.read_csv(path, sep=sep)
    gt.columns = [c.strip().lower() for c in gt.columns]
    if "sample_id" not in gt.columns or "true_class" not in gt.columns:
        raise ValueError(
            f"The ground-truth CSV must have columns sample_id and true_class. Found: {gt.columns.tolist()}"
        )
    gt["true_class"] = gt["true_class"].astype(str).str.strip()
    gt = gt[gt["true_class"] != ""]  # skip rows not yet filled in
    gt.loc[gt["true_class"].str.upper().isin(NOVEL_LABELS), "true_class"] = NOVEL_SENTINEL
    return gt[["sample_id", "true_class"]]


def evaluate_classification(merged: pd.DataFrame):
    """On known-fabric samples (numeric true_class), does the model's
    predicted_class match the true class?"""
    known = merged[merged["true_class"] != NOVEL_SENTINEL].copy()
    if known.empty:
        print("No known-fabric samples in the ground-truth file: skipping this evaluation.")
        return known

    known["true_class_num"] = known["true_class"].astype(int)
    known["correct"] = known["predicted_class"].astype(str) == known["true_class_num"].astype(str)

    acc = known["correct"].mean()
    print("\n" + "=" * 70)
    print("CLASSIFICATION ACCURACY (known-fabric samples)")
    print("=" * 70)
    print(f"Samples evaluated: {len(known)}")
    print(f"Accuracy: {acc*100:.1f}%  ({known['correct'].sum()}/{len(known)})")

    errors = known[~known["correct"]][["sample_id", "true_class_num", "predicted_class", "mahalanobis_mean", "status"]]
    if len(errors) > 0:
        print("\nMisclassified samples:")
        print(errors.sort_values("mahalanobis_mean", ascending=False).to_string(index=False))
    else:
        print("\nNo classification errors on the known samples.")

    return known


def evaluate_ood_detection(merged: pd.DataFrame, out_dir: str, threshold: float = None):
    """On samples with known ground truth (known + novel), is the status
    assigned by the detector correct? True positive = genuinely new sample
    flagged as outlier."""
    merged = merged.copy()
    merged["true_is_novel"] = merged["true_class"] == NOVEL_SENTINEL
    merged["pred_is_novel"] = merged["status"] == STATUS_NEW

    print("\n" + "=" * 70)
    print("OOD DETECTOR PERFORMANCE")
    print("=" * 70)
    print(f"Samples evaluated: {len(merged)}  (genuinely new: {merged['true_is_novel'].sum()})")

    cm = confusion_matrix(merged["true_is_novel"], merged["pred_is_novel"], labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    print("\nConfusion matrix (rows = truth, columns = predicted):")
    print(pd.DataFrame(cm, index=["true: KNOWN", "true: NOVEL"], columns=["pred: KNOWN", "pred: NOVEL"]))
    print(f"\nPrecision (flagged as new that really are new): {precision:.3f}")
    print(f"Recall (genuinely new samples that were found):  {recall:.3f}")
    print(f"F1: {f1:.3f}")

    if merged["true_is_novel"].nunique() == 2:
        n_pos = int(merged["true_is_novel"].sum())
        print(
            f"\n(ROC-AUC deliberately omitted: with only {n_pos} genuinely new "
            "samples it would be too noisy. See the dot plot for a sample-by-sample view.)"
        )
        plot_distance_dotplot(merged, out_dir, threshold=threshold)
    else:
        print("\nAt least one known AND one novel sample are needed in the ground truth for this comparison.")

    return merged


def plot_distance_dotplot(merged: pd.DataFrame, out_dir: str, threshold: float = None,
                          xlabel: str = "Mahalanobis distance / fold threshold (mean over folds)"):
    """One dot per sample: distance on the x axis, coloured by ground truth
    (known / novel), labelled with the sample_id, with an X where the
    predicted status is wrong. With few samples this is more informative
    than a summary AUC."""
    merged_sorted = merged.sort_values("mahalanobis_mean").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(merged_sorted))))
    colors = np.where(merged_sorted["true_is_novel"], "#C0392B", "#3A6EA5")
    y = np.arange(len(merged_sorted))
    ax.scatter(merged_sorted["mahalanobis_mean"], y, c=colors, s=60, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(merged_sorted["sample_id"], fontsize=8)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2)

    mismatches = merged_sorted["true_is_novel"] != merged_sorted["pred_is_novel"]
    ax.scatter(
        merged_sorted.loc[mismatches, "mahalanobis_mean"],
        y[mismatches],
        marker="x",
        s=120,
        c="black",
        linewidths=2,
        zorder=4,
    )

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3A6EA5", markersize=9, label="true: known"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C0392B", markersize=9, label="true: novel"),
        Line2D([0], [0], marker="x", color="black", markersize=9, linewidth=0, label="wrong predicted status"),
    ]
    if threshold is not None:
        legend_elems.append(Line2D([0], [0], color="black", linestyle="--", label=f"Threshold ({threshold:.2f})"))
    ax.legend(handles=legend_elems, loc="lower right")
    ax.set_xlabel(xlabel)
    ax.set_title("Per-sample distance - colour = ground truth, X = wrong OOD prediction")
    plt.tight_layout()

    save_path = os.path.join(out_dir, "ood_distance_dotplot.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Dot plot saved to {save_path}")
