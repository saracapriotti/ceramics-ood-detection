"""
Leave-one-fabric-out OOD detection experiment (controlled novelty).

For each requested class (1-based numbering, as in the CSV) and seed:

  1. Remove ALL images of that class from training and validation, so that
     the model never sees it, neither in training nor in threshold
     calibration.
  2. Remap the remaining classes to a contiguous range 1..(N-1) before
     passing them to FabricsDataset (which subtracts 1 to obtain the 0-based
     index); without this the numbering would have a gap.
  3. Train a DynamicViT from scratch on the N-1 classes (one checkpoint per
     experiment, in checkpoints/leave_out_<class>[_seed<n>]/).
  4. Estimate the in-distribution statistics (class means + shared
     within-class covariance) on the new training set, without augmentation.
  5. Score validation, in-distribution test and held-out class with the
     scores of ood_utils.analyze_ood_dataset, aggregated per sample.
  6. Compute AUROC, AUPR and FPR@95TPR (positives = held-out class,
     negatives = test samples of the known classes).

In the paper the held-out fabrics are class 1 (petrographically distinctive)
and class 16 (inside a group of closely related fabrics, the near-OOD case),
each repeated with three seeds.

Usage (from src/):
    python run_loco_ood.py --classes 1 16 --seeds 100 101 102
    python run_loco_ood.py --classes 1 16 --epochs 15        # quick check

If a checkpoint already exists, training is skipped and the existing weights
are evaluated (use --force_retrain to retrain).

The isolation-forest scores for this experiment are computed afterwards by
sklearn_ood_baselines.py on the same checkpoints.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

import config
from dataset_samples import (
    FabricsDataset,
    collate_fn,
    get_transforms,
    load_dataset_csv,
    set_seed,
    split_by_samples,
)
from model import build_model, load_model
from ood_utils import (
    analyze_ood_dataset,
    compute_id_statistics,
    compute_ood_threshold,
    make_clean_copy,
)
from train import get_class_weights, train

# Kept in the output file names and columns for compatibility with
# operating_point_eval.py and sklearn_ood_baselines.py.
COVARIANCE = "shared"


# -- Datasets and dataloaders ---------------------------------------------------

def load_full_dataframe(csv_file: str = config.CSV_FILE) -> pd.DataFrame:
    return load_dataset_csv(csv_file)


def build_loco_dataloaders(df_full: pd.DataFrame, held_out_class: int, seed: int = config.SEED):
    """held_out_class is the ORIGINAL 1-based label (as in the CSV), fully
    excluded from training and validation."""
    col_label = df_full.columns[1]

    df_ood = df_full[df_full[col_label] == held_out_class].copy()
    df_id = df_full[df_full[col_label] != held_out_class].copy()

    if df_ood.empty:
        raise ValueError(
            f"No images found for class {held_out_class}: check that it is the "
            "original 1-based class number from the CSV."
        )

    # Remap the remaining classes to a contiguous 1..(N-1) range
    # (e.g. excluding class 5 from 1..20 gives 1..19 instead of 1,2,3,4,6,...,20).
    remaining_classes = sorted(df_id[col_label].unique())
    remap = {old: new for new, old in enumerate(remaining_classes, start=1)}
    df_id[col_label] = df_id[col_label].map(remap)
    num_classes_id = len(remaining_classes)

    print(
        f"[leave-out class {held_out_class}] remaining classes: {num_classes_id} "
        f"({len(df_ood)} images of class {held_out_class} held out as OOD)"
    )

    set_seed(seed)
    train_df, val_df, test_df = split_by_samples(df_id, seed=seed)

    train_transform, val_transform = get_transforms()

    train_ds = FabricsDataset(train_df, img_dir=config.IMG_DIR, transform=train_transform)
    val_ds = FabricsDataset(val_df, img_dir=config.IMG_DIR, transform=val_transform)
    test_ds = FabricsDataset(test_df, img_dir=config.IMG_DIR, transform=val_transform)
    # The OOD set is a test set: deterministic transform, no augmentation
    ood_ds = FabricsDataset(df_ood, img_dir=config.IMG_DIR, transform=val_transform)

    loader_kwargs = dict(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(
        f"Images - Train: {len(train_ds)}, Val: {len(val_ds)}, "
        f"Test (ID): {len(test_ds)}, OOD (class {held_out_class}): {len(ood_ds)}"
    )

    return train_loader, val_loader, train_ds, val_ds, test_ds, ood_ds, num_classes_id


# -- Training one leave-one-out model -------------------------------------------

def loco_checkpoint_dir(held_out_class: int, seed: int) -> str:
    """The config seed has no suffix; other seeds get _seed<n>."""
    suffix = "" if seed == config.SEED else f"_seed{seed}"
    return os.path.join(config.CHECKPOINT_DIR, f"leave_out_{held_out_class}{suffix}")


def train_loco_model(
    held_out_class: int,
    train_loader,
    val_loader,
    train_ds,
    num_classes_id: int,
    device,
    epochs_override: int | None,
    force_retrain: bool = False,
    seed: int = config.SEED,
):
    exp_dir = loco_checkpoint_dir(held_out_class, seed)
    os.makedirs(exp_dir, exist_ok=True)
    best_model_path = os.path.join(exp_dir, "best_dynamicvit.pth")
    metrics_csv = os.path.join(exp_dir, "metrics.csv")

    if os.path.exists(best_model_path) and not force_retrain:
        print(
            f"Checkpoint already present at {best_model_path}: skipping training and loading "
            "the existing weights (use --force_retrain to retrain anyway)."
        )
        return load_model(best_model_path, num_classes=num_classes_id, device=device), exp_dir

    model = build_model(num_classes=num_classes_id, device=device)
    class_weights = get_class_weights(train_ds, device)

    orig_epochs = config.NUM_EPOCHS
    if epochs_override is not None:
        config.NUM_EPOCHS = epochs_override
    try:
        train(model, train_loader, val_loader, class_weights, device,
              best_model_path=best_model_path, metrics_csv=metrics_csv)
    finally:
        config.NUM_EPOCHS = orig_epochs

    # train() returns the model at the LAST epoch: reload the best checkpoint.
    return load_model(best_model_path, num_classes=num_classes_id, device=device), exp_dir


# -- Threshold-free detection metrics -------------------------------------------

def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
    """FPR at the most permissive threshold that reaches at least tpr_target
    on the positives (y_true == 1, truly OOD). Higher score = more likely OOD."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg

    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def aggregate_scores_by_sample(df_scores: pd.DataFrame) -> pd.DataFrame:
    """Per-sample mean of the scores (one row per image -> one row per sample)."""
    return (
        df_scores.groupby("sample_id")
        .agg(
            msp=("msp", "mean"),
            entropy=("entropy", "mean"),
            energy=("energy", "mean"),
            odin=("odin", "mean"),
            mahalanobis=("mahalanobis", "mean"),
        )
        .reset_index()
    )


# +1 if a higher value already means "more likely OOD", -1 if it must be
# inverted (msp and odin measure confidence).
SCORE_DIRECTIONS = {"msp": -1, "odin": -1, "entropy": 1, "energy": 1, "mahalanobis": 1}


def orient_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add msp_neg and odin_neg, i.e. the scores oriented so that
    HIGH = more likely new, with the same names used by run_ood_cv.py."""
    df = df.copy()
    for name, direction in SCORE_DIRECTIONS.items():
        if direction == -1 and name in df.columns:
            df[name + "_neg"] = -df[name]
    return df


def evaluate_ood_detection(id_scores: pd.DataFrame, ood_scores: pd.DataFrame) -> dict:
    id_scores = id_scores.copy()
    id_scores["y_true"] = 0
    ood_scores = ood_scores.copy()
    ood_scores["y_true"] = 1
    combined = pd.concat([id_scores, ood_scores], ignore_index=True)

    y_true = combined["y_true"].values
    results = {}
    for score_name, direction in SCORE_DIRECTIONS.items():
        s = direction * combined[score_name].values
        results[score_name] = dict(
            auroc=roc_auc_score(y_true, s),
            aupr=average_precision_score(y_true, s),
            fpr95=fpr_at_tpr(y_true, s, 0.95),
        )
    return results


# -- Single experiment ----------------------------------------------------------

def run_experiment(
    held_out_class: int,
    device,
    epochs_override: int | None,
    force_retrain: bool = False,
    seed: int = config.SEED,
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"LEAVE-ONE-FABRIC-OUT - held-out class: {held_out_class}  (seed {seed})")
    print("=" * 70)

    df_full = load_full_dataframe()
    (train_loader, val_loader, train_ds, val_ds, test_ds, ood_ds, num_classes_id) = build_loco_dataloaders(
        df_full, held_out_class, seed=seed
    )

    model, exp_dir = train_loco_model(
        held_out_class, train_loader, val_loader, train_ds, num_classes_id,
        device, epochs_override, force_retrain, seed=seed,
    )

    _, val_transform = get_transforms()
    clean_train_ds = make_clean_copy(train_ds, transform=val_transform)

    means, inv_cov = compute_id_statistics(model, clean_train_ds, device, num_classes_id)
    threshold, val_sample_distances = compute_ood_threshold(model, val_ds, device, means, inv_cov)
    # The validation set goes through the same function as test and OOD, so
    # that every score can be thresholded on it (operating_point_eval.py).
    val_df = analyze_ood_dataset(model, val_ds, device, means, inv_cov)
    id_df = analyze_ood_dataset(model, test_ds, device, means, inv_cov)
    ood_df = analyze_ood_dataset(model, ood_ds, device, means, inv_cov)

    with open(os.path.join(exp_dir, f"threshold_{COVARIANCE}.json"), "w") as f:
        json.dump({"threshold": threshold, "percentile": config.OOD_PERCENTILE}, f, indent=2)

    val_scores = orient_scores(aggregate_scores_by_sample(val_df))
    val_scores.to_csv(os.path.join(exp_dir, f"val_scores_{COVARIANCE}.csv"), index=False)

    # Consistency check: the validation Mahalanobis computed here must match
    # the one used for the threshold.
    expected = val_sample_distances.reindex(val_scores["sample_id"]).to_numpy()
    max_diff = np.nanmax(np.abs(expected - val_scores["mahalanobis"].to_numpy()))
    if max_diff > 1e-6:
        print(f"WARNING: the validation Mahalanobis computed twice differs by up to "
              f"{max_diff:.2e}: the two paths are not seeing the same data.")

    id_scores = orient_scores(aggregate_scores_by_sample(id_df))
    ood_scores = orient_scores(aggregate_scores_by_sample(ood_df))
    metrics = evaluate_ood_detection(id_scores, ood_scores)

    print(
        f"\nOOD detection metrics - held-out class {held_out_class} "
        f"({len(ood_scores)} OOD samples vs {len(id_scores)} known ID samples):"
    )
    for score_name, vals in metrics.items():
        print(
            f"  {score_name:12s}  AUROC={vals['auroc']:.3f}  "
            f"AUPR={vals['aupr']:.3f}  FPR@95TPR={vals['fpr95']:.3f}"
        )

    id_scores.to_csv(os.path.join(exp_dir, f"id_scores_{COVARIANCE}.csv"), index=False)
    ood_scores.to_csv(os.path.join(exp_dir, f"ood_scores_{COVARIANCE}.csv"), index=False)

    result_df = pd.DataFrame([
        dict(held_out_class=held_out_class, seed=seed, covariance=COVARIANCE, score=k, **v)
        for k, v in metrics.items()
    ])
    result_df.to_csv(os.path.join(exp_dir, "loco_ood_metrics.csv"), index=False)
    return result_df


def plot_comparison(all_results: pd.DataFrame, out_dir: str):
    """AUROC per score and held-out class (mean over seeds)."""
    scores = list(SCORE_DIRECTIONS.keys())
    classes = sorted(all_results["held_out_class"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(scores)
    x = np.arange(len(classes))

    for i, score in enumerate(scores):
        vals = [
            all_results[(all_results.held_out_class == c) & (all_results.score == score)]["auroc"].mean()
            for c in classes
        ]
        ax.bar(x + i * width, vals, width=width, label=score)

    ax.set_xticks(x + width * (len(scores) - 1) / 2)
    ax.set_xticklabels([f"Class {c} held out" for c in classes])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.set_title("AUROC per OOD score and held-out class")
    ax.legend(title="Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loco_auroc_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+", required=True,
                        help="Classes to hold out one at a time (original 1-based numbering, e.g. 1 16)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[config.SEED],
                        help="Repeat the experiment with several seeds (different split and "
                             "initialisation) to report mean and standard deviation.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.NUM_EPOCHS for this run (useful for a quick test).")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if a checkpoint already exists.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = os.path.join(config.REPO_DIR, "outputs", "loco_ood")
    os.makedirs(out_dir, exist_ok=True)

    all_results = pd.concat(
        [run_experiment(cls, device, args.epochs, args.force_retrain, seed=seed)
         for cls in args.classes for seed in args.seeds],
        ignore_index=True,
    )
    all_results.to_csv(os.path.join(out_dir, "loco_ood_all_results.csv"), index=False)
    plot_comparison(all_results, out_dir)

    if len(args.seeds) > 1:
        summary = (
            all_results.groupby(["held_out_class", "score"])
            .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", lambda x: x.std(ddof=1)),
                 aupr_mean=("aupr", "mean"), aupr_sd=("aupr", lambda x: x.std(ddof=1)),
                 fpr95_mean=("fpr95", "mean"), fpr95_sd=("fpr95", lambda x: x.std(ddof=1)),
                 n_seed=("auroc", "size"))
            .reset_index()
            .sort_values(["held_out_class", "auroc_mean"], ascending=[True, False])
            .round(4)
        )
        summary.to_csv(os.path.join(out_dir, "loco_ood_summary.csv"), index=False)
        print(f"\n{'=' * 70}")
        print(f"SUMMARY - mean +- std over {len(args.seeds)} seeds")
        print("=" * 70)
        print(summary.to_string(index=False))

    print(f"\nDone - results saved to {out_dir}/")
    print(all_results.to_string(index=False))
