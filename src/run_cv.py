"""
K-fold cross-validation, grouped by sample and stratified by fabric.

For each fold k:
    test        = fold k                              (~1/K of the samples)
    validation  = ~15% of the remaining samples, stratified by fabric
    training    = everything else

Every sample is used as test EXACTLY ONCE. Final metrics are computed on the
pooled out-of-fold predictions (i.e. over all samples) rather than by
averaging per-fold metrics: with classes that have only a few samples, the
mean of per-fold macro-F1 scores is unstable. Fold-to-fold variability is
reported separately.

Folds are built on sample_id, not on images, so images of the same sample
never end up in different subsets.

Usage (from src/):
    python run_cv.py --name whole_halves
    python run_cv.py --name whole_halves --only-fold 2   # run a single fold

Outputs in outputs/cv/<name>/:
    fold<k>/best_model.pth, fold<k>/train_metrics.csv
    dataset_composition.csv        samples and images per fabric
    per_sample_predictions.csv     one row per sample (all samples)
    per_image_predictions.csv      one row per image
    per_fold_metrics.csv           per-fold metrics
    summary.csv                    overall metrics (pooled predictions, Table 2)
The per-fabric classification report (Table 3) is printed at the end.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import config
from dataset_samples import FabricsDataset, collate_fn, describe_preprocessing, get_transforms, load_dataset_csv
from inference_utils import aggregate_by_sample, apply_temperature, collect_logits, fit_temperature
from model import build_model
from train import get_class_weights, train


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inner_val_split(samples: pd.DataFrame, col_label: str, val_frac: float, rng) -> set:
    """From the non-test samples, pick ~val_frac per class as validation,
    always leaving at least one sample per class in training."""
    val_ids = []
    for _, grp in samples.groupby(col_label):
        ids = grp["sample_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_val = max(1, int(round(val_frac * len(ids))))
        n_val = min(n_val, len(ids) - 1)
        val_ids.extend(ids[:n_val])
    return set(val_ids)


def make_folds(df: pd.DataFrame, col_label: str, n_folds: int, seed: int, val_frac: float):
    """Yield (k, train_ids, val_ids, test_ids) for each fold. Also used by
    run_ood_cv.py, so both scripts always see the same split."""
    samples = df.groupby("sample_id")[col_label].first().reset_index()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (rest_idx, test_idx) in enumerate(skf.split(samples["sample_id"], samples[col_label])):
        test_ids = set(samples.iloc[test_idx]["sample_id"])
        rest = samples.iloc[rest_idx]
        rng = np.random.default_rng(seed + k)
        val_ids = inner_val_split(rest, col_label, val_frac, rng)
        train_ids = set(rest["sample_id"]) - val_ids
        assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
        yield k, train_ids, val_ids, test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name (e.g. whole_halves)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED, help="seed for fold assignment and training")
    parser.add_argument("--only-fold", type=int, default=None, help="run only this fold (0-based)")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.label_smoothing is not None:
        config.LABEL_SMOOTHING = args.label_smoothing
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs

    img_dir = config.IMG_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(config.OUTPUT_DIR, "cv", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # -- Full dataset and sample table ----------------------------------------
    df = load_dataset_csv()
    col_img, col_label = df.columns[0], df.columns[1]

    counts = (
        df.groupby(col_label)
        .agg(samples=("sample_id", "nunique"), images=(col_img, "count"))
    )
    print(f"Preprocessing: {describe_preprocessing()}")
    print(f"Label smoothing: {config.LABEL_SMOOTHING}")
    counts.to_csv(os.path.join(out_dir, "dataset_composition.csv"))

    if counts.samples.min() < args.folds:
        raise SystemExit(
            f"The smallest fabric has {counts.samples.min()} samples but {args.folds} folds were requested: "
            f"use --folds {counts.samples.min()} or fewer."
        )

    num_classes = df[col_label].nunique()
    per_image_rows, per_sample_rows, fold_rows = [], [], []

    for k, train_ids, val_ids, test_ids in make_folds(df, col_label, args.folds, args.seed, args.val_frac):
        if args.only_fold is not None and k != args.only_fold:
            continue

        fold_dir = os.path.join(out_dir, f"fold{k}")
        os.makedirs(fold_dir, exist_ok=True)
        ckpt = os.path.join(fold_dir, "best_model.pth")
        metrics_csv = os.path.join(fold_dir, "train_metrics.csv")
        if os.path.exists(metrics_csv):
            os.remove(metrics_csv)

        train_df = df[df["sample_id"].isin(train_ids)].copy()
        val_df = df[df["sample_id"].isin(val_ids)].copy()
        test_df = df[df["sample_id"].isin(test_ids)].copy()

        train_tf, val_tf = get_transforms()
        train_ds = FabricsDataset(train_df, img_dir=img_dir, transform=train_tf)
        val_ds = FabricsDataset(val_df, img_dir=img_dir, transform=val_tf)
        test_ds = FabricsDataset(test_df, img_dir=img_dir, transform=val_tf)

        print(f"\n{'=' * 62}\nFOLD {k + 1}/{args.folds}\n{'=' * 62}")
        print(f"Samples - train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")
        print(f"Images  - train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

        loader_kwargs = dict(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                             pin_memory=True, collate_fn=collate_fn)
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        set_all_seeds(args.seed + k)
        model = build_model(num_classes=num_classes, device=device)
        train(model, train_loader, val_loader, get_class_weights(train_ds, device), device,
              best_model_path=ckpt, metrics_csv=metrics_csv)

        # -- Out-of-fold evaluation -------------------------------------------
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        val_rec = collect_logits(model, val_ds, device)
        test_rec = collect_logits(model, test_ds, device)
        val_logits = np.stack([r["logits"] for r in val_rec])
        val_labels = np.array([r["label"] for r in val_rec])
        test_logits = np.stack([r["logits"] for r in test_rec])
        test_labels = np.array([r["label"] for r in test_rec])

        T = fit_temperature(val_logits, val_labels)
        probs_cal = apply_temperature(test_logits, T)

        for r, p in zip(test_rec, probs_cal):
            per_image_rows.append(dict(fold=k, sample_id=r["sample_id"], polarization=r["polarization"],
                                       true_label=r["label"], pred=int(p.argmax()),
                                       confidence=float(p.max())))

        sample_df = aggregate_by_sample([dict(r, probs=p) for r, p in zip(test_rec, probs_cal)])
        sample_df["fold"] = k
        per_sample_rows.append(sample_df)

        img_acc = accuracy_score(test_labels, test_logits.argmax(1))
        img_f1 = f1_score(test_labels, test_logits.argmax(1), average="macro", zero_division=0)
        s_acc = accuracy_score(sample_df["true_label"], sample_df["soft_pred"])
        s_f1 = f1_score(sample_df["true_label"], sample_df["soft_pred"], average="macro", zero_division=0)
        fold_rows.append(dict(fold=k, n_test_samples=len(sample_df), n_test_images=len(test_labels),
                              temperature=round(T, 4),
                              img_acc=round(img_acc, 4), img_f1=round(img_f1, 4),
                              sample_acc=round(s_acc, 4), sample_f1=round(s_f1, 4)))
        print(f"-> fold {k}: image acc {img_acc:.4f} F1 {img_f1:.4f}  |  "
              f"sample acc {s_acc:.4f} F1 {s_f1:.4f}")

        del model
        torch.cuda.empty_cache()

    if args.only_fold is not None:
        print("\nSingle fold run: overall results are computed once all folds are available.")
        return

    # -- Pooled predictions across folds --------------------------------------
    images = pd.DataFrame(per_image_rows)
    samples_pred = pd.concat(per_sample_rows, ignore_index=True)
    images.to_csv(os.path.join(out_dir, "per_image_predictions.csv"), index=False)
    samples_pred.to_csv(os.path.join(out_dir, "per_sample_predictions.csv"), index=False)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(os.path.join(out_dir, "per_fold_metrics.csv"), index=False)

    yi, pi = images["true_label"], images["pred"]
    ys, ps = samples_pred["true_label"], samples_pred["soft_pred"]
    yh = samples_pred["hard_pred"]

    overall = dict(
        name=args.name,
        folds=args.folds,
        preprocessing=describe_preprocessing(),
        label_smoothing=config.LABEL_SMOOTHING,
        n_images=len(images),
        n_samples=len(samples_pred),
        img_acc=round(accuracy_score(yi, pi), 4),
        img_f1=round(f1_score(yi, pi, average="macro", zero_division=0), 4),
        sample_acc_soft=round(accuracy_score(ys, ps), 4),
        sample_f1_soft=round(f1_score(ys, ps, average="macro", zero_division=0), 4),
        sample_acc_hard=round(accuracy_score(ys, yh), 4),
        sample_f1_hard=round(f1_score(ys, yh, average="macro", zero_division=0), 4),
        img_acc_std_fold=round(fold_metrics["img_acc"].std(ddof=1), 4),
        sample_acc_std_fold=round(fold_metrics["sample_acc"].std(ddof=1), 4),
        sample_f1_std_fold=round(fold_metrics["sample_f1"].std(ddof=1), 4),
    )
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\nPer-fabric classification report (sample level, soft voting, all folds):")
    print(classification_report(ys, ps, zero_division=0))
    print("Confusion matrix (sample level, soft voting, all folds):")
    print(confusion_matrix(ys, ps))

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
