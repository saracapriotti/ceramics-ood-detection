"""
Out-of-distribution detection with the K models of the cross-validation.

For each of the K folds produced by run_cv.py:
  1. rebuild EXACTLY the same split (run_cv.make_folds);
  2. load outputs/cv/<name>/fold<k>/best_model.pth;
  3. estimate class means and the shared within-class covariance on the
     fold's TRAINING set, without augmentation;
  4. calibrate the threshold on the fold's VALIDATION set
     (percentile config.OOD_PERCENTILE);
  5. compute nine scores on three sets: out-of-fold test (known samples),
     OOD dataset (true loners + known-fabric samples photographed outside
     the training dataset) and validation (for standardisation).

Nine scores, all oriented the same way (HIGH = likely new):
    msp_neg, entropy, energy, odin_neg, mahalanobis      (from ood_utils)
    elliptic_envelope, isolation_forest, lof, ocsvm      (from sklearn_ood_baselines)

Two evaluation protocols, reported separately:
  A) negatives = KNOWN-fabric samples of the OOD dataset. They share site,
     preparation and imaging session with the loners, so separation cannot
     be explained by an acquisition artefact.
  B) negatives = out-of-fold test samples of the training dataset. Easier
     and more optimistic, reported for comparison only.

Every metric (AUROC, AUPR, FPR@95TPR) is reported as mean +- std over the
K folds.

Usage:
    python run_ood_cv.py --cv-name whole_halves \
        --ground-truth ../data/OOD-DATASET_ground_truth.csv

    python run_ood_cv.py --cv-name whole_halves --skip-sklearn   # faster

Outputs in outputs/ood/<cv-name>/.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import config
from dataset_samples import describe_preprocessing, get_transforms, load_dataset_csv
from ood_ground_truth_eval import NOVEL_SENTINEL
from ood_utils import STATUS_KNOWN, STATUS_NEW
from run_cv import make_folds

# Score name -> (column in the per-image dataframe, invert?)
# After inversion, for ALL scores: high = likely new.
BASE_SCORES = {
    "msp_neg": ("msp", True),
    "entropy": ("entropy", False),
    "energy": ("energy", False),
    "odin_neg": ("odin", True),
    "mahalanobis": ("mahalanobis", False),
}
SKLEARN_SCORES = ["elliptic_envelope", "isolation_forest", "lof", "ocsvm"]

DEFAULT_PERCENTILES = [70, 75, 80, 85, 90, 95, 99]


# -- Metrics --------------------------------------------------------------------

def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
    """Fraction of KNOWN samples wrongly flagged at the threshold that
    catches tpr_target of the truly new samples."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr, fpr = tp / n_pos, fp / n_neg
    idx = int(np.searchsorted(tpr, tpr_target))
    return float(fpr[-1]) if idx >= len(fpr) else float(fpr[idx])


def score_metrics(neg_scores: np.ndarray, pos_scores: np.ndarray) -> dict:
    """neg = known samples, pos = truly new samples."""
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return dict(auroc=np.nan, aupr=np.nan, fpr95=np.nan)
    y = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    s = np.concatenate([neg_scores, pos_scores])
    return dict(
        auroc=float(roc_auc_score(y, s)),
        aupr=float(average_precision_score(y, s)),
        fpr95=fpr_at_tpr(y, s, 0.95),
    )


# -- Per-sample aggregation -----------------------------------------------------

def per_sample_scores(img_df: pd.DataFrame, keep_features: bool = False) -> pd.DataFrame:
    """Per-sample mean over all images (PPL + XPL, all magnifications), with
    every score oriented the same way."""
    cols = [c for c, _ in BASE_SCORES.values()]
    agg = img_df.groupby("sample_id")[cols].mean().reset_index()
    agg["n_images"] = img_df.groupby("sample_id").size().to_numpy()

    out = pd.DataFrame({"sample_id": agg["sample_id"], "n_images": agg["n_images"]})
    for name, (col, invert) in BASE_SCORES.items():
        out[name] = -agg[col].to_numpy() if invert else agg[col].to_numpy()
    # raw distance, also needed for the threshold and the report
    out["mahalanobis_raw"] = agg["mahalanobis"].to_numpy()

    if "pred_class_idx" in img_df.columns:
        mode = img_df.groupby("sample_id")["pred_class_idx"].agg(lambda x: x.mode().iloc[0])
        out["pred_class_idx"] = mode.reindex(out["sample_id"]).to_numpy()

    if keep_features and "feature_vector" in img_df.columns:
        feats = (
            img_df.groupby("sample_id")["feature_vector"]
            .apply(lambda v: np.stack(v.tolist()).mean(axis=0))
            .reindex(out["sample_id"])
        )
        out["feature_vector"] = list(feats.to_numpy())
    return out


def robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Standardise a score with the median and IQR of the fold's VALIDATION
    set: scores of different folds have different scales."""
    med = np.median(reference)
    iqr = np.subtract(*np.percentile(reference, [75, 25]))
    return (values - med) / (iqr if iqr > 1e-12 else 1.0)


# -- One fold -------------------------------------------------------------------

def process_fold(k, train_ids, val_ids, test_ids, df, col_label, fold_dir, num_classes,
                 device, args, transforms_pair, img_dir, cache_dir):
    from dataset_samples import FabricsDataset
    from model import load_model
    from ood_utils import (
        OODDataset,
        analyze_ood_dataset,
        compute_id_statistics,
        extract_features_and_logits,
        make_clean_copy,
    )

    _, val_tf = transforms_pair
    ckpt = os.path.join(fold_dir, "best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint for fold {k}: {ckpt}")

    print(f"\n{'=' * 70}\nFOLD {k}\n{'=' * 70}")

    train_df = df[df["sample_id"].isin(train_ids)]
    val_df = df[df["sample_id"].isin(val_ids)]
    test_df = df[df["sample_id"].isin(test_ids)]

    train_ds = FabricsDataset(train_df, img_dir=img_dir, transform=val_tf)  # no augmentation
    val_ds = FabricsDataset(val_df, img_dir=img_dir, transform=val_tf)
    test_ds = FabricsDataset(test_df, img_dir=img_dir, transform=val_tf)
    ood_ds = OODDataset(config.OOD_CSV_FILE, config.OOD_IMG_DIR, transform=val_tf)

    model = load_model(ckpt, num_classes=num_classes, device=device)
    clean_train_ds = make_clean_copy(train_ds, transform=val_tf)

    # Training features are extracted ONCE: they are needed both for the
    # Mahalanobis statistics and for the sklearn detectors.
    train_rec = extract_features_and_logits(model, clean_train_ds, device, args.batch_size)
    means, inv_cov = compute_id_statistics(model, clean_train_ds, device, num_classes,
                                           batch_size=args.batch_size, records=train_rec)

    print("Scoring validation, out-of-fold test and OOD dataset...")
    val_img = analyze_ood_dataset(model, val_ds, device, means, inv_cov, batch_size=args.batch_size)
    test_img = analyze_ood_dataset(model, test_ds, device, means, inv_cov, batch_size=args.batch_size)
    ood_img = analyze_ood_dataset(model, ood_ds, device, means, inv_cov, batch_size=args.batch_size)

    val_s = per_sample_scores(val_img)
    test_s = per_sample_scores(test_img)
    ood_s = per_sample_scores(ood_img, keep_features=True)

    threshold = float(np.percentile(val_s["mahalanobis_raw"], args.percentile))
    print(f"Mahalanobis threshold ({args.percentile}th percentile over {len(val_s)} validation samples): "
          f"{threshold:.3f}")

    # Per-sample features, at the same aggregation level for every set
    train_feats = np.stack([r["feature"] for r in train_rec])
    train_sids = [r["sample_id"] for r in train_rec]
    train_by_sample = (
        pd.DataFrame(train_feats).assign(sample_id=train_sids).groupby("sample_id").mean()
    )

    def feats_of(df_img, table):
        m = (
            df_img.groupby("sample_id")["feature_vector"]
            .apply(lambda v: np.stack(v.tolist()).mean(axis=0))
            .reindex(table["sample_id"])
        )
        return np.stack(m.to_numpy())

    X = {"train": train_by_sample.to_numpy(), "val": feats_of(val_img, val_s),
         "test": feats_of(test_img, test_s), "ood": feats_of(ood_img, ood_s)}

    # Feature cache, read by consensus.py and all_figures.py (Figure 5):
    # no need to reload the models or the images.
    np.savez_compressed(
        os.path.join(cache_dir, f"features_fold{k}.npz"),
        train=X["train"], val=X["val"], test=X["test"], ood=X["ood"],
        train_ids=np.array(list(train_by_sample.index), dtype=object),
        val_ids=val_s["sample_id"].to_numpy(dtype=object),
        test_ids=test_s["sample_id"].to_numpy(dtype=object),
        ood_ids=ood_s["sample_id"].to_numpy(dtype=object),
    )

    # sklearn baselines: fit and score BOTH on per-sample features
    if not args.skip_sklearn:
        from sklearn_ood_baselines import fit_score_detectors

        sk_scores = fit_score_detectors(
            X["train"], {k_: X[k_] for k_ in ("val", "test", "ood")},
            pca_components=args.pca_components,
        )
        for name in SKLEARN_SCORES:
            val_s[name] = sk_scores["val"][name]
            test_s[name] = sk_scores["test"][name]
            ood_s[name] = sk_scores["ood"][name]

    score_names = list(BASE_SCORES) + ([] if args.skip_sklearn else SKLEARN_SCORES)

    # Fold-independent standardisation, so that scores can be averaged across folds
    for name in score_names:
        ref = val_s[name].to_numpy()
        for table in (val_s, test_s, ood_s):
            table[name + "_z"] = robust_z(table[name].to_numpy(), ref)
    ood_s["mahalanobis_ratio"] = ood_s["mahalanobis_raw"] / threshold
    test_s["mahalanobis_ratio"] = test_s["mahalanobis_raw"] / threshold

    for table, split in ((val_s, "val"), (test_s, "test_id"), (ood_s, "ood")):
        table.insert(0, "fold", k)
        table.insert(1, "split", split)

    del model
    torch.cuda.empty_cache()
    return dict(fold=k, threshold=threshold, val=val_s, test=test_s, ood=ood_s,
                score_names=score_names)


# -- Evaluation against the ground truth ----------------------------------------

def evaluate_folds(fold_results, gt: pd.DataFrame, score_names) -> pd.DataFrame:
    """AUROC/AUPR/FPR95 for every fold, score and protocol."""
    rows = []
    novel_ids = set(gt.loc[gt["true_class"] == NOVEL_SENTINEL, "sample_id"])
    known_ood_ids = set(gt.loc[gt["true_class"] != NOVEL_SENTINEL, "sample_id"])

    for res in fold_results:
        ood, test = res["ood"], res["test"]
        pos = ood[ood["sample_id"].isin(novel_ids)]
        neg_a = ood[ood["sample_id"].isin(known_ood_ids)]
        for name in score_names:
            rows.append(dict(fold=res["fold"], protocol="A_ood_known", score=name,
                             n_neg=len(neg_a), n_pos=len(pos),
                             **score_metrics(neg_a[name].to_numpy(), pos[name].to_numpy())))
            rows.append(dict(fold=res["fold"], protocol="B_test_out_of_fold", score=name,
                             n_neg=len(test), n_pos=len(pos),
                             **score_metrics(test[name].to_numpy(), pos[name].to_numpy())))
    return pd.DataFrame(rows)


def summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    g = per_fold.groupby(["protocol", "score"])
    out = g.agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", lambda x: x.std(ddof=1)),
                aupr_mean=("aupr", "mean"), aupr_sd=("aupr", lambda x: x.std(ddof=1)),
                fpr95_mean=("fpr95", "mean"), fpr95_sd=("fpr95", lambda x: x.std(ddof=1)),
                n_neg=("n_neg", "first"), n_pos=("n_pos", "first")).reset_index()
    return out.sort_values(["protocol", "auroc_mean"], ascending=[True, False]).round(4)


def threshold_sweep(fold_results, gt: pd.DataFrame, percentiles) -> pd.DataFrame:
    """Loner recall and false alarms at each threshold (Mahalanobis)."""
    novel_ids = set(gt.loc[gt["true_class"] == NOVEL_SENTINEL, "sample_id"])
    known_ood_ids = set(gt.loc[gt["true_class"] != NOVEL_SENTINEL, "sample_id"])
    rows = []
    for res in fold_results:
        ref = res["val"]["mahalanobis_raw"].to_numpy()
        pos = res["ood"][res["ood"]["sample_id"].isin(novel_ids)]["mahalanobis_raw"].to_numpy()
        neg = res["ood"][res["ood"]["sample_id"].isin(known_ood_ids)]["mahalanobis_raw"].to_numpy()
        neg_id = res["test"]["mahalanobis_raw"].to_numpy()
        for p in percentiles:
            t = float(np.percentile(ref, p))
            rows.append(dict(
                fold=res["fold"], percentile=p, threshold=round(t, 3),
                recall_loner_pct=round(100 * (pos > t).mean(), 1) if len(pos) else np.nan,
                fp_known_ood_pct=round(100 * (neg > t).mean(), 1) if len(neg) else np.nan,
                fp_test_id_pct=round(100 * (neg_id > t).mean(), 1),
            ))
    df = pd.DataFrame(rows)
    return df.groupby("percentile").agg(
        threshold_mean=("threshold", "mean"),
        recall_loner_pct=("recall_loner_pct", "mean"),
        recall_sd=("recall_loner_pct", lambda x: x.std(ddof=1)),
        fp_known_ood_pct=("fp_known_ood_pct", "mean"),
        fp_test_id_pct=("fp_test_id_pct", "mean"),
    ).reset_index().round(2)


# -- Per-sample report pooling the folds ----------------------------------------

def pooled_report(fold_results, score_names, gt=None) -> pd.DataFrame:
    """A sample is flagged if it is flagged in the MAJORITY of folds."""
    majority = len(fold_results) // 2 + 1
    ood_all = pd.concat([r["ood"] for r in fold_results], ignore_index=True)
    agg = {name + "_z": (name + "_z", "mean") for name in score_names}
    agg.update(mahalanobis_ratio=("mahalanobis_ratio", "mean"),
               n_fold_flagged=("mahalanobis_ratio", lambda x: int((x > 1).sum())),
               n_images=("n_images", "first"))
    report = ood_all.groupby("sample_id").agg(**agg).reset_index()

    pred = (ood_all.groupby("sample_id")["pred_class_idx"]
            .agg(lambda x: x.mode().iloc[0]).reindex(report["sample_id"]))
    report["predicted_class"] = pred.to_numpy().astype(int) + 1
    report["status"] = np.where(report["n_fold_flagged"] >= majority, STATUS_NEW, STATUS_KNOWN)
    report["mahalanobis_mean"] = report["mahalanobis_ratio"]  # alias used by the plots

    if gt is not None:
        report = report.merge(gt, on="sample_id", how="left")
    return report.sort_values("mahalanobis_ratio", ascending=False).reset_index(drop=True)


# -- Plots ----------------------------------------------------------------------

def plot_score_separation(fold_results, score_names, gt, out_dir):
    novel_ids = set(gt.loc[gt["true_class"] == NOVEL_SENTINEL, "sample_id"])
    known_ood = set(gt.loc[gt["true_class"] != NOVEL_SENTINEL, "sample_id"])
    ood_all = pd.concat([r["ood"] for r in fold_results], ignore_index=True)
    test_all = pd.concat([r["test"] for r in fold_results], ignore_index=True)

    n = len(score_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, score_names):
        data = [
            test_all[name + "_z"].to_numpy(),
            ood_all.loc[ood_all["sample_id"].isin(known_ood), name + "_z"].to_numpy(),
            ood_all.loc[ood_all["sample_id"].isin(novel_ids), name + "_z"].to_numpy(),
        ]
        try:
            bp = ax.boxplot(data, tick_labels=["in-domain test", "known fabric", "loner"],
                            patch_artist=True, widths=0.6, showfliers=False)
        except TypeError:  # matplotlib < 3.9
            bp = ax.boxplot(data, labels=["in-domain test", "known fabric", "loner"],
                            patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], ["#BFD3E6", "#6BAED6", "#08519C"]):
            patch.set_facecolor(c)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Scores standardised on each fold validation set (higher = more novel)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_separation.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(sweep: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(sweep["fp_known_ood_pct"], sweep["recall_loner_pct"], marker="o", color="#08519C",
            label="false alarms among known-fabric specimens")
    ax.plot(sweep["fp_test_id_pct"], sweep["recall_loner_pct"], marker="s", color="#6BAED6",
            linestyle="--", label="false alarms on in-domain test specimens")
    for _, r in sweep.iterrows():
        ax.annotate(f"{int(r['percentile'])}th", (r["fp_known_ood_pct"], r["recall_loner_pct"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("False alarms (%)")
    ax.set_ylabel("Loners recovered (%)")
    ax.set_title("Loner recovery against false alarms")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(out_dir, f"threshold_sweep.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-name", required=True,
                        help="name of the run_cv.py run (folder outputs/cv/<name>)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--ground-truth", default=None,
                        help="CSV with columns sample_id;true_class")
    parser.add_argument("--percentile", type=float, default=config.OOD_PERCENTILE)
    parser.add_argument("--percentiles", type=float, nargs="+", default=DEFAULT_PERCENTILES)
    parser.add_argument("--pca-components", type=int, default=50,
                        help="PCA components before the sklearn detectors (required at sample "
                             "level: fewer samples than dimensions)")
    parser.add_argument("--skip-sklearn", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    img_dir = config.IMG_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_dir = os.path.join(config.OUTPUT_DIR, "cv", args.cv_name)
    out_dir = os.path.join(config.OUTPUT_DIR, "ood", args.cv_name)
    cache_dir = os.path.join(out_dir, "features")
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 70)
    print(f"OOD detection on cross-validation - run '{args.cv_name}'")
    print("=" * 70)
    print(f"Preprocessing: {describe_preprocessing()}")
    print(f"Models: {cv_dir}/fold*/best_model.pth")
    print(f"OOD dataset: {config.OOD_CSV_FILE}")
    print(f"OOD images: {config.OOD_IMG_DIR}")

    df = load_dataset_csv()
    col_label = df.columns[1]
    num_classes = df[col_label].nunique()
    transforms_pair = get_transforms()

    fold_results = []
    for k, train_ids, val_ids, test_ids in make_folds(df, col_label, args.folds,
                                                         args.seed, args.val_frac):
        if args.only_fold is not None and k != args.only_fold:
            continue
        fold_results.append(process_fold(k, train_ids, val_ids, test_ids, df, col_label,
                                         os.path.join(cv_dir, f"fold{k}"), num_classes,
                                         device, args, transforms_pair, img_dir, cache_dir))

    score_names = fold_results[0]["score_names"]

    pd.concat([r["ood"].drop(columns=["feature_vector"], errors="ignore") for r in fold_results],
              ignore_index=True).to_csv(os.path.join(out_dir, "ood_scores_per_fold.csv"), index=False)
    pd.concat([r["test"] for r in fold_results], ignore_index=True).to_csv(
        os.path.join(out_dir, "test_id_scores_per_fold.csv"), index=False)
    # Validation scores: used to recompute thresholds at other percentiles
    # without new forward passes (see operating_point_eval.py).
    pd.concat([r["val"] for r in fold_results], ignore_index=True).to_csv(
        os.path.join(out_dir, "val_scores_per_fold.csv"), index=False)
    pd.DataFrame([dict(fold=r["fold"], threshold=r["threshold"]) for r in fold_results]).to_csv(
        os.path.join(out_dir, "thresholds.csv"), index=False)

    gt = None
    if args.ground_truth:
        from ood_ground_truth_eval import evaluate_classification, evaluate_ood_detection, load_ground_truth
        gt = load_ground_truth(args.ground_truth)
        missing = set(gt["sample_id"]) - set(fold_results[0]["ood"]["sample_id"])
        if missing:
            print(f"\nWARNING: {len(missing)} ground-truth sample_id are not in the OOD dataset: "
                  f"{sorted(missing)}")

    report = pooled_report(fold_results, score_names, gt)
    report.drop(columns=["feature_vector"], errors="ignore").to_csv(
        os.path.join(out_dir, "ood_report.csv"), index=False)

    majority = len(fold_results) // 2 + 1
    print(f"\n{'=' * 70}\nFLAGGED SAMPLES (in at least {majority} of "
          f"{len(fold_results)} folds)\n{'=' * 70}")
    flagged = report[report["status"] == STATUS_NEW]
    cols = ["sample_id", "mahalanobis_ratio", "n_fold_flagged", "predicted_class", "n_images"]
    if gt is not None:
        cols.append("true_class")
    print(flagged[cols].to_string(index=False) if len(flagged) else "  none")

    if gt is not None:
        per_fold = evaluate_folds(fold_results, gt, score_names)
        per_fold.round(4).to_csv(os.path.join(out_dir, "ood_metrics_per_fold.csv"), index=False)
        summary = summarise(per_fold)
        summary.to_csv(os.path.join(out_dir, "ood_metrics_summary.csv"), index=False)

        print(f"\n{'=' * 70}\nOOD METRICS - mean +- std over {len(fold_results)} folds\n{'=' * 70}")
        for proto, grp in summary.groupby("protocol"):
            label = ("A) negatives = KNOWN-fabric samples of the OOD dataset"
                     if proto.startswith("A") else
                     "B) negatives = out-of-fold test samples of the training dataset")
            print(f"\n{label}   (n_neg={grp['n_neg'].iloc[0]}, n_loner={grp['n_pos'].iloc[0]})")
            print(f"{'score':<20}{'AUROC':>16}{'AUPR':>16}{'FPR@95TPR':>16}")
            for _, r in grp.iterrows():
                print(f"{r['score']:<20}"
                      f"{r['auroc_mean']:>9.3f} +-{r['auroc_sd']:.3f}"
                      f"{r['aupr_mean']:>9.3f} +-{r['aupr_sd']:.3f}"
                      f"{r['fpr95_mean']:>9.3f} +-{r['fpr95_sd']:.3f}")

        sweep = threshold_sweep(fold_results, gt, args.percentiles)
        sweep.to_csv(os.path.join(out_dir, "threshold_sweep.csv"), index=False)
        print(f"\n{'=' * 70}\nTHRESHOLD SWEEP (Mahalanobis, mean over folds)\n{'=' * 70}")
        print(sweep.to_string(index=False))
        plot_sweep(sweep, out_dir)
        plot_score_separation(fold_results, score_names, gt, out_dir)

        merged = report.dropna(subset=["true_class"])
        evaluate_classification(merged)
        evaluate_ood_detection(merged, out_dir, threshold=1.0)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
