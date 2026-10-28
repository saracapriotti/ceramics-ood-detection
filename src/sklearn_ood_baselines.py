"""
scikit-learn novelty-detection baselines on the 512-d features extracted
from model.norm. No retraining: existing weights are reused (the fold models
for run_ood_cv.py, the leave-one-class-out checkpoints for the benchmark
below).

All methods are used in "novelty detection" mode: fitted ONLY on known
features, then scored on unseen data.

  - EllipticEnvelope:   Gaussian with robust covariance (MinCovDet), close
                        to the Mahalanobis score but with a different
                        covariance estimate;
  - LocalOutlierFactor: k-nearest-neighbour density (novelty=True), no
                        Gaussian assumption;
  - IsolationForest:    tree-based partitioning;
  - OneClassSVM:        kernel-based baseline.

Two precautions are applied:

1. Fit and score at the SAME aggregation level. Fitting on per-image features
   and scoring per-sample means would compare distributions with different
   variance (averaging reduces it). fit_score_detectors() expects everything
   already aggregated at the same level.

2. Dimensionality. At sample level there are ~400 training samples for 512
   dimensions, a regime where MinCovDet is not even defined. PCA is therefore
   required; it is applied after StandardScaler and estimated on the
   training set only.

Library usage:
    from sklearn_ood_baselines import fit_score_detectors, fpr_at_tpr

Leave-one-class-out benchmark (uses the checkpoints produced by
run_loco_ood.py, which are loaded, not retrained):
    python sklearn_ood_baselines.py --classes 1 16 --seeds 100 101 102

Besides threshold-free metrics (AUROC, AUPR, FPR@95TPR), per-sample scores
are saved for the validation, in-distribution test and OOD sets, so that
the methods can also be compared at an operating point: the validation set
sets the threshold, the other two give recall and false alarms.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import config

# Imports from run_loco_ood, dataset_samples, model and ood_utils are done
# inside the functions that need them, so that build_detectors(),
# fit_score_detectors() and fpr_at_tpr() can be imported by run_ood_cv.py
# without loading the leave-one-class-out pipeline.

# Expected fraction of anomalies in the training set. 0.1 is the sklearn
# default and is used for ALL methods that accept it, to favour none of them.
CONTAMINATION = 0.1


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
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


def build_detectors() -> dict:
    """sklearn default hyperparameters, except contamination, which is the
    same for every method that accepts it (see CONTAMINATION)."""
    return {
        "elliptic_envelope": EllipticEnvelope(contamination=CONTAMINATION, random_state=config.SEED),
        "isolation_forest": IsolationForest(contamination=CONTAMINATION, random_state=config.SEED,
                                            n_estimators=200),
        "lof": LocalOutlierFactor(novelty=True, contamination=CONTAMINATION, n_neighbors=20),
        "ocsvm": OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAMINATION),
    }


def fit_score_detectors(train_features: np.ndarray, to_score: dict,
                        pca_components: int | None = 50, verbose: bool = True) -> dict:
    """Fit the four detectors on known features (training set, no
    augmentation) and score every set in `to_score`.

    train_features and the values of to_score must be at the SAME
    aggregation level (all per image, or all per sample).

    Returns {set_name: {detector_name: scores}}, oriented so that
    HIGH = likely new (sklearn's decision_function is the opposite).
    """
    scaler = StandardScaler().fit(train_features)
    train_s = scaler.transform(train_features)
    scored = {k: scaler.transform(v) for k, v in to_score.items()}

    n_train, dim = train_s.shape
    if pca_components:
        n_comp = int(min(pca_components, n_train - 1, dim))
        if n_comp < pca_components and verbose:
            print(f"  PCA reduced to {n_comp} components ({n_train} training samples available)")
        pca = PCA(n_components=n_comp, random_state=config.SEED).fit(train_s)
        train_s = pca.transform(train_s)
        scored = {k: pca.transform(v) for k, v in scored.items()}
        if verbose:
            print(f"  PCA with {n_comp} components: explained variance {pca.explained_variance_ratio_.sum():.1%}")
    elif n_train <= dim and verbose:
        print(f"  WARNING: {n_train} training samples for {dim} dimensions without PCA: "
              "EllipticEnvelope cannot be reliably estimated in this regime.")

    results = {k: {} for k in to_score}
    for name, det in build_detectors().items():
        try:
            det.fit(train_s)
        except Exception as e:  # e.g. MinCovDet with too few observations
            print(f"  {name}: fit failed ({type(e).__name__}: {e}), scores set to NaN")
            for k in to_score:
                results[k][name] = np.full(len(to_score[k]), np.nan)
            continue
        for k, X in scored.items():
            results[k][name] = -det.decision_function(X)
    return results


# -- Leave-one-class-out benchmark ----------------------------------------------

def _features_and_ids(model, dataset, device, clean_transform=None):
    """Return (sample_ids, per-sample mean features). The ids are needed to
    attach the scores back to the samples."""
    from ood_utils import extract_features_and_logits, make_clean_copy

    if clean_transform is not None:
        dataset = make_clean_copy(dataset, transform=clean_transform)
    records = extract_features_and_logits(model, dataset, device)
    feats = np.stack([r["feature"] for r in records])
    sids = [r["sample_id"] for r in records]
    g = pd.DataFrame(feats).assign(sample_id=sids).groupby("sample_id").mean()
    return list(g.index), g.to_numpy()


def loco_checkpoint_dir(held_out_class: int, seed: int) -> str:
    """Same convention as run_loco_ood.py: the config seed has no suffix."""
    suffix = "" if seed == config.SEED else f"_seed{seed}"
    return os.path.join(config.CHECKPOINT_DIR, f"leave_out_{held_out_class}{suffix}")


def run_for_class(held_out_class: int, device, pca_components: int | None,
                  seed: int = None, out_dir: str = None):
    """Return (metrics, per-sample scores) for one held-out class and seed."""
    from dataset_samples import get_transforms
    from model import load_model
    from run_loco_ood import build_loco_dataloaders, load_full_dataframe

    if seed is None:
        seed = config.SEED
    exp_dir = loco_checkpoint_dir(held_out_class, seed)
    best_model_path = os.path.join(exp_dir, "best_dynamicvit.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"No checkpoint at {best_model_path}: run "
            f"run_loco_ood.py --classes {held_out_class} --seeds {seed} first"
        )

    print(f"\n{'=' * 70}\nHeld-out class {held_out_class}  (seed {seed})\n{'=' * 70}")

    df_full = load_full_dataframe()
    # The split depends on the seed: it must be passed, otherwise a model
    # trained on one split would be evaluated on another.
    (_, _, train_ds, val_ds, test_ds, ood_ds, num_classes_id) = build_loco_dataloaders(
        df_full, held_out_class, seed=seed)

    model = load_model(best_model_path, num_classes=num_classes_id, device=device)
    _, val_transform = get_transforms()

    print("Extracting per-sample features: train, val, ID test, OOD...")
    _, train_X = _features_and_ids(model, train_ds, device, clean_transform=val_transform)
    val_ids, val_X = _features_and_ids(model, val_ds, device, clean_transform=val_transform)
    test_ids, test_X = _features_and_ids(model, test_ds, device)
    ood_ids, ood_X = _features_and_ids(model, ood_ds, device)
    print(f"Samples - train: {len(train_X)}, val: {len(val_X)}, "
          f"ID test: {len(test_X)}, OOD: {len(ood_X)}")

    scores = fit_score_detectors(
        train_X, {"val": val_X, "test": test_X, "ood": ood_X}, pca_components)

    # Threshold-free metrics
    rows = []
    for name in build_detectors():
        y = np.concatenate([np.zeros(len(test_X)), np.ones(len(ood_X))])
        s = np.concatenate([scores["test"][name], scores["ood"][name]])
        if np.isnan(s).any():
            print(f"  {name:20s}  scores not available, skipped")
            continue
        m = dict(auroc=roc_auc_score(y, s), aupr=average_precision_score(y, s),
                 fpr95=fpr_at_tpr(y, s, 0.95))
        rows.append(dict(held_out_class=held_out_class, seed=seed, score=name, **m))

    # Per-sample scores
    parts = []
    for split, ids in (("val", val_ids), ("test", test_ids), ("ood", ood_ids)):
        d = pd.DataFrame({"sample_id": ids})
        d.insert(0, "held_out_class", held_out_class)
        d.insert(1, "seed", seed)
        d.insert(2, "split", split)
        for name in build_detectors():
            d[name] = scores[split][name]
        parts.append(d)
    per_sample = pd.concat(parts, ignore_index=True)

    if out_dir:
        p = os.path.join(out_dir, f"sklearn_scores_class{held_out_class}_seed{seed}.csv")
        per_sample.to_csv(p, index=False)
        print(f"  per-sample scores -> {p}")

    return pd.DataFrame(rows), per_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="one or more seeds, the same passed to run_loco_ood.py "
                             "(default: the config seed only)")
    parser.add_argument("--pca-components", type=int, default=50,
                        help="PCA components before the detectors (0 to disable)")
    args = parser.parse_args()
    pca = args.pca_components or None
    seeds = args.seeds or [config.SEED]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = os.path.join(config.OUTPUT_DIR, "loco_ood")
    os.makedirs(out_dir, exist_ok=True)

    metrics, samples = [], []
    for c in args.classes:
        for s in seeds:
            m, p = run_for_class(c, device, pca, seed=s, out_dir=out_dir)
            metrics.append(m)
            samples.append(p)
    all_results = pd.concat(metrics, ignore_index=True)
    suffix = f"_pca{pca}" if pca else ""
    out_path = os.path.join(out_dir, f"sklearn_baselines{suffix}.csv")
    all_results.to_csv(out_path, index=False)
    pd.concat(samples, ignore_index=True).to_csv(
        os.path.join(out_dir, "sklearn_scores_per_sample.csv"), index=False)
    print(f"\nSaved to {out_path}")
    print(f"Per-sample scores in {os.path.join(out_dir, 'sklearn_scores_per_sample.csv')}")

    if len(seeds) > 1:
        print("\nMean +- std over seeds:")
        agg = (all_results.groupby(["held_out_class", "score"])[["auroc", "aupr", "fpr95"]]
               .agg(["mean", "std"]).round(4))
        print(agg.to_string())

    # Merge with the metrics already computed by run_loco_ood.py, if present
    combined = [all_results]
    for c in args.classes:
        for s in seeds:
            existing_path = os.path.join(loco_checkpoint_dir(c, s), "loco_ood_metrics.csv")
            if os.path.exists(existing_path):
                existing = pd.read_csv(existing_path)
                existing = existing[existing["covariance"] == "shared"]
                columns = [x for x in ["held_out_class", "seed", "score", "auroc", "aupr", "fpr95"]
                           if x in existing.columns]
                combined.append(existing[columns])

    if len(combined) > 1:
        full_table = pd.concat(combined, ignore_index=True).sort_values(
            ["held_out_class", "auroc"], ascending=[True, False])
        full_path = os.path.join(out_dir, f"all_scores_comparison{suffix}.csv")
        full_table.to_csv(full_path, index=False)
        print(f"\nFull comparison table saved to {full_path}")
