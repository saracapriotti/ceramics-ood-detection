"""
Consensus grouping on a user-chosen set of samples.

run_ood_cv.py builds a consensus dendrogram on the samples IT flags
(majority vote over folds, threshold at config.OOD_PERCENTILE). The paper
instead uses the 85th percentile with the mean-percentile rule of
operating_point_eval.py, and the two sets do not coincide. This script
repeats the same computation on any list of samples, reading the feature
cache written by run_ood_cv.py: no model to reload, a few seconds.

Method: feature spaces of different folds are not comparable, so features
cannot be averaged. Samples are clustered within each fold (average linkage,
cosine distance), the number of times two samples end up together is
counted (co-association matrix), and a second clustering is run on that
matrix. The result does not depend on a single model.

Usage:
    # from the per-sample details saved by operating_point_eval.py
    python consensus.py --from-detail ../outputs/detail_case_study_p85.csv \\
        --method mahalanobis

    # only samples flagged by at least two methods out of three
    python consensus.py --from-detail ../outputs/detail_case_study_p85.csv \\
        --min-methods 2

    # or a manual list
    python consensus.py --samples HS-CAMP28 HS-CAMP36 HS45 X.677
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

import config

BLUE, INK, INK_SOFT = "#1B4F72", "#1a1a1a", "#5c5c5c"


def load_features(cv_name):
    """Per-fold OOD features from the run_ood_cv.py cache."""
    d = os.path.join(config.REPO_DIR, "outputs", "ood", cv_name, "features")
    if not os.path.isdir(d):
        raise SystemExit(f"Folder not found: {d}/ (run run_ood_cv.py first)")
    folds = []
    for f in sorted(os.listdir(d)):
        if f.startswith("features_fold") and f.endswith(".npz"):
            z = np.load(os.path.join(d, f), allow_pickle=True)
            folds.append(({str(s): i for i, s in enumerate(z["ood_ids"])}, z["ood"]))
    if not folds:
        raise SystemExit(f"No features_fold*.npz in {d}/")
    return folds


def consensus(folds, ids, fold_threshold):
    """Co-association matrix over the folds that contain all requested samples."""
    n = len(ids)
    co = np.zeros((n, n))
    used = 0
    for pos, X in folds:
        if any(s not in pos for s in ids):
            continue
        sub = np.stack([X[pos[s]] for s in ids])
        Z = linkage(sub, method="average", metric="cosine")
        labels = fcluster(Z, t=fold_threshold, criterion="distance")
        co += (labels[:, None] == labels[None, :]).astype(float)
        used += 1
    if used == 0:
        raise SystemExit("No fold contains all the requested samples.")
    co /= used
    D = 1.0 - co
    np.fill_diagonal(D, 0.0)
    return co, D, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-name", default="whole_halves")
    ap.add_argument("--from-detail", default=None,
                    help="CSV saved by operating_point_eval.py")
    ap.add_argument("--min-methods", type=int, default=1,
                    help="with --from-detail: how many methods must flag a sample "
                         "for it to be included")
    ap.add_argument("--method", default=None,
                    help="with --from-detail: keep only samples flagged by this method "
                         "(e.g. mahalanobis, msp, isolation). Overrides --min-methods.")
    ap.add_argument("--samples", nargs="+", default=None)
    ap.add_argument("--fold-threshold", type=float,
                    default=config.OOD_CLUSTER_DISTANCE_THRESHOLD,
                    help="cut within each fold, cosine distance")
    ap.add_argument("--consensus-cut", type=float, default=0.5,
                    help="0.5 = together in the majority of folds")
    ap.add_argument("--suffix", default="p85")
    args = ap.parse_args()

    if args.samples:
        ids = sorted(args.samples)
        origin = "manual list"
    elif args.from_detail:
        d = pd.read_csv(args.from_detail)
        flag_cols = [c for c in d.columns if c.startswith("flag_")]
        if not flag_cols:
            raise SystemExit(f"No flag_* columns in {args.from_detail}")
        if args.method:
            c = f"flag_{args.method.lower()}"
            if c not in flag_cols:
                raise SystemExit(f"Method '{args.method}' not found. "
                                 f"Available: {', '.join(x[5:] for x in flag_cols)}")
            ids = sorted(d.loc[d[c].astype(bool), "sample_id"].astype(str))
            origin = f"{os.path.basename(args.from_detail)}, flagged by {args.method}"
        else:
            n_flags = d[flag_cols].sum(axis=1)
            ids = sorted(d.loc[n_flags >= args.min_methods, "sample_id"].astype(str))
            origin = (f"{os.path.basename(args.from_detail)}, flagged by at least "
                      f"{args.min_methods} of {len(flag_cols)} methods")
    else:
        raise SystemExit("Either --from-detail or --samples is required")

    if len(ids) < 3:
        raise SystemExit(f"Only {len(ids)} samples: clustering is not meaningful.")

    folds = load_features(args.cv_name)
    co, D, used = consensus(folds, ids, args.fold_threshold)

    print("=" * 74)
    print(f"CONSENSUS GROUPING - {len(ids)} samples, {used} folds")
    print("=" * 74)
    print(f"Sample list: {origin}")
    print(f"Cut within each fold: cosine distance {args.fold_threshold}\n")

    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=args.consensus_cut, criterion="distance")
    groups = {}
    for s, lab in zip(ids, labels):
        groups.setdefault(int(lab), []).append(s)

    isolated = [g[0] for g in groups.values() if len(g) == 1]
    together = {k: g for k, g in groups.items() if len(g) > 1}
    print(f"Isolated samples: {len(isolated)}")
    if together:
        print(f"Groups with more than one sample: {len(together)}")
        for g in together.values():
            print(f"  {', '.join(g)}")
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    a, b = ids.index(g[i]), ids.index(g[j])
                    print(f"      {g[i]} and {g[j]}: together in "
                          f"{100 * co[a, b]:.0f}% of folds")
    else:
        print("No groups: every sample remains isolated.")

    out = os.path.join(config.REPO_DIR, "outputs", "figures", args.cv_name)
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, max(3.2, 0.26 * len(ids))))
    dn = dendrogram(Z, labels=ids, orientation="left", no_plot=True,
                    color_threshold=args.consensus_cut,
                    above_threshold_color="__base__")
    for xs, ys, c in zip(dn["dcoord"], dn["icoord"], dn["color_list"]):
        below_cut = c != "__base__"
        ax.plot(xs, ys, color=BLUE if below_cut else "#8f8f8f",
                linewidth=2.0 if below_cut else 0.8,
                solid_capstyle="round", solid_joinstyle="round")
    ax.set_yticks([5 + 10 * i for i in range(len(dn["ivl"]))], dn["ivl"], fontsize=8)
    ax.set_ylim(0, 10 * len(dn["ivl"]))
    ax.axvline(args.consensus_cut, color=INK, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Consensus distance")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, color=INK_SOFT)
    ax.tick_params(axis="y", length=0, pad=6)
    p = os.path.join(out, f"fig_consensus_{args.suffix}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    co_path = os.path.join(config.REPO_DIR, "outputs", f"coassociation_{args.suffix}.csv")
    pd.DataFrame(co, index=ids, columns=ids).to_csv(co_path)
    print(f"\n{p}")
    print(co_path)


if __name__ == "__main__":
    main()
