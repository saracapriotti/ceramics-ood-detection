"""
Threshold-free comparison of the novelty-detection methods across the three
settings.

The three settings differ in the NATURE of the novelty:

  fabric 1 held out    a whole, petrographically distinctive class: it leaves
                       an empty region in feature space;
  fabric 16 held out   a whole class close to the others: the empty region it
                       leaves borders known classes;
  case study           individual sherds of heterogeneous origin that do not
                       form any region.

All metrics here are threshold-free (AUROC, AUPR, FPR@95TPR): they describe
how a method ranks the samples, not how it behaves at the threshold where it
is used. The operating-point comparison is in operating_point_eval.py.

Inputs:
  control      outputs/loco_ood/all_scores_comparison_pca50.csv
               (written by sklearn_ood_baselines.py, which merges its four
               detectors with the five scores of run_loco_ood.py)
  case study   outputs/ood/<cv-name>/ood_scores_per_fold.csv + --ground-truth

Usage:
    python method_comparison.py \\
        --ground-truth ../data/OOD-DATASET_ground_truth.csv

    python method_comparison.py --representatives-only \\
        --ground-truth ../data/OOD-DATASET_ground_truth.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config

NOVEL_LABELS = {"NOVEL", "NEW", "LONER", "NUOVA", "NUOVO"}

# internal name -> (label for the paper, family, column in the case-study CSV;
# msp and odin are stored negated there)
METHODS = {
    "mahalanobis":       ("Mahalanobis distance",  "feature distance",     "mahalanobis"),
    "msp":               ("MSP",                   "output uncertainty",   "msp_neg"),
    "entropy":           ("Predictive entropy",    "output uncertainty",   "entropy"),
    "odin":              ("ODIN",                  "output uncertainty",   "odin_neg"),
    "energy":            ("Free energy",           "output uncertainty",   "energy"),
    "isolation_forest":  ("Isolation Forest",      "density / support",    "isolation_forest"),
    "lof":               ("Local Outlier Factor",  "density / support",    "lof"),
    "elliptic_envelope": ("Elliptic Envelope",     "density / support",    "elliptic_envelope"),
    "ocsvm":             ("One-Class SVM",         "density / support",    "ocsvm"),
}

# One representative per family, chosen on the CONTROL, not on the case study
REPRESENTATIVES = ["mahalanobis", "msp", "isolation_forest"]


def fpr_at_tpr(y, s, target=0.95):
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tpr = np.cumsum(y == 1) / n_pos
    fpr = np.cumsum(y == 0) / n_neg
    i = int(np.searchsorted(tpr, target))
    return float(fpr[-1] if i >= len(fpr) else fpr[i])


def find_column(df, name):
    """The score column, in its raw or standardised version."""
    for c in (name, name + "_raw", name + "_z"):
        if c in df.columns:
            return c
    return None


def case_study(cv_name, gt_path):
    """AUROC/AUPR/FPR@95 for each method, computed within each fold and then
    averaged (same form as the control, where the average is over seeds).
    Positives = loners, negatives = known-fabric samples."""
    d = os.path.join(config.OUTPUT_DIR, "ood", cv_name)
    p = os.path.join(d, "ood_scores_per_fold.csv")
    if not os.path.exists(p):
        raise SystemExit(f"File not found: {p}")
    ood = pd.read_csv(p)

    with open(gt_path) as f:
        sep = ";" if ";" in f.readline() else ","
    gt = pd.read_csv(gt_path, sep=sep)
    gt.columns = [c.strip().lower() for c in gt.columns]
    is_novel = {str(s).strip(): str(t).strip().upper() in NOVEL_LABELS
                for s, t in zip(gt["sample_id"], gt["true_class"])}

    ood["is_novel"] = ood["sample_id"].astype(str).map(is_novel)
    ood = ood.dropna(subset=["is_novel"])
    ood["is_novel"] = ood["is_novel"].astype(int)

    rows = []
    for name, (_, _, case_col) in METHODS.items():
        c = find_column(ood, case_col)
        if c is None:
            continue
        for fold, g in ood.groupby("fold"):
            y, s = g["is_novel"].to_numpy(), g[c].to_numpy()
            if np.isnan(s).any() or y.sum() == 0 or y.sum() == len(y):
                continue
            rows.append(dict(setting="case study", score=name, repetition=fold,
                             auroc=roc_auc_score(y, s),
                             aupr=average_precision_score(y, s),
                             fpr95=fpr_at_tpr(y, s)))
    return pd.DataFrame(rows)


def control():
    p = None
    d = os.path.join(config.OUTPUT_DIR, "loco_ood")
    for name in ("all_scores_comparison_pca50.csv", "all_scores_comparison.csv"):
        if os.path.exists(os.path.join(d, name)):
            p = os.path.join(d, name)
            break
    if p is None:
        raise SystemExit(
            f"Control table not found in {d}/.\n"
            f"Create it with:  python sklearn_ood_baselines.py --classes 1 16 "
            f"--seeds 100 101 102")
    t = pd.read_csv(p)
    t = t.rename(columns={"seed": "repetition"})
    t["setting"] = "fabric " + t["held_out_class"].astype(str) + " held out"
    return t[["setting", "score", "repetition", "auroc", "aupr", "fpr95"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-name", default="whole_halves")
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--representatives-only", action="store_true",
                    help="only the three representatives instead of all nine methods")
    args = ap.parse_args()

    allres = pd.concat([control(), case_study(args.cv_name, args.ground_truth)],
                       ignore_index=True)
    allres = allres[allres["score"].isin(METHODS)]
    if args.representatives_only:
        allres = allres[allres["score"].isin(REPRESENTATIVES)]

    agg = (allres.groupby(["setting", "score"])[["auroc", "aupr", "fpr95"]]
           .agg(["mean", "std", "count"]))

    setting_order = [s for s in ["fabric 1 held out", "fabric 16 held out", "case study"]
                     if s in set(allres["setting"])]
    score_order = [s for s in METHODS if s in set(allres["score"])]

    print("=" * 78)
    print("METHOD COMPARISON - mean +- std over repetitions")
    print("=" * 78)
    print("The control has 3 repetitions (seeds), the case study 5 (folds).\n")

    for setting in setting_order:
        print(f"--- {setting} " + "-" * (74 - len(setting)))
        print(f"{'method':<22}{'family':<20}{'AUROC':>16}{'AUPR':>16}{'FPR@95':>14}")
        rows = [(s, agg.loc[(setting, s)]) for s in score_order if (setting, s) in agg.index]
        rows.sort(key=lambda kv: -kv[1][("auroc", "mean")])
        for s, r in rows:
            label, family, _ = METHODS[s]

            def fmt(m):
                sd = r[(m, "std")]
                return f"{r[(m, 'mean')]:.3f}+-{0 if np.isnan(sd) else sd:.3f}"

            print(f"{label:<22}{family:<20}{fmt('auroc'):>16}{fmt('aupr'):>16}"
                  f"{fmt('fpr95'):>14}")
        print()

    # Wide table: one row per method, one column per setting
    wide = (allres.pivot_table(index="score", columns="setting", values="auroc",
                               aggfunc="mean")
            .reindex(score_order)[setting_order].round(3))
    wide.index = [METHODS[s][0] for s in wide.index]
    print("AUROC, one row per method:")
    print(wide.to_string())

    out = os.path.join(config.OUTPUT_DIR, "method_comparison.csv")
    agg.round(4).to_csv(out)
    print(f"\n{out}")

    print("\nLaTeX rows (mean AUROC per setting):")
    for s in score_order:
        label, family, _ = METHODS[s]
        values = []
        for setting in setting_order:
            m = agg.loc[(setting, s), ("auroc", "mean")] if (setting, s) in agg.index else np.nan
            sd = agg.loc[(setting, s), ("auroc", "std")] if (setting, s) in agg.index else np.nan
            values.append("--" if np.isnan(m) else
                          f"${m:.3f} \\pm {0 if np.isnan(sd) else sd:.3f}$")
        print(f"{label} & {family} & " + " & ".join(values) + r" \\")

    print("\nNote: the four sklearn detectors operate on features reduced to 50 "
          "principal components,\nthe other five do not.")


if __name__ == "__main__":
    main()
