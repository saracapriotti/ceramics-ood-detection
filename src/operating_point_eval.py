"""
Comparison of novelty-detection methods AT THE OPERATING POINT.

AUROC and AUPR describe how a method ranks samples over all possible
thresholds, but a method is used at a single threshold, where two methods
with the same AUROC can behave very differently. For each method and a given
threshold, this script reports how many new samples are found, how many
known samples are wrongly flagged, and the five open-set outcomes.

DECISION RULE. Within each run (fold or seed), every sample is placed in the
VALIDATION distribution of that run and its percentile is taken; percentiles
are averaged across runs, and the sample is flagged as new if the mean
percentile exceeds the chosen threshold. The rule is dimensionless, so it
applies to any score (including energy or negated MSP, which can be negative)
and puts methods with different scales on the same footing.

The five mutually exclusive outcomes, one per sample:
    novel_found     loner, flagged
    novel_missed    loner, not flagged
    false_alarm     known fabric, flagged
    correct_known   known fabric, not flagged, correct fabric
    wrong_class     known fabric, not flagged, wrong fabric
    open-set accuracy = (correct_known + novel_found) / total

Usage:
    python operating_point_eval.py --setting case-study --percentile 85 \\
        --ground-truth ../data/OOD-DATASET_ground_truth.csv

    python operating_point_eval.py --setting case-study --sweep \\
        --ground-truth ../data/OOD-DATASET_ground_truth.csv

    python operating_point_eval.py --setting loco --held-out-class 1 --percentile 85
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

BLUE, ORANGE, GREEN = "#1B4F72", "#E69F00", "#009E73"
INK, INK_SOFT, GRID = "#1a1a1a", "#5c5c5c", "#ececea"
METHOD_COLOUR = {"Mahalanobis distance": BLUE, "MSP": ORANGE,
                 "Isolation Forest": GREEN}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.labelpad": 10,
    "axes.edgecolor": INK_SOFT, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# Accepted labels for novel samples in the ground-truth CSV
NOVEL_LABELS = {"NOVEL", "NEW", "LONER", "NUOVA", "NUOVO"}

# internal name -> (column in the CSV files, label for the paper, family)
METHODS = {
    # feature distance
    "mahalanobis":      ("mahalanobis",       "Mahalanobis distance", "feature distance"),
    # output uncertainty
    "msp":              ("msp_neg",           "MSP",                  "output uncertainty"),
    "entropy":          ("entropy",           "Predictive entropy",   "output uncertainty"),
    "energy":           ("energy",            "Free energy",          "output uncertainty"),
    "odin":             ("odin_neg",          "ODIN (T=1000)",        "output uncertainty"),
    # density / support
    "isolation_forest": ("isolation_forest",  "Isolation Forest",     "density / support"),
    "lof":              ("lof",               "Local Outlier Factor", "density / support"),
    "elliptic":         ("elliptic_envelope", "Elliptic Envelope",    "density / support"),
    "ocsvm":            ("ocsvm",             "One-Class SVM",        "density / support"),
}

# The three methods reported in the paper, one per family (default everywhere)
REPRESENTATIVES = ["mahalanobis", "msp", "isolation_forest"]
OUTCOMES = ["correct_known", "wrong_class", "false_alarm",
            "novel_found", "novel_missed"]


def _clean_axes(ax, y_grid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color=INK_SOFT)


def plot_dotplot(details, percentile, setting_name, out_dir):
    """One panel per method, with the same samples on the same rows.

    Row order is set by the first method, so its panel is monotonic and the
    others show how far they deviate: reading across a row shows which
    methods flag a given sample."""
    methods = list(details)
    if not methods:
        return
    base = details[methods[0]].sort_values("mean_percentile").reset_index(drop=True)
    order = list(base["sample_id"])
    y = np.arange(len(order))

    fig, axes = plt.subplots(
        1, len(methods), figsize=(3.6 * len(methods) + 1.4, max(4.0, 0.115 * len(order))),
        sharey=True, gridspec_kw=dict(wspace=0.12))
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, methods):
        d = details[m].set_index("sample_id").reindex(order)
        x = d["mean_percentile"].to_numpy()
        novel = d["is_novel"].to_numpy(dtype=bool)
        wrong = (d["is_novel"] != d["flagged"]).to_numpy(dtype=bool)
        ax.scatter(x[~novel], y[~novel], s=26, color=BLUE, edgecolor="white",
                   linewidth=0.4, label="known fabric", zorder=3)
        ax.scatter(x[novel], y[novel], s=26, color=ORANGE, edgecolor="white",
                   linewidth=0.4, label="loner", zorder=3)
        ax.scatter(x[wrong], y[wrong], marker="x", s=26, color="#000000",
                   linewidth=1.0, zorder=4, label="verdict disagrees")
        ax.axvline(percentile, color=INK, linestyle="--", linewidth=1.0, zorder=2)
        ax.set_xlim(-2, 102)
        ax.set_title(m, fontsize=9, color=INK, pad=8)
        ax.set_xlabel("Novelty percentile")
        _clean_axes(ax)

    axes[0].set_yticks(y, order, fontsize=5.5)
    axes[0].set_ylim(len(order), -1.5)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", labelspacing=0.3,
                   handletextpad=0.5, borderaxespad=0.8)

    p = os.path.join(out_dir, f"fig_dotplot_methods_{setting_name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def plot_curves(table, setting_name, out_dir):
    """Recall of new samples and false alarms as a function of the threshold,
    one colour per method."""
    if table.empty or table["percentile"].nunique() < 3:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                               gridspec_kw=dict(wspace=0.22))
    for m, g in table.groupby("method"):
        g = g.sort_values("percentile")
        c = METHOD_COLOUR.get(m, INK_SOFT)
        a.plot(g["percentile"], g["recall_novel"], color=c, linewidth=2, label=m)
        a.scatter(g["percentile"], g["recall_novel"], color=c, s=18, zorder=3)
        b.plot(g["percentile"], g["false_alarms"], color=c, linewidth=2, label=m)
        b.scatter(g["percentile"], g["false_alarms"], color=c, s=18, zorder=3)
    for ax, ylabel in ((a, "New samples recovered (%)"), (b, "False alarms (%)")):
        ax.set_xlabel("Percentile of the validation distribution used as threshold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-3, 103)
        _clean_axes(ax)
    a.legend(frameon=False, fontsize=8, loc="lower left", labelspacing=0.3)

    p = os.path.join(out_dir, f"fig_curves_methods_{setting_name}.png")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def percentile_in_distribution(values, reference):
    """Position of each value within the reference distribution, in percent.
    90 means 'higher than 90% of the validation samples'."""
    ref = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ref, np.asarray(values, dtype=float), side="right") / len(ref) * 100


def read_ground_truth(path):
    with open(path) as f:
        sep = ";" if ";" in f.readline() else ","
    g = pd.read_csv(path, sep=sep)
    g.columns = [c.strip().lower() for c in g.columns]
    is_novel, fabric = {}, {}
    for sid, tc in zip(g["sample_id"], g["true_class"]):
        sid, tc = str(sid).strip(), str(tc).strip()
        is_novel[sid] = tc.upper() in NOVEL_LABELS
        if not is_novel[sid]:
            try:
                fabric[sid] = int(float(tc))
            except ValueError:
                pass
    return is_novel, fabric


def load_case_study(cv_name, gt_path):
    d = os.path.join(config.OUTPUT_DIR, "ood", cv_name)
    ood = pd.read_csv(os.path.join(d, "ood_scores_per_fold.csv"))
    val = pd.read_csv(os.path.join(d, "val_scores_per_fold.csv"))
    is_novel, fabric = read_ground_truth(gt_path)
    return ood, val, is_novel, fabric, "case study"


def load_sklearn_loco(held_out_class):
    """sklearn detector scores for the leave-one-fabric-out experiment, if
    available (computed by sklearn_ood_baselines.py on the same checkpoints)."""
    p = os.path.join(config.OUTPUT_DIR, "loco_ood", "sklearn_scores_per_sample.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d["held_out_class"] == held_out_class].copy()
    d["sample_id"] = d["sample_id"].astype(str)
    return d if len(d) else None


def merge_sklearn(table, sk, seed, split):
    """Attach the sklearn columns to a run_loco_ood table, for the same seed and split."""
    if sk is None:
        return table
    q = sk[(sk["seed"] == seed) & (sk["split"] == split)]
    if q.empty:
        return table
    columns = [c for c in ("elliptic_envelope", "isolation_forest", "lof", "ocsvm")
               if c in q.columns]
    table = table.copy()
    table["sample_id"] = table["sample_id"].astype(str)
    return table.merge(q[["sample_id"] + columns], on="sample_id", how="left")


def load_loco(held_out_class, seeds, cov="shared"):
    """Read the leave-one-fabric-out outputs. Here each 'fold' is a seed."""
    base = config.CHECKPOINT_DIR
    ood_parts, val_parts, is_novel, fabric = [], [], {}, {}
    found = []
    sk = load_sklearn_loco(held_out_class)
    if sk is None:
        print("  (no sklearn_scores_per_sample.csv: Isolation Forest will not be "
              "available in this setting)")
    for s in (seeds or [config.SEED]):
        suffix = "" if s == config.SEED else f"_seed{s}"
        d = os.path.join(base, f"leave_out_{held_out_class}{suffix}")
        p_val = os.path.join(d, f"val_scores_{cov}.csv")
        p_ood = os.path.join(d, f"ood_scores_{cov}.csv")
        p_id = os.path.join(d, f"id_scores_{cov}.csv")
        if not all(os.path.exists(p) for p in (p_val, p_ood, p_id)):
            print(f"  skipping seed {s}: missing files in {d}/")
            continue
        val = pd.read_csv(p_val)
        if "msp_neg" not in val.columns and "msp" not in val.columns:
            raise SystemExit(
                f"{p_val} contains only the Mahalanobis score. Re-run run_loco_ood.py "
                f"so that all scores are computed on the validation set.")
        ood = pd.read_csv(p_ood)
        idd = pd.read_csv(p_id)
        val = merge_sklearn(val, sk, s, "val")
        idd = merge_sklearn(idd, sk, s, "test")
        ood = merge_sklearn(ood, sk, s, "ood")
        for t in (val, ood, idd):
            t["fold"] = s
        val_parts.append(val)
        ood_parts.append(pd.concat([idd, ood], ignore_index=True))
        for sid in ood["sample_id"]:
            is_novel[str(sid)] = True
        for sid in idd["sample_id"]:
            is_novel.setdefault(str(sid), False)
        found.append(s)
    if not ood_parts:
        raise SystemExit(f"No usable seed for class {held_out_class}.")
    print(f"  seeds used: {', '.join(map(str, found))}")
    return (pd.concat(ood_parts, ignore_index=True),
            pd.concat(val_parts, ignore_index=True),
            is_novel, fabric, f"fabric {held_out_class} excluded")


def find_column(df, name):
    for c in (name, name + "_raw", name.replace("_neg", ""), name + "_z"):
        if c in df.columns:
            return c
    return None


def evaluate_method(ood, val, is_novel, fabric, col, percentile):
    """Open-set outcomes at one threshold for one method.
    Returns (counts, per-sample details)."""
    folds = sorted(set(ood["fold"]) & set(val["fold"]))
    pct_per_sample = {}
    for k in folds:
        ref = val.loc[val["fold"] == k, col].to_numpy()
        g = ood[ood["fold"] == k]
        p = percentile_in_distribution(g[col].to_numpy(), ref)
        for sid, v in zip(g["sample_id"].astype(str), p):
            pct_per_sample.setdefault(sid, []).append(float(v))

    # predicted class: most frequent across folds
    pred_class = {}
    if "pred_class_idx" in ood.columns:
        for sid, g in ood.groupby(ood["sample_id"].astype(str)):
            pred_class[sid] = int(g["pred_class_idx"].mode().iloc[0]) + 1

    rows = []
    for sid, values in pct_per_sample.items():
        if sid not in is_novel:
            continue
        mean_pct = float(np.mean(values))
        flagged = mean_pct > percentile
        if is_novel[sid]:
            outcome = "novel_found" if flagged else "novel_missed"
        elif flagged:
            outcome = "false_alarm"
        elif not fabric:
            outcome = "correct_known"  # LOCO: known ID samples have no fabric ground truth here
        elif pred_class.get(sid) == fabric.get(sid):
            outcome = "correct_known"
        else:
            outcome = "wrong_class"
        rows.append(dict(sample_id=sid, mean_percentile=mean_pct,
                         is_novel=is_novel[sid], flagged=flagged, outcome=outcome))

    d = pd.DataFrame(rows)
    c = {o: int((d["outcome"] == o).sum()) for o in OUTCOMES}
    n = len(d)
    n_novel = c["novel_found"] + c["novel_missed"]
    n_known = c["correct_known"] + c["wrong_class"] + c["false_alarm"]
    c.update(
        n=n,
        recall_novel=100 * c["novel_found"] / max(n_novel, 1),
        false_alarms=100 * c["false_alarm"] / max(n_known, 1),
        openset=100 * (c["correct_known"] + c["novel_found"]) / max(n, 1),
    )
    return c, d


def analyze(ood, val, is_novel, fabric, percentiles, chosen_percentile, methods=None):
    """Evaluate the methods at several thresholds. Returns the table of
    counts and, for the chosen threshold, the per-sample details of each
    method. Shared with the figure scripts, so that the decision rule is
    implemented in one place only."""
    methods = methods or REPRESENTATIVES
    rows, details = [], {}
    for perc in percentiles:
        for m in methods:
            if m not in METHODS:
                print(f"  unknown method: {m}")
                continue
            col_name, label, _ = METHODS[m]
            co, cv = find_column(ood, col_name), find_column(val, col_name)
            if co is None or cv is None:
                if perc == percentiles[0]:
                    print(f"{label:<24}  column '{col_name}' missing, skipped")
                continue
            c, det = evaluate_method(ood, val, is_novel, fabric, co, perc)
            rows.append(dict(percentile=perc, method=label, **c))
            if perc == chosen_percentile:
                details[label] = det
    return pd.DataFrame(rows), details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["case-study", "loco"], required=True)
    ap.add_argument("--cv-name", default="whole_halves")
    ap.add_argument("--ground-truth", default=None)
    ap.add_argument("--held-out-class", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--percentile", type=float, default=85)
    ap.add_argument("--sweep", action="store_true",
                    help="evaluate several percentiles instead of a single one")
    ap.add_argument("--methods", nargs="+", default=REPRESENTATIVES,
                    help="one or more of " + ", ".join(METHODS) + "; or 'all' for all nine")
    ap.add_argument("--figures", action="store_true",
                    help="also save the three-panel dot plot and the curves")
    args = ap.parse_args()
    if len(args.methods) == 1 and args.methods[0] == "all":
        args.methods = list(METHODS)

    if args.setting == "case-study":
        if not args.ground_truth:
            raise SystemExit("The case study requires --ground-truth")
        ood, val, is_novel, fabric, setting_name = load_case_study(args.cv_name, args.ground_truth)
    else:
        if args.held_out_class is None:
            raise SystemExit("The leave-one-fabric-out setting requires --held-out-class")
        ood, val, is_novel, fabric, setting_name = load_loco(args.held_out_class, args.seeds)

    percentiles = [70, 75, 80, 85, 90, 95, 99] if args.sweep else [args.percentile]

    print("=" * 80)
    print(f"OPERATING-POINT EVALUATION - {setting_name}")
    print("=" * 80)
    n_novel = sum(1 for s in is_novel.values() if s)
    print(f"{len(is_novel)} samples, {n_novel} of them new. "
          f"Repetitions (folds or seeds): {ood['fold'].nunique()}.")
    print("Threshold = percentile of the validation distribution of the same method.\n")

    table, chosen_details = analyze(
        ood, val, is_novel, fabric, percentiles, args.percentile, args.methods)

    table = table.assign(setting=setting_name)
    slug = setting_name.replace(" ", "_")
    out = os.path.join(config.OUTPUT_DIR, f"operating_point_{slug}.csv")
    table.to_csv(out, index=False)
    print(out)

    # Per-sample details at the chosen threshold: the list of flagged samples,
    # with the verdict of every method side by side.
    if chosen_details:
        merged = None
        for label, d in chosen_details.items():
            key = label.split()[0].lower()
            q = d[["sample_id", "is_novel", "mean_percentile", "flagged"]].rename(
                columns={"mean_percentile": f"pct_{key}", "flagged": f"flag_{key}"})
            merged = q if merged is None else merged.merge(
                q.drop(columns="is_novel"), on="sample_id", how="outer")
        flag_cols = [c for c in merged.columns if c.startswith("flag_")]
        merged["n_methods_flagging"] = merged[flag_cols].sum(axis=1)
        merged = merged.sort_values("n_methods_flagging", ascending=False)
        p = os.path.join(config.OUTPUT_DIR, f"detail_{slug}_p{int(args.percentile)}.csv")
        merged.to_csv(p, index=False)
        print(p)

        print(f"\nAgreement between methods at the {args.percentile:g}th percentile:")
        for n in sorted(merged["n_methods_flagging"].unique(), reverse=True):
            q = merged[merged["n_methods_flagging"] == n]
            print(f"  flagged by {int(n)} of {len(flag_cols)} methods: "
                  f"{len(q)} samples ({int(q['is_novel'].sum())} new, "
                  f"{int((~q['is_novel'].astype(bool)).sum())} of known fabric)")

    if args.figures:
        fig_dir = os.path.join(config.OUTPUT_DIR, "figures", args.cv_name)
        os.makedirs(fig_dir, exist_ok=True)
        print("\nFigures:")
        if chosen_details:
            plot_dotplot(chosen_details, args.percentile, slug, fig_dir)
        else:
            print(f"  (no dot plot: percentile {args.percentile:g} is not among the "
                  f"evaluated ones; add it or remove --sweep)")
        plot_curves(table, slug, fig_dir)


if __name__ == "__main__":
    main()
