"""
LaTeX tables for the three representative detectors, on the control and on
the case study.

The numbers are not recomputed: they are read from the operating_point_*.csv
files written by operating_point_eval.py, so they cannot diverge from those
used in the figures.

Create the three input files first:

    python operating_point_eval.py --setting loco --held-out-class 1  \\
        --seeds 100 101 102 --sweep
    python operating_point_eval.py --setting loco --held-out-class 16 \\
        --seeds 100 101 102 --sweep
    python operating_point_eval.py --setting case-study --sweep \\
        --ground-truth ../data/OOD-DATASET_ground_truth.csv

Then:

    python three_method_tables.py --percentile 85

Outputs in outputs/:
    tab_loco_three_methods.tex         control, three methods, chosen threshold
    tab_loco_sweep.tex                 control, three methods, all percentiles
                                       (supplementary)
    tab_case_study_three_methods.tex   case study, three methods, five outcomes
    tab_case_study_own_threshold.tex   case study, each method at the percentile
                                       selected for it on the control (written
                                       only if the methods disagree)
"""

import argparse
import os

import pandas as pd

import config

ORDER = ["Mahalanobis distance", "MSP", "Isolation Forest"]
FAMILY = {"Mahalanobis distance": "feature distance",
          "MSP": "output uncertainty",
          "Isolation Forest": "density / support"}


def read_table(slug):
    p = os.path.join(config.OUTPUT_DIR, f"operating_point_{slug}.csv")
    if not os.path.exists(p):
        print(f"  missing {p}")
        return None
    d = pd.read_csv(p)
    print(f"  read {os.path.basename(p)}: {d['method'].nunique()} methods, "
          f"{d['percentile'].nunique()} percentiles")
    return d


def ordered_methods(d):
    known = [m for m in ORDER if m in set(d["method"])]
    others = [m for m in d["method"].unique() if m not in ORDER]
    return known + sorted(others)


def fmt(v, dec=1):
    return "--" if v is None or pd.isna(v) else f"{v:.{dec}f}"


def tab_loco(d1, d16, perc, out, classes):
    """Three methods x two held-out fabrics, at the chosen threshold."""
    rows = []
    for m in ordered_methods(d1):
        a = d1[(d1["method"] == m) & (d1["percentile"] == perc)]
        b = d16[(d16["method"] == m) & (d16["percentile"] == perc)]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        rows.append(f"{m} & {FAMILY.get(m, '--')} & "
                    f"{fmt(a['recall_novel'])} & {fmt(a['false_alarms'])} & "
                    f"{fmt(b['recall_novel'])} & {fmt(b['false_alarms'])} \\\\")
    tex = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Leave-one-fabric-out detection at the "
        f"{int(perc)}th percentile of the validation distribution, for one "
        "representative of each family of detectors. Recall is the proportion "
        "of the excluded fabric recognised as new; false alarms are the "
        "proportion of known samples exceeding the threshold. Each sample's "
        "score is converted to its percentile in the validation distribution "
        "of the same seed, and the percentiles are averaged over the three "
        "seeds before the threshold is applied.}",
        "\\label{tab:loco3}",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "\\multirow{2}{*}{Detector} & \\multirow{2}{*}{Family} & "
        f"\\multicolumn{{2}}{{c}}{{Class {classes[0]} excluded}} & "
        f"\\multicolumn{{2}}{{c}}{{Class {classes[1]} excluded}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        " & & Recall (\\%) & False alarms (\\%) & Recall (\\%) & False alarms (\\%) \\\\",
        "\\midrule", *rows, "\\bottomrule",
        "\\end{tabular}", "\\end{table}",
    ]
    write(tex, out, "tab_loco_three_methods.tex")


def tab_sweep(d1, d16, out, classes):
    """All percentiles, three methods, two fabrics (supplementary)."""
    rows = []
    for m in ordered_methods(d1):
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{m}}}}} \\\\")
        for p in sorted(d1["percentile"].unique()):
            a = d1[(d1["method"] == m) & (d1["percentile"] == p)]
            b = d16[(d16["method"] == m) & (d16["percentile"] == p)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            rows.append(f"\\quad {p:g} & {fmt(a['recall_novel'])} & "
                        f"{fmt(a['false_alarms'])} & {fmt(b['recall_novel'])} & "
                        f"{fmt(b['false_alarms'])} \\\\")
        rows.append("\\addlinespace")
    tex = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\caption{Leave-one-fabric-out detection as a function of the "
        "percentile of the validation distribution used as threshold, for the "
        "three detectors and the two excluded fabrics.}",
        "\\label{tab:locosweep}",
        "\\begin{tabular}{lrrrr}", "\\toprule",
        f"\\multirow{{2}}{{*}}{{Percentile}} & \\multicolumn{{2}}{{c}}{{Class {classes[0]} excluded}} & "
        f"\\multicolumn{{2}}{{c}}{{Class {classes[1]} excluded}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        " & Recall (\\%) & False alarms (\\%) & Recall (\\%) & False alarms (\\%) \\\\",
        "\\midrule", *rows[:-1], "\\bottomrule",
        "\\end{tabular}", "\\end{table}",
    ]
    write(tex, out, "tab_loco_sweep.tex")


def tab_case_study(d, perc, out, best=None, name="tab_case_study_three_methods.tex"):
    """The five outcomes on the case study, one method per row.

    best = dict method -> percentile. If given, each detector is evaluated at
    the threshold selected for it on the control, so that no detector is
    evaluated at another's operating point."""
    # Totals are derived from the data, so the caption never carries stale numbers
    q0 = d[d["percentile"] == perc]
    n_loners = n_known = "--"
    if not q0.empty:
        z = q0.iloc[0]
        n_loners = int(z["novel_found"] + z["novel_missed"])
        n_known = int(z["false_alarm"] + z["correct_known"] + z["wrong_class"])

    rows = []
    for m in ordered_methods(d):
        p_m = (best or {}).get(m, perc)
        q = d[(d["method"] == m) & (d["percentile"] == p_m)]
        if q.empty:
            continue
        q = q.iloc[0]
        col_p = f"{p_m:g} & " if best else ""
        rows.append(f"{m} & {FAMILY.get(m, '--')} & " + col_p +
                    f"{int(q['novel_found'])} & {int(q['novel_missed'])} & "
                    f"{int(q['false_alarm'])} & {int(q['correct_known'])} & "
                    f"{int(q['wrong_class'])} & {fmt(q['openset'])} \\\\")
    tex = [
        "\\begin{table}[htbp]", "\\centering", "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{" + (
            "Outcome on the held-out assemblage, each detector at the "
            "percentile selected for it on the control, so that no detector is "
            "evaluated at another's operating point."
            if best else
            f"Outcome on the held-out assemblage at the {int(perc)}th-percentile "
            "threshold, for the three detectors. The threshold is the same "
            "percentile of each detector's own validation distribution, so the "
            "three are compared at the same expected rate of wrongly flagged "
            "known samples."
        ) + " Open-set accuracy is the proportion of specimens receiving the "
        "correct verdict, that is correctly attributed known sherds plus "
        "flagged \\textit{loners}.}",
        "\\label{tab:caso3" + ("b" if best else "") + "}",
        "\\begin{tabular}{ll" + ("r" if best else "") + "rrrrrr}", "\\toprule",
        "\\multirow{2}{*}{Detector} & \\multirow{2}{*}{Family} & "
        + ("\\multirow{2}{*}{Perc.} & " if best else "")
        + f"\\multicolumn{{2}}{{c}}{{\\textit{{Loners}} ({n_loners})}} & "
        + f"\\multicolumn{{3}}{{c}}{{Known fabric ({n_known})}} & "
        + "\\multirow{2}{*}{Open-set (\\%)} \\\\",
        ("\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}" if best
         else "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}"),
        " & & " + ("& " if best else "")
        + "found & missed & flagged & correct & wrong & \\\\",
        "\\midrule", *rows, "\\bottomrule",
        "\\end{tabular}", "\\end{table}",
    ]
    write(tex, out, name)


def select_threshold(d1, d16):
    """Percentile maximising recall minus false alarms on the control.

    Computed for each method separately and as the mean over methods. The
    latter is the threshold to adopt when the three detectors are compared on
    equal terms: choosing it on one method and applying it to all would favour
    that method.
    """
    def score(d, m, p):
        q = d[(d["method"] == m) & (d["percentile"] == p)]
        if q.empty:
            return None
        q = q.iloc[0]
        return q["recall_novel"] - q["false_alarms"]

    def per_method(m, p):
        a, b = score(d1, m, p), score(d16, m, p)
        return None if a is None or b is None else (a + b) / 2

    methods = ordered_methods(d1)
    percentiles = sorted(set(d1["percentile"]) & set(d16["percentile"]))
    print("\n" + "=" * 70)
    print("THRESHOLD SELECTION on the control (recall minus false alarms)")
    print("=" * 70)
    print("percentile" + "".join(f"{m[:18]:>20}" for m in methods) + f"{'mean':>12}")
    means = {}
    for p in percentiles:
        values = [per_method(m, p) for m in methods]
        valid = [x for x in values if x is not None]
        means[p] = sum(valid) / len(valid) if valid else None
        print(f"{p:>10g}"
              + "".join(f"{x:>20.1f}" if x is not None else f"{'--':>20}" for x in values)
              + (f"{means[p]:>12.1f}" if means[p] is not None else f"{'--':>12}"))

    print("\nBest percentile for each method:")
    best = {}
    for m in methods:
        points = {p: per_method(m, p) for p in percentiles}
        points = {k: v for k, v in points.items() if v is not None}
        if points:
            best[m] = max(points, key=points.get)
            print(f"    {m:<24} {best[m]:g}th percentile ({points[best[m]]:.1f} points)")

    valid = {k: v for k, v in means.items() if v is not None}
    if valid:
        top = max(valid, key=valid.get)
        second = sorted(valid.values(), reverse=True)[1] if len(valid) > 1 else None
        print(f"\n    Averaging over the methods: {top:g}th percentile "
              f"({valid[top]:.1f} points"
              + (f", against {second:.1f} for the runner-up)" if second else ")"))
        print("    (note: if one method collapses early, it dominates the mean)")
    return best


def write(tex, out, name):
    p = os.path.join(out, name)
    with open(p, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"  {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=85)
    ap.add_argument("--classes", type=int, nargs=2, default=[1, 16])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or config.OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    print("Input files:")
    d1 = read_table(f"fabric_{args.classes[0]}_excluded")
    d16 = read_table(f"fabric_{args.classes[1]}_excluded")
    dc = read_table("case_study")

    best = None
    if d1 is not None and d16 is not None:
        best = select_threshold(d1, d16)

    print("\nTables written:")
    if d1 is not None and d16 is not None:
        tab_loco(d1, d16, args.percentile, out, args.classes)
        tab_sweep(d1, d16, out, args.classes)
    if dc is not None:
        tab_case_study(dc, args.percentile, out)
        if best and len(set(best.values())) > 1:
            tab_case_study(dc, args.percentile, out, best,
                           "tab_case_study_own_threshold.tex")

    if dc is not None:
        print("\nCase study summary:")
        q = dc[dc["percentile"] == args.percentile]
        cols = ["method", "novel_found", "novel_missed", "false_alarm",
                "correct_known", "wrong_class", "openset"]
        print(q[[c for c in cols if c in q.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
