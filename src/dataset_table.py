"""
Samples and micrographs per fabric: printed table, CSV and LaTeX.

Two modes:

  single      one dataset at a time (training, or OOD with --ood)
  --combined  training and OOD side by side, one row per fabric plus a row
              for the loners (Table 1 of the paper): shows both the class
              imbalance and which fabrics are represented in the test material.

Outputs:
    dataset_table.csv
    dataset_table.tex    (requires booktabs; --combined also requires multirow)

Usage:
    python dataset_table.py
    python dataset_table.py --ood
    python dataset_table.py --combined \\
        --ground-truth ../data/OOD-DATASET_ground_truth.csv
"""

import argparse
import os

import pandas as pd

import config

NOVEL_LABELS = {"NOVEL", "NEW", "LONER", "NUOVA", "NUOVO"}


def read_dataset(path, extract_sample_id, extract_polarization):
    df = pd.read_csv(path, delimiter=";", encoding="latin-1")
    df.columns = [c.replace("\u00ef\u00bb\u00bf", "").strip() for c in df.columns]  # strip UTF-8 BOM
    col_img = df.columns[0]
    col_lab = df.columns[1] if len(df.columns) > 1 else None
    df = df.dropna(subset=[col_img]).copy()
    df["sample_id"] = df[col_img].apply(extract_sample_id)
    df["pol"] = df[col_img].apply(extract_polarization)
    return df, col_img, col_lab


def count(df, col_img, col_lab):
    df = df.dropna(subset=[col_lab]).copy()
    df[col_lab] = df[col_lab].astype(int)
    t = (df.groupby(col_lab)
           .agg(samples=("sample_id", "nunique"), images=(col_img, "count"))
           .reset_index().rename(columns={col_lab: "fabric"}))
    pol = (df.pivot_table(index=col_lab, columns="pol", values=col_img,
                          aggfunc="count", fill_value=0)
             .reset_index().rename(columns={col_lab: "fabric"}))
    t = t.merge(pol, on="fabric", how="left")
    # The merge introduces floats: cast back so LaTeX shows "19" and not "19.0"
    for c in t.columns:
        t[c] = t[c].fillna(0).astype(int)
    t["img_per_sample"] = (t["images"] / t["samples"]).round(1)
    return t


def write_tex(data_rows, header, total, path, caption, label, alignment=None):
    n = len(header)
    align = alignment or ("l" + "r" * (n - 1))
    out = ["\\begin{table}[htbp]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    out += header + ["\\midrule"]
    out += [" & ".join(r) + r" \\" for r in data_rows]
    out += ["\\midrule", " & ".join(total) + r" \\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


# -- Single table ---------------------------------------------------------------

def single_table(args, extract_sample_id, extract_polarization, out_dir):
    path = args.csv or (config.OOD_CSV_FILE if args.ood else config.CSV_FILE)
    df, col_img, col_lab = read_dataset(path, extract_sample_id, extract_polarization)
    t = count(df, col_img, col_lab)

    tot = {"fabric": "Total", "samples": int(t["samples"].sum()),
           "images": int(t["images"].sum())}
    for c in ("PPL", "XPL"):
        if c in t.columns:
            tot[c] = int(t[c].sum())
    tot["img_per_sample"] = round(tot["images"] / tot["samples"], 1)

    print(path)
    print(f"Smallest fabric: {t['samples'].min()} samples "
          f"(fabric {t.loc[t['samples'].idxmin(), 'fabric']})")
    print(f"Largest fabric:  {t['samples'].max()} samples "
          f"(fabric {t.loc[t['samples'].idxmax(), 'fabric']})")

    columns = ["fabric", "samples", "images"]
    header = ["Fabric", "Samples", "Micrographs"]
    for c in ("PPL", "XPL"):
        if c in t.columns:
            columns.append(c)
            header.append(c)
    columns += ["img_per_sample"]
    header += ["Micrographs per sample"]

    def fmt(v, col):
        return f"{float(v):.1f}" if col == "img_per_sample" else str(v)

    pd.concat([t, pd.DataFrame([tot])], ignore_index=True).to_csv(
        os.path.join(out_dir, "dataset_table.csv"), index=False)
    write_tex([[fmt(r[c], c) for c in columns] for _, r in t.iterrows()],
              [" & ".join(header) + r" \\"],
              [fmt(tot[c], c) for c in columns],
              os.path.join(out_dir, "dataset_table.tex"), args.caption, args.label)


# -- Combined table -------------------------------------------------------------

def combined_table(args, extract_sample_id, extract_polarization, out_dir):
    if not args.ground_truth:
        raise SystemExit("--combined requires --ground-truth: it gives the fabric "
                         "of each OOD sample.")

    tr, tr_img, tr_lab = read_dataset(args.csv or config.CSV_FILE,
                                      extract_sample_id, extract_polarization)
    t_tr = count(tr, tr_img, tr_lab)

    oo, oo_img, _ = read_dataset(args.ood_csv or config.OOD_CSV_FILE,
                                 extract_sample_id, extract_polarization)
    with open(args.ground_truth) as f:
        first_line = f.readline()
    gt = pd.read_csv(args.ground_truth, sep=";" if ";" in first_line else ",")
    gt.columns = [c.strip().lower() for c in gt.columns]
    gt["sample_id"] = gt["sample_id"].astype(str).str.strip()
    gt["true_class"] = gt["true_class"].astype(str).str.strip()

    oo["sample_id"] = oo["sample_id"].astype(str).str.strip()
    oo = oo.merge(gt[["sample_id", "true_class"]], on="sample_id", how="left")
    missing = oo[oo["true_class"].isna()]["sample_id"].nunique()
    if missing:
        print(f"WARNING: {missing} OOD samples have no ground-truth entry and are "
              f"left out of the table.")
    oo = oo.dropna(subset=["true_class"]).copy()
    oo["is_loner"] = oo["true_class"].str.upper().isin(NOVEL_LABELS)

    known = oo[~oo["is_loner"]].copy()
    known["fabric"] = known["true_class"].astype(float).astype(int)
    t_oo = (known.groupby("fabric")
                 .agg(samples=("sample_id", "nunique"), images=(oo_img, "count"))
                 .reset_index())
    loners = oo[oo["is_loner"]]
    n_loner_samples = loners["sample_id"].nunique()
    n_loner_images = len(loners)

    t = t_tr[["fabric", "samples", "images"]].merge(
        t_oo, on="fabric", how="outer", suffixes=("_g1", "_g2")).fillna(0)
    for c in t.columns:
        t[c] = t[c].astype(int)
    t = t.sort_values("fabric").reset_index(drop=True)

    represented = int((t["samples_g2"] > 0).sum())
    print(f"Fabrics represented in the test material: {represented} of {len(t)}")

    # "--" instead of 0: an empty cell reads as an absence, a zero as a value
    def cell(v):
        return "--" if int(v) == 0 else str(int(v))

    rows = [[str(int(r["fabric"])), cell(r["samples_g1"]), cell(r["images_g1"]),
             cell(r["samples_g2"]), cell(r["images_g2"])] for _, r in t.iterrows()]
    rows.append(["Loners", "--", "--", str(n_loner_samples), str(n_loner_images)])
    total = ["Total", str(int(t["samples_g1"].sum())), str(int(t["images_g1"].sum())),
             str(int(t["samples_g2"].sum()) + n_loner_samples),
             str(int(t["images_g2"].sum()) + n_loner_images)]

    header = [r"\multirow{2}{*}{Fabric} & \multicolumn{2}{c}{Group 1 (training)} "
              r"& \multicolumn{2}{c}{Group 2 (held-out)} \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
              r" & Samples & Micrographs & Samples & Micrographs \\"]

    pd.concat([t, pd.DataFrame([{"fabric": "Loners", "samples_g2": n_loner_samples,
                                 "images_g2": n_loner_images}])],
              ignore_index=True).to_csv(
        os.path.join(out_dir, "dataset_table.csv"), index=False)
    write_tex(rows, header, total,
              os.path.join(out_dir, "dataset_table.tex"),
              args.caption, args.label, alignment="lrrrr")
    print("\n(the .tex file uses booktabs and multirow: both are needed in the preamble)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="default: the training CSV from config")
    p.add_argument("--ood", action="store_true", help="use config.OOD_CSV_FILE")
    p.add_argument("--combined", action="store_true",
                   help="training and OOD in the same table")
    p.add_argument("--ood-csv", default=None)
    p.add_argument("--ground-truth", default=None)
    p.add_argument("--out-dir", default=None, help="default: outputs/")
    p.add_argument("--caption", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if args.caption is None:
        args.caption = ("Composition of the two dataset groups: number of samples "
                        "and micrographs per fabric." if args.combined else
                        "Composition of the dataset: number of samples and "
                        "micrographs per fabric.")
    if args.label is None:
        args.label = "tab:dataset"

    from dataset_samples import extract_polarization, extract_sample_id

    out_dir = args.out_dir or config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.combined:
        combined_table(args, extract_sample_id, extract_polarization, out_dir)
    else:
        single_table(args, extract_sample_id, extract_polarization, out_dir)

    print(f"\n{os.path.join(out_dir, 'dataset_table.csv')}")
    print(os.path.join(out_dir, "dataset_table.tex"))


if __name__ == "__main__":
    main()
