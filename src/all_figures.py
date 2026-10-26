"""
All the figures of the paper (except Figure 1, the map, made in QGIS) in a
single script.

Numbering follows the published paper:

    fig2   micrograph plate (one per fabric)
    fig3   F1 per fabric
    fig4   confusion matrix
    fig5   structure of the 20 fabrics (A: dendrogram, B: confusion vs distance)
    fig6   recall / false alarms against threshold, leave-one-fabric-out
    fig7   per-sample novelty score, three detectors (case study)
    fig8   consensus dendrogram + micrographs of the discovered groups
    fig9   comparison of the three architectures (D-ViT, DeiT-small, ResNet50)
    figS1  as fig7, extended version with all sample names (supplementary)
    figS2  full consensus dendrogram, not restricted to the groups (supplementary)

Each figure is independent: if its inputs are missing it is skipped with a
message saying what to run first, and the others are still produced.

Usage:

    # list the available figures
    python all_figures.py --list

    # everything the available data allow (default paths from config.py)
    python all_figures.py

    # only some of them
    python all_figures.py --only fig3 fig4 fig9

Inputs required by each figure (paths default to config.py):

    fig2   data/fig2_selection.csv (columns fabric;folder;image;name;magnification)
           and the training micrographs (--fabric-root)
    fig3   outputs/cv/<cv-name>/per_sample_predictions.csv
    fig4   same as fig3
    fig5   same as fig3 + outputs/ood/<cv-name>/features/features_fold*.npz
    fig6   outputs/operating_point_fabric_<c>_excluded.csv for each --loco-classes
           (written by operating_point_eval.py --sweep)
    fig7   outputs/detail_case_study_p<percentile>.csv
           (written by operating_point_eval.py --setting case-study)
    fig8   outputs/coassociation_maha.csv (written by consensus.py), the OOD
           dataset CSV and the OOD micrographs (--ood-root)
    fig9   none: the values are in the ARCH_AUROC dictionary (fig9 section)
    figS1  same as fig7
    figS2  outputs/coassociation_maha.csv

The user-provided CSV files (fig2 selection, sample names, group images) are
read with ';' as separator. The Italian column headers of earlier versions
(nome, cartella, immagine, ingrandimento, sottotitolo) are also accepted.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import f1_score

import config

# =========================================================================
# SHARED STYLE - same palette and rcParams for every figure
# =========================================================================
# Reordered Okabe-Ito palette: adjacent pairs remain distinguishable with
# colour-vision deficiencies. Colour is never the only channel: marker shape
# or line style always carries the same information.
BLUE, ORANGE, GREEN, VERMILLION = "#1B4F72", "#E69F00", "#009E73", "#D55E00"
BORDEAUX = "#A32C33"
SKY, YELLOW, PURPLE, BLACK = "#56B4E9", "#F0E442", "#CC79A7", "#000000"
INK, INK_SOFT, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"
LIGHT_GRID = "#ececea"
SCAFFOLD = "#8f8f8f"

NOVEL_LABELS = {"NOVEL", "NEW", "LONER", "NUOVA", "NUOVO"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Italian column headers of earlier versions of the user-provided CSV files
LEGACY_COLUMNS = {"nome": "name", "cartella": "folder", "immagine": "image",
                  "ingrandimento": "magnification", "sottotitolo": "subtitle"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.labelpad": 12,
    "axes.edgecolor": INK_SOFT,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

def read_user_csv(path):
    """Read a ';'-separated user CSV as strings, mapping legacy Italian headers."""
    d = pd.read_csv(path, sep=";", dtype=str).fillna("")
    d.columns = [c.strip() for c in d.columns]
    return d.rename(columns=LEGACY_COLUMNS)


def clean_axes(ax, x_grid=False, y_grid=True):
    """Less non-data ink: no frame, thin grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.5)
    if x_grid:
        ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


DEFAULT_OFFSETS = [(7, 4), (-7, 4), (7, -9), (-7, -9),
                   (0, 11), (0, -13), (7, 14), (-7, 14),
                   (0, 19), (0, -21), (14, 19), (-14, 19)]


def sparse_labels(ax, points, fontsize=7.5, colour=None, offsets=None, forbid_x=None):
    """Place labels next to points without overlaps.

    Candidate positions are compared in pixels; the cost combines overlap
    with already placed labels and distance from the point. `forbid_x` keeps
    labels off vertical reference lines (e.g. the threshold)."""
    if colour is None:
        colour = INK_SOFT
    if offsets is None:
        offsets = DEFAULT_OFFSETS
    ax.figure.canvas.draw()
    transform = ax.transData.transform
    pt = ax.figure.dpi / 72.0

    def overlap(a, b):
        return (max(0.0, min(a[2], b[2]) - max(a[0], b[0])) *
                max(0.0, min(a[3], b[3]) - max(a[1], b[1])))

    forbidden = []
    if forbid_x:
        y0, _ = ax.get_ylim()
        for xv in forbid_x:
            bx = transform((xv, y0))[0]
            forbidden.append((bx - 3 * pt, -1e9, bx + 3 * pt, 1e9))
    occupied = []

    for x, y, text in points:
        px, py = transform((x, y))
        w = 0.68 * fontsize * len(text) * pt + 5 * pt
        h = 1.05 * fontsize * pt
        choice, box, min_cost = None, None, None
        fallback = None
        area = max(w * h, 1.0)
        for dx, dy in offsets:
            ox, oy = dx * pt, dy * pt
            x0 = px + ox if dx > 0 else (px + ox - w if dx < 0 else px - w / 2)
            trial = (x0, py + oy - h / 2, x0 + w, py + oy + h / 2)
            if fallback is None:
                fallback = ((dx, dy), trial)
            if any(overlap(trial, v) > 0 for v in forbidden):
                continue
            covered = sum(overlap(trial, b) for b in occupied) / area
            cost = 3.0 * covered + (abs(dx) + abs(dy)) / 8.0
            if min_cost is None or cost < min_cost:
                choice, box, min_cost = (dx, dy), trial, cost
        if box is None:
            choice, box = fallback
        occupied.append(box)
        dx, dy = choice
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    va="center", fontsize=fontsize, color=colour)


def panel_letter(ax, text, dx=-0.13, dy=1.02):
    """A) and B) at the top left of the panel, outside the data area."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=11,
            color=INK, va="bottom", ha="left")


def draw_dendrogram(ax, Z, labels, threshold=None, cluster_colour=BLUE,
                    label_fontsize=8, grid=False,
                    base_width=0.8, cluster_width=2.0,
                    base_colour=None):
    """Dendrogram drawn by hand (not dendrogram(ax=...)): solid strokes,
    rounded joins, grey scaffold, colour only for the groups below the
    threshold. Returns the leaf order."""
    d = dendrogram(Z, labels=list(labels), no_plot=True,
                   color_threshold=threshold if threshold is not None else -1,
                   above_threshold_color="__base__")

    accents = cluster_colour if isinstance(cluster_colour, (list, tuple)) else [cluster_colour]
    assigned = {}
    for xs, ys, c in zip(d["dcoord"], d["icoord"], d["color_list"]):
        inside = c != "__base__"
        if inside and c not in assigned:
            assigned[c] = accents[len(assigned) % len(accents)]
        ax.plot(xs, ys,
                color=assigned[c] if inside else (base_colour or SCAFFOLD),
                linewidth=cluster_width if inside else base_width,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=4 if inside else 3)

    n = len(d["ivl"])
    ax.set_yticks([5 + 10 * i for i in range(n)], d["ivl"], fontsize=label_fontsize)
    ax.set_ylim(0, 10 * n)
    xmax = max((max(x) for x in d["dcoord"]), default=1.0) or 1.0
    ax.set_xlim(-0.012 * xmax, 1.02 * xmax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.tick_params(axis="y", pad=6)
    ax.tick_params(axis="x", length=3, color=INK_SOFT)
    return d["ivl"]


# Internal figure name -> figure number in the paper. fig2, fig8 and fig9
# are saved directly under their number and do not go through save().
NUMBER = {
    "fig_per_fabric_f1":              "fig3",
    "fig_confusion_matrix":           "fig4",
    "fig_fabric_structure":           "fig5",
    "fig_loco_curve":                 "fig6",
    "fig_ood_swarm":                  "fig7",
    "fig_ood_swarm_list":             "figS1",
}


def missing(name, *items):
    """Print which input is missing and skip the figure."""
    for label, obj in items:
        if obj is None or (hasattr(obj, "__len__") and len(obj) == 0):
            print(f"  skipping {name}: missing {label}")
            return True
    return False


def save(fig, ctx, name: str):
    number = NUMBER.get(name, name)
    for ext in ctx.formats:
        fig.savefig(os.path.join(ctx.fig_dir, f"{number}.{ext}"),
                    dpi=ctx.dpi, bbox_inches="tight")
    plt.close(fig)
    label = f"{number}." + "/.".join(ctx.formats)
    print(f"  {label}" + (f"   ({name[4:]})" if number != name else ""))


def save_direct(fig, fig_dir, formats, dpi, number: str):
    """Like save(), for the figures (fig2, fig8, fig9, figS2) saved directly
    under their number."""
    for ext in formats:
        fig.savefig(os.path.join(fig_dir, f"{number}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  {number}." + "/.".join(formats))


# =========================================================================
# CONTEXT - data are read once, and only when needed
# (fig3, fig4, fig5, fig6, fig7, figS1)
# =========================================================================

class Ctx:
    def __init__(self, args):
        self.cv_name = args.cv_name
        self.cv_dir = os.path.join(config.REPO_DIR, "outputs", "cv", args.cv_name)
        self.ood_dir = os.path.join(config.REPO_DIR, "outputs", "ood", args.cv_name)
        self.fig_dir = args.out_dir or os.path.join(
            config.REPO_DIR, "outputs", "figures", args.cv_name)
        os.makedirs(self.fig_dir, exist_ok=True)
        self.percentile = args.percentile
        self.names = args.names
        self.loco_classes = args.loco_classes
        self.formats = args.formats
        self.dpi = args.dpi
        self._cache = {}

    def _csv(self, path):
        if path in self._cache:
            return self._cache[path]
        self._cache[path] = pd.read_csv(path) if os.path.exists(path) else None
        return self._cache[path]

    @property
    def samples(self):
        return self._csv(os.path.join(self.cv_dir, "per_sample_predictions.csv"))

    @property
    def fabric_distance(self):
        if "fabdist" in self._cache:
            return self._cache["fabdist"]
        self._cache["fabdist"] = self._compute_fabric_distance()
        return self._cache["fabdist"]

    def _compute_fabric_distance(self):
        """Mahalanobis distance between class prototypes, from the cached
        training features of each fold, normalised and averaged over folds."""
        feat_dir = os.path.join(self.ood_dir, "features")
        files = sorted(os.listdir(feat_dir)) if os.path.isdir(feat_dir) else []
        files = [f for f in files if f.startswith("features_fold") and f.endswith(".npz")]
        if not files or self.samples is None:
            return None

        label_of = dict(zip(self.samples["sample_id"], self.samples["true_label"]))
        classes = sorted(set(label_of.values()))
        matrices = []

        for f in files:
            z = np.load(os.path.join(feat_dir, f), allow_pickle=True)
            X = z["train"]
            ids = [str(s) for s in z["train_ids"]]
            y = np.array([label_of.get(s, -1) for s in ids])
            keep = y >= 0
            X, y = X[keep], y[keep]
            if len(X) < len(classes) * 2:
                continue

            means, residuals = {}, []
            for c in classes:
                Xc = X[y == c]
                if len(Xc) == 0:
                    continue
                means[c] = Xc.mean(axis=0)
                residuals.append(Xc - means[c])
            if len(means) < len(classes):
                continue
            prec = LedoitWolf().fit(np.vstack(residuals)).get_precision()

            P = np.stack([means[c] for c in classes])
            D = np.zeros((len(classes), len(classes)))
            for i in range(len(classes)):
                diff = P - P[i]
                D[i] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, prec, diff), 0))
            off = D[~np.eye(len(classes), dtype=bool)]
            matrices.append(D / off.mean())

        if not matrices:
            return None

        tri = np.triu_indices(len(classes), k=1)
        vecs = [M[tri] for M in matrices]
        rho = [spearmanr(vecs[i], vecs[j]).correlation
               for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        return dict(D=np.mean(matrices, axis=0), classes=classes,
                    n_folds=len(matrices), median_rho=float(np.median(rho)) if rho else np.nan)

# =========================================================================
# FIG. 3 - F1 per fabric
# =========================================================================

def fig_per_fabric_f1(ctx):
    if missing("per_fabric_f1", ("per_sample_predictions.csv", ctx.samples)):
        return
    s = ctx.samples
    classes = sorted(set(s["true_label"]))
    f1 = f1_score(s["true_label"], s["soft_pred"], labels=classes, average=None, zero_division=0)
    n_samples = s.groupby("true_label").size().reindex(classes).to_numpy()

    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(classes) + 1.2))
    y = np.arange(len(classes))
    ax.barh(y, f1[order], color=BLUE, height=0.7)
    ax.set_yticks(y, [f"{classes[i] + 1}  (n={n_samples[i]})" for i in order])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 (sample level, soft voting)")
    for yi, v in zip(y, f1[order]):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK_SOFT)
    clean_axes(ax, x_grid=True, y_grid=False)
    save(fig, ctx, "fig_per_fabric_f1")


# =========================================================================
# FIG. 4 - confusion matrix
# =========================================================================

def fig_confusion_matrix(ctx):
    if missing("confusion_matrix", ("per_sample_predictions.csv", ctx.samples)):
        return
    s = ctx.samples
    classes = sorted(set(s["true_label"]) | set(s["soft_pred"]))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((n, n))
    for t, p in zip(s["true_label"], s["soft_pred"]):
        M[idx[t], idx[p]] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    Mn = M / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    labels = [str(c + 1) for c in classes]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted fabric")
    ax.set_ylabel("True fabric")
    for i in range(n):
        for j in range(n):
            if M[i, j] > 0:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if Mn[i, j] > 0.55 else INK)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.outline.set_visible(False)
    save(fig, ctx, "fig_confusion_matrix")


# =========================================================================
# FIG. 5 - structure of the 20 fabrics (A: dendrogram, B: confusion vs distance)
# =========================================================================

def _dendrogram_on_ax(ctx, ax, label_fontsize=9, grid=False):
    fd = ctx.fabric_distance
    D, classes = fd["D"].copy(), fd["classes"]
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")

    cut = 0.62 * Z[:, 2].max()
    draw_dendrogram(ax, Z, [str(c + 1) for c in classes], threshold=cut,
                    cluster_colour=[BLUE, BORDEAUX, GREEN, PURPLE, VERMILLION],
                    label_fontsize=label_fontsize, grid=grid)
    ax.set_xlabel("Mahalanobis distance between class prototypes")
    return fd


def _confusion_vs_distance_on_ax(ctx, ax, legend_loc="upper right"):
    fd = ctx.fabric_distance
    s, D, classes = ctx.samples, fd["D"], fd["classes"]

    errors = defaultdict(int)
    for t, p in zip(s["true_label"], s["soft_pred"]):
        if t != p:
            errors[tuple(sorted((t, p)))] += 1

    dist, n_err, pairs = [], [], []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            dist.append(D[i, j])
            n_err.append(errors.get(tuple(sorted((classes[i], classes[j]))), 0))
            pairs.append(f"{classes[i] + 1}\u2013{classes[j] + 1}")
    dist, n_err = np.array(dist), np.array(n_err)
    rho, p = spearmanr(dist, n_err)

    exchanged = n_err > 0
    rng = np.random.default_rng(0)
    y = n_err.astype(float)
    y[~exchanged] += rng.uniform(-0.12, 0.12, (~exchanged).sum())
    ax.scatter(dist[~exchanged], y[~exchanged], s=22, color=GRID, edgecolor="none",
               label="no samples exchanged", zorder=2)
    ax.scatter(dist[exchanged], y[exchanged], s=50, color=BLUE, edgecolor="white",
               linewidth=0.6, label="at least one exchanged", zorder=3)
    ax.set_xlabel("Mahalanobis distance between prototypes")
    ax.set_ylabel("Samples exchanged between the two fabrics")
    ax.legend(frameon=False, fontsize=8, loc=legend_loc, labelspacing=0.4,
              handletextpad=0.5)
    clean_axes(ax)
    ax.tick_params(axis="x", length=3, color=INK_SOFT)
    sparse_labels(ax, [(d, e, lab) for d, e, lab in zip(dist, n_err, pairs) if e >= 3],
                  fontsize=7)
    return rho, p


def fig_fabric_structure(ctx):
    fd = ctx.fabric_distance
    if missing("fabric_structure", ("per_sample_predictions.csv", ctx.samples),
               ("features/features_fold*.npz", fd)):
        return
    n = len(fd["classes"])
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12.6, 0.17 * n + 1.3),
        gridspec_kw=dict(width_ratios=(1.0, 1.12), wspace=0.16))

    _dendrogram_on_ax(ctx, ax_a, label_fontsize=8, grid=False)
    rho, p = _confusion_vs_distance_on_ax(ctx, ax_b)

    for ax in (ax_a, ax_b):
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9.5)
    ax_b.yaxis.label.set_size(9.5)

    panel_letter(ax_a, "A)", dx=-0.09)
    panel_letter(ax_b, "B)", dx=-0.11)

    save(fig, ctx, "fig_fabric_structure")
    print(f"     agreement between folds: median Spearman {fd['median_rho']:.2f} "
          f"over {fd['n_folds']} folds")
    print(f"     Spearman rho = {rho:.2f}, p = {p:.1e}  (for the caption)")


# =========================================================================
# FIG. 6 - recall / false alarms against threshold, leave-one-fabric-out
# =========================================================================

METHOD_STYLE = {
    "Mahalanobis distance": (BLUE,       "-",  "o"),
    "MSP":                  (BORDEAUX,   "--", "s"),
    "Isolation Forest":     (GREEN,      ":",  "^"),
    "Predictive entropy":   (PURPLE,     "-.", "v"),
    "Free energy":          (ORANGE,     "-",  "D"),
    "ODIN (T=1000)":        (SKY,        "--", "P"),
    "Local Outlier Factor": (GREEN,      "-.", "X"),
    "Elliptic Envelope":    (YELLOW,     ":",  "*"),
    "One-Class SVM":        (INK_SOFT,   "-",  "h"),
}
REPRESENTATIVES = ["Mahalanobis distance", "MSP", "Isolation Forest"]
CURVE_TITLES = ("Excluded fabric recognised as new", "Known samples wrongly flagged")


def _operating_point(slug):
    p = os.path.join(config.REPO_DIR, "outputs", f"operating_point_{slug}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _curve_pair(d, methods, threshold, ax_r, ax_f, letters, x_label):
    for m in methods:
        q = d[d["method"] == m].sort_values("percentile")
        if q.empty:
            continue
        colour, linestyle, marker = METHOD_STYLE.get(m, (BLUE, "-", "o"))
        for ax, col in ((ax_r, "recall_novel"), (ax_f, "false_alarms")):
            ax.plot(q["percentile"], q[col], color=colour, linestyle=linestyle,
                    marker=marker, markersize=4, linewidth=1.2, label=m,
                    clip_on=False)

    percentiles = sorted(d["percentile"].unique())
    for k, (ax, title) in enumerate(((ax_r, CURVE_TITLES[0]), (ax_f, CURVE_TITLES[1]))):
        ax.axvline(threshold, color=GRID, linestyle="--", linewidth=1.0, zorder=1)
        if x_label:
            ax.set_xlabel("Percentile of the validation distribution")
        ax.set_ylabel("%")
        ax.set_title(f"{letters[k]} {title}")
        ax.set_ylim(-2, 102)
        ax.set_xlim(min(percentiles) - 1, max(percentiles) + 1)
        ax.set_xticks(percentiles, [f"{p:g}" for p in percentiles])
        clean_axes(ax, y_grid=False)
        ax.tick_params(axis="x", length=3, color=INK_SOFT)


def fig_loco_curve(ctx):
    classes = ctx.loco_classes or [1, 16]
    data = []
    for c in classes:
        d = _operating_point(f"fabric_{c}_excluded")
        if d is None:
            print(f"  skipping loco_curve: missing operating_point_fabric_{c}_excluded.csv")
            print(f"     run: python operating_point_eval.py --setting loco "
                  f"--held-out-class {c} --seeds 100 101 102 --sweep")
            return
        data.append((c, d))

    present = set(data[0][1]["method"])
    methods = [m for m in REPRESENTATIVES if m in present]
    if len(methods) < len(REPRESENTATIVES):
        print("  warning, missing: " + ", ".join(m for m in REPRESENTATIVES if m not in present))
    if not methods:
        print("  skipping loco_curve: none of the three detectors is in the file")
        return

    nr = len(data)
    fig, axes = plt.subplots(nr, 2, figsize=(10.4, 3.7 * nr), squeeze=False)
    alphabet = "abcdefgh"
    for r, (held_out, d) in enumerate(data):
        _curve_pair(d, methods, ctx.percentile, axes[r][0], axes[r][1],
                    (f"({alphabet[2 * r]})", f"({alphabet[2 * r + 1]})"),
                    x_label=(r == nr - 1))
        axes[r][0].set_ylabel(f"Class {held_out} excluded (%)")
    axes[0][0].legend(frameon=False, loc="lower left", handlelength=2.4, labelspacing=0.35)
    fig.tight_layout(pad=1.0, h_pad=2.0)
    save(fig, ctx, "fig_loco_curve")
    print(f"     vertical line at the {ctx.percentile:g}th percentile")


# =========================================================================
# FIG. 7 / FIG. S1 - per-sample novelty score, case study
# =========================================================================

DETAIL_METHODS = [("pct_mahalanobis", "Mahalanobis distance"),
                  ("pct_msp",         "MSP"),
                  ("pct_isolation",   "Isolation Forest")]


def _swarm(x, bin_width=2.5, step=0.075, maximum=0.3):
    """Vertical offsets for a 1-D swarm plot: points in the same bin are
    spread alternately above and below the axis."""
    x = np.asarray(x, float)
    off = np.zeros(len(x))
    counts = {}
    for i in np.argsort(x):
        b = int(x[i] // bin_width)
        k = counts.get(b, 0)
        counts[b] = k + 1
        off[i] = min(((k + 1) // 2) * step, maximum) * (1 if k % 2 else -1)
    return off


def _sample_aliases(ctx):
    """Optional display names for the samples (CSV sample_id;name)."""
    if "alias" in ctx._cache:
        return ctx._cache["alias"]
    p = ctx.names or os.path.join(config.DATASET_DIR, "sample_names.csv")
    a = {}
    if os.path.isfile(p):
        d = read_user_csv(p)
        a = {str(r["sample_id"]).strip(): str(r["name"]).strip()
             for _, r in d.iterrows() if str(r["name"]).strip()}
        print(f"     sample names from {os.path.basename(p)}: {len(a)} samples")
    ctx._cache["alias"] = a
    return a


def _case_study_detail(ctx):
    p = os.path.join(config.REPO_DIR, "outputs",
                     f"detail_case_study_p{int(ctx.percentile)}.csv")
    if not os.path.exists(p):
        print(f"  missing {os.path.basename(p)}")
        print("     run: python operating_point_eval.py --setting "
              f"case-study --percentile {ctx.percentile:g} --ground-truth <gt>")
        return None, None
    d = pd.read_csv(p)
    d["is_novel"] = d["is_novel"].astype(bool)
    methods = [(c, n) for c, n in DETAIL_METHODS if c in d.columns]
    if not methods:
        print(f"  {os.path.basename(p)} has no pct_* columns")
        return None, None
    return d, methods


def fig_ood_swarm(ctx):
    d, methods = _case_study_detail(ctx)
    if d is None:
        print("  skipping ood_swarm")
        return
    known, loners = d[~d["is_novel"]], d[d["is_novel"]]

    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, name) in zip(axes, methods):
        for y, group, colour in ((1, known, BLUE), (0, loners, BORDEAUX)):
            x = group[col].to_numpy()
            ax.scatter(x, y + _swarm(x), s=20, color=colour, linewidth=0.4,
                       edgecolor="white", zorder=3, clip_on=False)
            ax.text(105, y, f"{int((x > ctx.percentile).sum())}/{len(x)}",
                    color=colour, va="center", ha="left")
        ax.axvline(ctx.percentile, color=INK, linestyle="--", linewidth=1.0, zorder=1)
        ax.set_title(name)
        ax.set_xlim(0, 122)
        ax.set_ylim(-0.6, 1.6)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Novelty percentile")
        clean_axes(ax, y_grid=False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.tick_params(axis="x", length=3, color=INK_SOFT)
    axes[0].set_yticks([1, 0], ["Known\nfabric", "Loner"])
    fig.tight_layout(pad=0.8, w_pad=2.2)
    save(fig, ctx, "fig_ood_swarm")


def fig_ood_swarm_list(ctx):
    d, methods = _case_study_detail(ctx)
    if d is None:
        print("  skipping ood_swarm_list")
        return
    key = methods[0][0]
    known = d[~d["is_novel"]].sort_values(key, ascending=False)
    loners = d[d["is_novel"]].sort_values(key, ascending=False)
    ordered = pd.concat([known, loners])
    n = len(ordered)
    y = np.arange(n)[::-1]
    colours = [BORDEAUX if v else BLUE for v in ordered["is_novel"]]

    fig, axes = plt.subplots(1, len(methods),
                             figsize=(3.3 * len(methods), max(5.0, 0.16 * n)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (col, name) in zip(axes, methods):
        for yy in y:
            ax.axhline(yy, color=LIGHT_GRID, linewidth=0.4, zorder=0)
        ax.axhspan(-0.5, y[len(known)] + 0.5, color="#f9eef1", zorder=0)
        ax.axvline(ctx.percentile, color=INK, linestyle="--", linewidth=1.0, zorder=2)
        ax.scatter(ordered[col], y, s=16, c=colours, linewidth=0.4,
                   edgecolor="white", zorder=3)
        ax.set_title(name)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Novelty percentile")
        clean_axes(ax, y_grid=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=3, color=INK_SOFT)
    axes[0].set_ylim(-0.8, n - 0.2)
    alias = _sample_aliases(ctx)
    axes[0].set_yticks(y, [alias.get(str(s), str(s)) for s in ordered["sample_id"]], fontsize=6.5)
    fig.tight_layout(pad=0.8, w_pad=1.4)
    save(fig, ctx, "fig_ood_swarm_list")


# =========================================================================
# FIG. 2 - micrograph plate, one per fabric
# =========================================================================

def normalise_name(s):
    """File-name key that ignores spaces, commas, dots, dashes, underscores and case."""
    return re.sub(r"[\s,._\-]+", "", str(s)).lower()


def index_images(root):
    """(normalised stem, full path) for every image under root."""
    index = []
    for folder, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                index.append((normalise_name(os.path.splitext(f)[0]), os.path.join(folder, f)))
    return index


def fig2_find(index, ident, hint=""):
    """Resolve an image identifier to a path. `hint` (a folder name) breaks
    ties between files with the same name in different folders."""
    target = normalise_name(ident)

    def narrow(cand):
        if len(cand) > 1 and hint:
            strict = [p for p in cand if normalise_name(hint) in normalise_name(os.path.dirname(p))]
            if strict:
                return strict
        return cand

    literal = [p for _, p in index
               if os.path.basename(p) == ident
               or os.path.splitext(os.path.basename(p))[0] == ident]
    if len(literal) == 1:
        return literal[0], ""
    if len(literal) > 1:
        literal = narrow(literal)
        if len(literal) == 1:
            return literal[0], ""
        return None, ("two files with the same name in different folders:\n        "
                      + "\n        ".join(literal))

    exact = narrow([p for n, p in index if n == target])
    if len(exact) == 1:
        return exact[0], ""
    if len(exact) > 1:
        return None, ("several candidate files. Write in the CSV the EXACT name "
                      "of the one you want, extension included:\n        "
                      + "\n        ".join(os.path.basename(p) for p in exact))
    partial = narrow([p for n, p in index if n.startswith(target)])
    if not partial:
        partial = narrow([p for n, p in index if target in n])
    if len(partial) == 1:
        return partial[0], f"found as {os.path.basename(partial[0])}"
    if len(partial) > 1:
        return None, ("several candidate files. Write in the CSV the EXACT name "
                      "of the one you want, extension included:\n        "
                      + "\n        ".join(os.path.basename(p) for p in partial))
    return None, "no matching file"


def fig2_magnification(name):
    """Magnification parsed from a file name, e.g. '...2.5X...' -> '2.5X'."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*[xX](?![a-zA-Z])", os.path.basename(name))
    if not m:
        return None
    return m.group(1).replace(",", ".") + "X"


def fig2_lookup(magnification, table):
    if not table or magnification is None:
        return None
    for k, v in table.items():
        if normalise_name(k) == normalise_name(magnification):
            return v
    return None


def fig2_square(path, side):
    """Central square crop, downscaled to `side` pixels if larger."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    c = min(w, h)
    im = im.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))
    if c > side:
        im = im.resize((side, side), Image.LANCZOS)
    return im, c


def fig2_scale_bar(ax, side_px, um_per_px, forced=None, fontsize=5.0):
    """Draw a scale bar; returns the geometry needed for its background box."""
    if not um_per_px:
        return None
    um = side_px * um_per_px
    length = forced
    if length is None:
        for cand in (5000, 2000, 1000, 500, 200, 100, 50, 20, 10):
            if um / 6 <= cand <= um / 2.5:
                length = cand
                break
    if length is None:
        length = max(10, int(round(um / 4 / 10) * 10))
    if length > 0.92 * um:
        print(f"    warning: a {length:g} um bar does not fit in the panel, "
              f"which is {um:.0f} um wide")
    text = (f"{length:g} $\\mu$m" if length < 1000 else f"{length / 1000:g} mm")
    fr = min(length / um, 0.86)
    h, y0, right = 0.016, 0.050, 0.955
    ax.add_patch(Rectangle((right - fr, y0), fr, h, transform=ax.transAxes,
                           facecolor=INK, edgecolor="none", zorder=5))
    t = ax.text(right, y0 + h + 0.012, text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fontsize, color=INK, zorder=5)
    return ax, right - fr, y0, fr, h, right, t


def fig2_bar_backgrounds(fig, bars):
    """Semi-transparent white box behind each scale bar and its label."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, x0, y0, fr, h, right, t in bars:
        inv = ax.transAxes.inverted()
        bb = t.get_window_extent(renderer)
        (tx0, ty0) = inv.transform((bb.x0, bb.y0))
        (tx1, ty1) = inv.transform((bb.x1, bb.y1))
        X0, X1 = min(x0, tx0) - 0.018, max(right, tx1) + 0.018
        Y0, Y1 = y0 - 0.020, max(y0 + h, ty1) + 0.018
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0,
                               transform=ax.transAxes, facecolor="white",
                               alpha=0.82, edgecolor="none", zorder=4))


def fig2_title(fabric, name, magnification, show_magnification, max_char=30):
    first = f"Fabric {fabric}"
    if show_magnification and magnification:
        first += f" ({magnification})"
    second = (name or "").strip()
    if len(second) > max_char:
        k = second.rfind(" ", 0, max_char)
        k = k if k > 0 else second.rfind("-", 0, max_char)
        if k > 0:
            second = second[:k + 1].rstrip() + "\n" + second[k + 1:].lstrip()
    return first + "\n" + second


def fig2_fabric_plate(args, fig_dir):
    """Figure 2: one micrograph per fabric. See --help for the options
    (--fabric-root, --selection, --scale/--field/--bar)."""
    selection = args.selection
    if not os.path.isfile(selection):
        print(f"  skipping fig2: {selection} not found")
        return
    if not os.path.isdir(args.fabric_root):
        print(f"  skipping fig2: {args.fabric_root} not found")
        return

    scale, field, bar = {}, {}, {}
    for s in args.scale:
        k, v = s.split("=", 1)
        scale[k.strip()] = float(v)
    for c in args.field:
        k, v = c.split("=", 1)
        field[k.strip()] = float(v)
    for b in args.bar:
        k, v = b.split("=", 1)
        bar[k.strip()] = float(v)

    d = read_user_csv(selection)
    d["fabric"] = d["fabric"].astype(int)
    if args.fabrics:
        d = d[d["fabric"].isin(args.fabrics)]

    index = index_images(args.fabric_root)
    if not index:
        print(f"  skipping fig2: no images under {args.fabric_root}")
        return
    print(f"  fig2: {len(index)} images under {args.fabric_root}")

    rows = d[d["image"].str.strip() != ""]
    if rows.empty:
        print("  skipping fig2: no row of the CSV has the image column filled in")
        return

    found, problems = [], []
    for _, r in rows.iterrows():
        path, msg = fig2_find(index, r["image"], r.get("folder", "").strip())
        if path is None:
            problems.append((r["fabric"], r["image"], msg))
            print(f"  fig2: fabric {r['fabric']:>2}  NOT RESOLVED: {msg}")
        else:
            mag = (r.get("magnification", "") or "").strip() or fig2_magnification(path)
            upp = fig2_lookup(mag, scale)
            field_um = fig2_lookup(mag, field)
            if field_um:
                with Image.open(path) as _im:
                    upp = field_um / _im.size[1]
            found.append((int(r["fabric"]), r.get("name", ""), path, mag, upp,
                          fig2_lookup(mag, bar)))

    if not found:
        print("  skipping fig2: no images found")
        return
    if problems:
        print(f"  fig2: {len(problems)} rows not resolved, drawing the others")

    n = len(found)
    nc = min(args.columns, n)
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(2.20 * nc, 2.45 * nr))
    axes = np.atleast_1d(np.asarray(axes)).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    bars = []
    border = "#c8c8c4"
    for ax, (fab, name, path, mag, upp, forced) in zip(axes, found):
        im, side_px = fig2_square(path, args.side)
        ax.imshow(np.asarray(im))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(border)
            sp.set_linewidth(0.6)
        b = fig2_scale_bar(ax, side_px, upp, forced, args.bar_font)
        if b:
            bars.append(b)
        ax.set_title(fig2_title(fab, name, mag, args.show_magnification),
                     fontsize=args.title_font, color=INK, pad=3, linespacing=1.3)
    fig.subplots_adjust(wspace=0.05, hspace=0.30)
    fig2_bar_backgrounds(fig, bars)
    save_direct(fig, fig_dir, args.formats, args.dpi, "fig2")

    magnifications = sorted({t[3] for t in found if t[3]})
    if len(magnifications) > 1:
        print(f"     note for the caption: the panels are NOT all at the same "
              f"magnification ({', '.join(magnifications)}).")


# =========================================================================
# FIG. 8 / FIG. S2 - consensus dendrogram + micrographs of the groups
# =========================================================================

GROUP_COLOURS = ["#1B4F72", "#A32C33", "#009E73", "#CC79A7", "#D55E00"]


def read_sample_names(path):
    if not path or not os.path.isfile(path):
        return {}
    d = read_user_csv(path)
    return {str(r["sample_id"]).strip(): str(r["name"]).strip()
            for _, r in d.iterrows() if str(r["name"]).strip()}


def group_find(index, ident):
    literal = [p for _, p in index
               if os.path.basename(p) == ident
               or os.path.splitext(os.path.basename(p))[0] == ident]
    if len(literal) == 1:
        return literal[0], ""
    b = normalise_name(ident)
    exact = [p for n, p in index if n == b]
    if len(exact) == 1:
        return exact[0], ""
    cand = exact or [p for n, p in index if n.startswith(b)] or [p for n, p in index if b in n]
    if len(cand) == 1:
        return cand[0], f"found as {os.path.basename(cand[0])}"
    if len(cand) > 1:
        return None, ("several candidate files, write the exact name in the CSV:\n        "
                      + "\n        ".join(os.path.basename(p) for p in cand))
    return None, "no matching file"


def group_magnification(name):
    m = re.search(r"(\d+[.,]?\d*)\s*[xX]", str(name))
    if not m:
        return float("inf")
    try:
        return float(m.group(1).replace(",", "."))
    except ValueError:
        return float("inf")


def is_crossed_polars(name):
    parts = re.split(r"[\s,._\-]+", str(name).lower())
    return any(p in ("px", "xpl", "xp", "n+") for p in parts)


def choose_image(names):
    """Prefer crossed polars, then the lowest magnification."""
    return sorted(names, key=lambda n: (not is_crossed_polars(n), group_magnification(n), str(n)))


def from_coassociation(path):
    co = pd.read_csv(path, index_col=0)
    if list(co.index) != list(co.columns):
        sys.exit(f"{path} is not a square matrix with the same names on rows and columns")
    ids = [str(s).strip() for s in co.index]
    D = 1.0 - co.to_numpy(dtype=float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return ids, D


def consensus_groups(ids, D, cut):
    """Groups with more than one sample at the given cut, the linkage
    restricted to them, their names and the number of isolated samples."""
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=cut, criterion="distance")
    sizes = pd.Series(labels).value_counts()
    inside = [i for i, e in enumerate(labels) if sizes[e] > 1]
    if len(inside) < 2:
        sys.exit(f"No sample forms a group at cut {cut:g}: nothing to draw.")

    Dr = D[np.ix_(inside, inside)]
    names = [ids[i] for i in inside]
    Zr = linkage(squareform(Dr, checks=False), method="average")

    leaves = list(reversed(dendrogram(Zr, labels=names, no_plot=True)["ivl"]))
    lab_r = fcluster(Zr, t=cut, criterion="distance")
    group_of = dict(zip(names, lab_r))
    groups, seen = [], {}
    for s in leaves:
        e = group_of[s]
        if e not in seen:
            seen[e] = len(groups)
            groups.append([])
        groups[seen[e]].append(s)
    return groups, Zr, names, len(ids) - len(inside)


def draw_group_dendrogram(ax, Z, labels, cut, colour_of, alias=None):
    """Dendrogram whose branches are coloured by the group of their leaf
    (colour_of), so that colours match the frames of the micrographs.
    Unlike draw_dendrogram(), which colours by scipy threshold."""
    d = dendrogram(Z, labels=list(labels), no_plot=True,
                   color_threshold=cut, above_threshold_color="__base__")
    for xs, ys, c in zip(d["dcoord"], d["icoord"], d["color_list"]):
        if c == "__base__":
            ax.plot(xs, ys, color=SCAFFOLD, linewidth=0.9, solid_capstyle="round", zorder=2)
            continue
        i = int(round((np.mean(ys) - 5) / 10))
        i = min(max(i, 0), len(d["ivl"]) - 1)
        ax.plot(xs, ys, color=colour_of.get(d["ivl"][i], GROUP_COLOURS[0]),
                linewidth=2.4, solid_capstyle="round", solid_joinstyle="round", zorder=3)

    n = len(d["ivl"])
    alias = alias or {}
    ax.set_yticks([5 + 10 * i for i in range(n)], [alias.get(s, s) for s in d["ivl"]])
    for t, s in zip(ax.get_yticklabels(), d["ivl"]):
        t.set_color(colour_of.get(s, INK))
    ax.set_ylim(0, 10 * n)
    ax.set_xlim(-0.012, 1.02)
    ax.axvline(cut, color=INK, linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlabel("Consensus distance")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", length=3, color=INK_SOFT)


def fig8_consensus_grouping(args, fig_dir):
    """Figure 8: consensus dendrogram restricted to the groups + their
    micrographs, one row per group."""
    coassociation = args.coassociation
    alias = read_sample_names(args.group_names)
    show = lambda s: alias.get(s, s)

    if os.path.exists(coassociation):
        ids, D = from_coassociation(coassociation)
        print(f"  fig8: read {coassociation}: {len(ids)} flagged samples")
    else:
        print(f"  skipping fig8: {coassociation} not found")
        print("     run first: python consensus.py --from-detail "
              "../outputs/detail_case_study_p85.csv --method mahalanobis --suffix maha")
        return

    groups, Z, names, n_isolated = consensus_groups(ids, D, args.consensus_cut)
    colour_of = {s: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, g in enumerate(groups) for s in g}
    print(f"  fig8: {len(groups)} groups at cut {args.consensus_cut:g}, {n_isolated} isolated")
    for i, g in enumerate(groups):
        print(f"      group {i + 1}: " + ", ".join(
            f"{show(s)} ({s})" if show(s) != s else s for s in g))

    choice = {}
    if args.group_images:
        tab = read_user_csv(args.group_images)
        choice = {str(r["sample_id"]).strip(): (r["image"].strip(), r.get("subtitle", "").strip())
                  for _, r in tab.iterrows()}
    elif args.ood_dataset:
        d = pd.read_csv(args.ood_dataset, sep=None, engine="python")
        d.columns = [c.strip().lower().lstrip("\ufeff") for c in d.columns]
        c_id = next(c for c in d.columns if "sample" in c)
        c_img = next(c for c in d.columns if c != c_id and "fabric" not in c)
        c_fab = next((c for c in d.columns if "fabric" in c), None)
        for sid, g in d.groupby(d[c_id].astype(str).str.strip()):
            if sid not in colour_of:
                continue
            ordered = choose_image(g[c_img].astype(str).str.strip())
            fab = str(g[c_fab].iloc[0]).strip() if c_fab else ""
            choice[sid] = (os.path.splitext(ordered[0])[0], f"fabric {fab}" if fab else "")
    else:
        print("  skipping fig8: --ood-dataset or --group-images is required")
        return

    if not args.ood_root:
        print("  skipping fig8: missing --ood-root")
        return
    if not os.path.isdir(args.ood_root):
        print(f"  skipping fig8: {args.ood_root} not found")
        return
    index = index_images(args.ood_root)
    if not index:
        print(f"  skipping fig8: no images under {args.ood_root}")
        return

    found, problems = {}, 0
    for g in groups:
        for s in g:
            image_name, sub = choice.get(s, ("", ""))
            if not image_name:
                problems += 1
                continue
            path, _ = group_find(index, image_name)
            if path is None:
                problems += 1
            else:
                found[s] = (path, sub)
    if not found:
        print("  skipping fig8: no images found, check --ood-root")
        return
    if problems:
        print(f"  fig8: {problems} samples without an image, drawing the others")

    widest = max(len(g) for g in groups)
    pw = args.panel_width
    ratios = []
    for path, _ in found.values():
        with Image.open(path) as im:
            ratios.append(im.size[0] / im.size[1])
    aspect = args.group_aspect or max(ratios)
    row_height = pw / aspect + 0.40

    dendro_width = 2.4
    fig = plt.figure(figsize=(dendro_width + pw * widest + 0.5,
                              max(2.8, row_height * len(groups) + 0.8)))
    gs = fig.add_gridspec(len(groups), widest + 1,
                          width_ratios=[dendro_width / pw] + [1] * widest,
                          wspace=0.06, hspace=0.34)

    draw_group_dendrogram(fig.add_subplot(gs[:, 0]), Z, names, args.consensus_cut, colour_of, alias)

    for r, g in enumerate(groups):
        for c in range(widest):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(g) or g[c] not in found:
                ax.axis("off")
                continue
            s = g[c]
            path, sub = found[s]
            im = Image.open(path).convert("RGB")
            w, h = im.size
            if args.group_aspect and w / h > args.group_aspect:
                im = im.crop((w - int(round(h * args.group_aspect)), 0, w, h))
            ax.imshow(np.asarray(im), aspect="auto")
            for sp in ax.spines.values():
                sp.set_color(colour_of[s])
                sp.set_linewidth(1.6)
            ax.set_title(show(s) + ("\n" + sub if sub else ""), color=INK,
                         fontsize=8, pad=4, linespacing=1.2)

    save_direct(fig, fig_dir, args.formats, args.dpi, "fig8")


def figS2_full_consensus_dendrogram(args, fig_dir):
    """Figure S2 (supplementary): the FULL consensus dendrogram, not
    restricted to the groups."""
    coassociation = args.coassociation
    if os.path.exists(coassociation):
        ids, D = from_coassociation(coassociation)
    else:
        print(f"  skipping figS2: {coassociation} not found")
        return
    Z = linkage(squareform(D, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(6.0, max(3.2, 0.22 * len(ids))))
    draw_dendrogram(ax, Z, ids, threshold=args.consensus_cut, cluster_colour=BLUE, label_fontsize=7)
    ax.axvline(args.consensus_cut, color=INK, linestyle="--", linewidth=1.0, zorder=2)
    ax.set_xlabel("Consensus distance")
    save_direct(fig, fig_dir, args.formats, args.dpi, "figS2")


# =========================================================================
# FIG. 9 - comparison of the three architectures
# =========================================================================

ARCH_BLUE, ARCH_RED, ARCH_GREEN = "#1f4e79", "#c0504d", "#4f8a3b"

ARCH_STYLE = {
    "mahalanobis":      ("Mahalanobis distance", ARCH_BLUE,  "-",  "o"),
    "msp":              ("MSP",                  ARCH_RED,   "--", "s"),
    "isolation_forest": ("Isolation Forest",     ARCH_GREEN, ":",  "^"),
}

# AUROC mean and standard deviation over three seeds
# ((fabric 1 held out), (fabric 16 held out))
ARCH_AUROC = {
    "D-ViT": {
        "mahalanobis":      ((0.976, 0.012), (0.918, 0.017)),
        "msp":              ((0.908, 0.010), (0.844, 0.094)),
        "isolation_forest": ((0.922, 0.021), (0.808, 0.048)),
    },
    "DeiT-small": {
        "mahalanobis":      ((0.962, 0.021), (0.919, 0.006)),
        "msp":              ((0.906, 0.055), (0.910, 0.017)),
        "isolation_forest": ((0.824, 0.055), (0.668, 0.057)),
    },
    "ResNet50": {
        "mahalanobis":      ((0.928, 0.029), (0.755, 0.035)),
        "msp":              ((0.838, 0.046), (0.934, 0.013)),
        "isolation_forest": ((0.862, 0.056), (0.519, 0.017)),
    },
}
ARCH_X_LABELS = ["Fabric 1\n(distinctive)", "Fabric 16\n(near)"]


def fig9_architectures(args, fig_dir):
    archs = list(ARCH_AUROC)
    fig, axes = plt.subplots(1, len(archs), figsize=(9.0, 3.6), sharey=True)
    x = [0, 1]

    for ax, arch in zip(axes, archs):
        for det, (lab, col, ls, mk) in ARCH_STYLE.items():
            (m1, s1), (m16, s16) = ARCH_AUROC[arch][det]
            ax.errorbar(x, [m1, m16], yerr=[s1, s16],
                        color=col, linestyle=ls, linewidth=1.8,
                        marker=mk, markersize=6.5,
                        markeredgecolor="white", markeredgewidth=0.8,
                        ecolor=col, elinewidth=1.0, capsize=2.5,
                        label=lab if ax is axes[0] else None, zorder=3)
        ax.set_title(arch, pad=8)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks(x, ARCH_X_LABELS)
        clean_axes(ax, y_grid=False)

    axes[0].set_ylim(0.45, 1.01)
    axes[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_ylabel("AUROC")
    axes[0].legend(frameon=False, loc="lower left", handletextpad=0.6, borderaxespad=0.6)
    fig.tight_layout()
    save_direct(fig, fig_dir, args.formats, args.dpi, "fig9")


# =========================================================================
# MAIN
# =========================================================================

CTX_FIGURES = {
    "fig3": fig_per_fabric_f1,
    "fig4": fig_confusion_matrix,
    "fig5": fig_fabric_structure,
    "fig6": fig_loco_curve,
    "fig7": fig_ood_swarm,
    "figS1": fig_ood_swarm_list,
}
ALL_FIGURES = ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9", "figS1", "figS2"]


def main():
    p = argparse.ArgumentParser(
        description="All the figures of the paper (except the map).",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # shared (fig3, fig4, fig5, fig6, fig7, figS1)
    p.add_argument("--cv-name", default="whole_halves")
    p.add_argument("--names", default=None,
                   help="CSV sample_id;name with display names for figS1 "
                        "(default: data/sample_names.csv)")
    p.add_argument("--out-dir", default=None, help="default: outputs/figures/<cv-name>")
    p.add_argument("--percentile", type=float, default=85,
                   help="threshold used in fig6/fig7/figS1 (85th percentile in the paper)")
    p.add_argument("--loco-classes", type=int, nargs="+", default=None)

    # fig2 (plate of the 20 fabrics)
    p.add_argument("--fabric-root", default=config.ORIGINAL_IMG_DIR,
                   help="folder with the training micrographs, for fig2")
    p.add_argument("--selection", default=os.path.join(config.DATASET_DIR, "fig2_selection.csv"),
                   help="CSV with the micrograph chosen for each fabric, for fig2")
    p.add_argument("--columns", type=int, default=5)
    p.add_argument("--side", type=int, default=1000, help="panel side in pixels")
    p.add_argument("--scale", action="append", default=[], metavar="MAG=UM_PER_PIXEL")
    p.add_argument("--field", action="append", default=[], metavar="MAG=MICRON",
                   help="field-of-view height in microns for a magnification")
    p.add_argument("--bar", action="append", default=[], metavar="MAG=MICRON",
                   help="force the scale-bar length for a magnification")
    p.add_argument("--title-font", type=float, default=8.0)
    p.add_argument("--bar-font", type=float, default=5.0)
    p.add_argument("--show-magnification", action="store_true")
    p.add_argument("--fabrics", nargs="+", type=int, default=None, help="for fig2: only these fabrics")

    # fig8 / figS2 (consensus)
    p.add_argument("--coassociation",
                   default=os.path.join(config.REPO_DIR, "outputs", "coassociation_maha.csv"),
                   help="CSV written by consensus.py, for fig8/figS2")
    p.add_argument("--consensus-cut", type=float, default=0.5)
    p.add_argument("--ood-dataset", default=config.OOD_CSV_FILE, help="OOD dataset CSV, for fig8")
    p.add_argument("--group-images", default=None,
                   help="CSV sample_id;image;subtitle chosen by hand, for fig8")
    p.add_argument("--ood-root", default=config.ORIGINAL_OOD_IMG_DIR,
                   help="folder with the OOD micrographs, for fig8")
    p.add_argument("--group-names", default=os.path.join(config.DATASET_DIR, "sample_names.csv"),
                   help="CSV sample_id;name with display names, for fig8")
    p.add_argument("--group-aspect", type=float, default=None, metavar="WIDTH/HEIGHT")
    p.add_argument("--panel-width", type=float, default=2.4)

    # output
    p.add_argument("--only", nargs="+", default=None, help="only these figures, e.g. --only fig3 fig9")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        print("Available figures:")
        for name in ALL_FIGURES:
            print(f"  {name}")
        return

    todo = args.only or ALL_FIGURES
    unknown = [n for n in todo if n not in ALL_FIGURES]
    if unknown:
        raise SystemExit(f"Unknown: {', '.join(unknown)}. Available: {', '.join(ALL_FIGURES)}")

    fig_dir = args.out_dir or os.path.join(config.REPO_DIR, "outputs", "figures", args.cv_name)
    os.makedirs(fig_dir, exist_ok=True)
    ctx = Ctx(args)

    print("=" * 70)
    print(f"Figures - run '{args.cv_name}'")
    print("=" * 70)
    print(f"Output folder: {fig_dir}/\n")

    for name in todo:
        print(name)
        try:
            if name == "fig2":
                fig2_fabric_plate(args, fig_dir)
            elif name == "fig8":
                fig8_consensus_grouping(args, fig_dir)
            elif name == "figS2":
                figS2_full_consensus_dendrogram(args, fig_dir)
            elif name == "fig9":
                fig9_architectures(args, fig_dir)
            else:
                CTX_FIGURES[name](ctx)
        except SystemExit as e:
            print(f"  skipping {name}: {e}")
        except Exception as e:
            print(f"  ERROR in {name}: {type(e).__name__}: {e}")

    print(f"\nDone. All figures in {fig_dir}/")


if __name__ == "__main__":
    main()
