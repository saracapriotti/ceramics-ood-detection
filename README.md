# ceramics-ood-detection

Code for the paper *Recognizing unknown fabrics in the petrographic classification of
Levantine archaeological ceramics through out-of-distribution detection*
(S. Capriotti, D. Genovese, S. Mignardi, S. Scardapane, L. Medeghini).

A DynamicViT (LV-ViT-M backbone with token pruning) is trained to classify petrographic
micrographs of thin sections (PPL and XPL, 20 fabric classes) and is then used to
recognise samples that do not belong to any known fabric. Three detectors, one per
family, are compared: Mahalanobis distance (feature distance), maximum softmax
probability (output uncertainty) and Isolation Forest (density/support), in two settings:

- **leave-one-fabric-out control**: a known fabric (class 1 or class 16) is removed from
  training and must be recognised as new;
- **case study**: a held-out assemblage of thin sections never seen during training,
  containing both known fabrics and *loners*.

## Repository layout

```
ceramics-ood-detection/
├── src/            all the code (run the scripts from here)
├── data/           dataset, not included (see "Data")
├── pretrained/     DynamicViT weights (see "Setup")
├── DynamicViT/     official DynamicViT repository (see "Setup")
├── checkpoints/    leave-one-fabric-out models   (created by the scripts)
└── outputs/        results, tables and figures   (created by the scripts)
```

All paths are set in `src/config.py` and resolved relative to the repository root.
`DATASET_DIR` and `DYNAMICVIT_REPO` can also be set as environment variables.

## Setup

1. Clone the official DynamicViT repository into the repository root and patch it for
   recent PyTorch versions (`torch._six` was removed):

   ```bash
   git clone https://github.com/raoyongming/DynamicViT.git
   sed -i 's/from torch._six import inf/inf = float("inf")/' DynamicViT/utils.py
   ```

2. Install PyTorch for your platform (see https://pytorch.org/get-started), then the
   other dependencies (Python >= 3.10):

   ```bash
   pip install -r requirements.txt
   ```

3. Download the ImageNet-pretrained DynamicViT backbone:

   ```bash
   mkdir -p pretrained
   gdown 1dNab1B5ZOTVNpnpO6H1TsXKFM8BAlA3I -O pretrained/dynamic-vit_lv-m_r0.7.pth
   ```

## Data

The dataset is not included in the repository. All CSV files use `;` as separator.

| File (in `data/`) | Content |
|---|---|
| `dataset.csv` | Training set: `IMAGE_ID;CLASS`, image path relative to `final_dataset/` and fabric number (1-20) |
| `final_dataset/` | Training micrographs |
| `OOD-DATASET.csv` | Held-out assemblage: `IMAGE_ID;SAMPLE_ID` |
| `OOD-DATASET/` | Held-out micrographs |
| `OOD-DATASET_ground_truth.csv` | Petrographic attribution of the held-out samples: `sample_id;true_class`, a fabric number for known fabrics and `NOVEL` for loners |


The sample identifier and the polarization (PPL/XPL) of each training image are parsed
from its file name (`extract_sample_id` and `extract_polarization` in
`dataset_samples.py`), so that all images of a thin section always fall in the same split.

## Code

| File | Purpose |
|---|---|
| `config.py` | Paths, hyperparameters, seed, preprocessing |
| `dataset_samples.py` | Dataset, multi-view preprocessing (whole image + two square halves), sample-level split |
| `model.py` | DynamicViT with new classification heads; multi-view wrapper |
| `train.py` | Training loop (class-weighted cross-entropy + token keep-ratio loss) |
| `inference_utils.py` | Temperature scaling and PPL+XPL late fusion |
| `preresize_images.py` | One-off: downscaled copies of the images for faster loading |
| `run_cv.py` | 5-fold cross-validation grouped by sample and stratified by fabric (Tables 2 and 3) |
| `ood_utils.py` | Feature extraction, Mahalanobis distance, MSP and the other output-based scores |
| `sklearn_ood_baselines.py` | Isolation Forest and the other scikit-learn one-class detectors |
| `run_ood_cv.py` | Novelty detection on the held-out assemblage with the 5 cross-validation models |
| `run_loco_ood.py` | Leave-one-fabric-out control |
| `ood_ground_truth_eval.py` | Evaluation against the petrographic ground truth (used by `run_ood_cv.py`) |
| `operating_point_eval.py` | Recall and false alarms at a percentile threshold of the validation distribution |
| `consensus.py` | Cross-fold consensus grouping of the flagged samples |
| `dataset_table.py` | Table 1 (dataset composition) |
| `three_method_tables.py` | Tables 4 and 5 (three detectors, control and case study) |
| `method_comparison.py` | Threshold-free comparison (AUROC / AUPR / FPR@95TPR) |
| `all_figures.py` | Figures 2-9 and supplementary S1-S2 (`--list` shows them) |

Figure 1 (site map) was made in QGIS.

## Reproducing the results

Run from `src/`. The paper uses the 85th percentile of the validation distribution as
threshold.

```bash
cd src
GT=../data/OOD-DATASET_ground_truth.csv

# 1. Training and closed-set evaluation (Tables 2 and 3)
python preresize_images.py
python run_cv.py --name whole_halves

# 2. Novelty detection
python run_ood_cv.py --cv-name whole_halves --ground-truth $GT
python run_loco_ood.py --classes 1 16 --seeds 100 101 102
python sklearn_ood_baselines.py --classes 1 16 --seeds 100 101 102

python operating_point_eval.py --setting case-study --sweep --percentile 85 --ground-truth $GT
python operating_point_eval.py --setting loco --held-out-class 1  --seeds 100 101 102 --sweep
python operating_point_eval.py --setting loco --held-out-class 16 --seeds 100 101 102 --sweep

python consensus.py --from-detail ../outputs/detail_case_study_p85.csv \
    --method mahalanobis --suffix maha

# 3. Tables and figures
python dataset_table.py --combined --ground-truth $GT
python three_method_tables.py
python method_comparison.py --ground-truth $GT
python all_figures.py
```

Outputs are written to `outputs/`:

- `cv/whole_halves/`: per-fold checkpoints and predictions; `summary.csv` holds the
  values of Table 2, and `run_cv.py` prints the per-fabric report of Table 3;
- `ood/whole_halves/`: scores per fold, metrics and cached features;
- `loco_ood/`: leave-one-fabric-out metrics and scores;
- `operating_point_*.csv`, `detail_case_study_p85.csv` (verdict of each detector for
  every held-out sample), tables as LaTeX, and `figures/whole_halves/`.

Each figure in `all_figures.py` is skipped, with a message saying what to run first, if
its inputs are missing.

## License

MIT, see `LICENSE`.
