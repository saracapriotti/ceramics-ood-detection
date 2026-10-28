"""
Utilities for out-of-distribution detection on ceramic thin-section micrographs.

  - OODDataset / make_clean_copy
  - extract_features_and_logits: one forward pass -> logits + features
    (mean of all tokens after model.norm, class token included, 512-d)
  - compute_id_statistics: class means + shared within-class covariance
    (Lee et al., 2018) with Ledoit-Wolf shrinkage
  - mahalanobis_distances: distance to the closest known class
  - compute_ood_threshold: percentile of the validation distances,
    aggregated per sample (sample_id)
  - analyze_ood_dataset: MSP, entropy, energy, ODIN and Mahalanobis per image

Implementation notes:
  - The shared covariance is estimated on class-centred features
    (feature - mean of its own class), so it describes only the variability
    within fabrics and not the variance between them.
  - Without shrinkage, the sample covariance uses N - C degrees of freedom.
  - Classes with no training images have NaN means and are ignored when
    computing the minimum distance.
  - Mahalanobis distances are computed in chunks to limit memory use.
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from torch.utils.data import Dataset

import config

STATUS_NEW = "NEW CLASS / OUTLIER"
STATUS_KNOWN = "KNOWN CLASS"


# -- OOD dataset ----------------------------------------------------------------

class OODDataset(Dataset):
    """Reads a CSV with columns (IMAGE_ID, SAMPLE_ID), ';' separator.

    A corrupted or missing file does not stop the script: it returns the
    sentinel label -1 and is discarded downstream.
    """

    def __init__(self, csv_file: str, img_dir: str, transform=None, sep: str = ";"):
        df = pd.read_csv(csv_file, sep=sep)
        df.columns = [c.strip().replace("\u00ef\u00bb\u00bf", "") for c in df.columns]  # strip UTF-8 BOM

        col_img, col_sample = df.columns[0], df.columns[1]
        df = df.rename(columns={col_sample: "sample_id"}).reset_index(drop=True)

        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.col_img = col_img
        self.image_paths = self.df[col_img].apply(
            lambda x: os.path.join(img_dir, str(x).replace("\\", "/"))
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths.iloc[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, 0  # placeholder label: image loaded, no known class
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, config.CROP_SIZE, config.CROP_SIZE), -1


def make_clean_copy(fabrics_dataset, transform):
    """Same dataset (same df, sample_ids and labels) with a deterministic
    transform instead of the training augmentation. Used to compute the
    in-distribution statistics on the training set."""
    from dataset_samples import FabricsDataset

    return FabricsDataset(fabrics_dataset.df, img_dir=fabrics_dataset.img_dir, transform=transform)


# -- Features + logits in a single forward pass ---------------------------------

def extract_features_and_logits(model, dataset, device, batch_size: int = 32):
    """Run the model over the dataset and collect logits and features.

    Iterates over the dataset directly (no DataLoader) to keep the
    index -> dataset.df row alignment even when some images are skipped.

    Features: output of model.norm averaged over all tokens (class token
    included), i.e. the input of the classification heads. In eval mode the
    number of tokens after pruning is the same for every image, so the
    average is always taken over the same number of tokens.

    Returns a list of dicts with: index, sample_id (None if absent), label
    (0-based class for FabricsDataset, placeholder for OODDataset), logits,
    probs, feature.
    """
    model.eval()
    has_sample_id = hasattr(dataset, "df") and "sample_id" in dataset.df.columns

    features_buffer = []

    def hook_fn(module, inp, out):
        if out.dim() == 3:
            x = out.mean(dim=1)
        elif out.dim() == 4:
            x = out.mean(dim=[2, 3])
        else:
            x = out
        features_buffer.append(x.detach().float().cpu())

    handle = model.norm.register_forward_hook(hook_fn)

    records = []
    n = len(dataset)

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                imgs, labels, idxs = [], [], []
                for i in range(start, end):
                    img, label = dataset[i]
                    if label == -1:
                        continue
                    imgs.append(img)
                    labels.append(label)
                    idxs.append(i)

                if not imgs:
                    continue

                features_buffer.clear()
                batch = torch.stack(imgs).to(device)
                output = model(batch)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                logits = logits.float()

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                logits_np = logits.cpu().numpy()

                if len(features_buffer) != 1:
                    raise RuntimeError(
                        f"The hook on model.norm was called {len(features_buffer)} times "
                        "in a single forward pass: check the architecture."
                    )
                feats = features_buffer[0].numpy()

                for j, i in enumerate(idxs):
                    records.append(
                        {
                            "index": i,
                            "sample_id": dataset.df.loc[i, "sample_id"] if has_sample_id else None,
                            "label": labels[j],
                            "logits": logits_np[j],
                            "probs": probs[j],
                            "feature": feats[j],
                        }
                    )
    finally:
        handle.remove()

    return records


# -- In-distribution statistics (Mahalanobis) -----------------------------------

def compute_id_statistics(
    model,
    clean_train_ds,
    device,
    num_classes: int,
    batch_size: int = 32,
    shrinkage: bool = None,
    records: list = None,
):
    """Class means and inverse of the shared within-class covariance
    (Lee et al., 2018), computed on training features WITHOUT augmentation
    (clean_train_ds must use val_transform, see make_clean_copy).

    The covariance is estimated on features centred on their own class mean,
    so it describes only the variability within fabrics, not between them.
    Ledoit-Wolf shrinkage keeps the estimate well conditioned when the
    feature dimension (512) is large relative to the number of images per
    class.

    `records` can be passed when the training features have already been
    extracted (e.g. for the sklearn detectors), to avoid a second pass over
    the images.

    Returns (means, inv_cov): means is a list of C vectors; classes with no
    training images have NaN means and are ignored by mahalanobis_distances.
    """
    shrinkage = config.COVARIANCE_SHRINKAGE if shrinkage is None else shrinkage

    if records is None:
        records = extract_features_and_logits(model, clean_train_ds, device, batch_size)
    features = np.stack([r["feature"] for r in records]).astype(np.float64)
    labels = np.array([r["label"] for r in records], dtype=int)
    dim = features.shape[1]

    means = []
    for c in range(num_classes):
        class_feats = features[labels == c]
        if len(class_feats) == 0:
            print(f"WARNING: no training images for class {c}: excluded from the distance computation.")
            means.append(np.full(dim, np.nan))
        else:
            means.append(class_feats.mean(axis=0))

    # Shared within-class covariance: each feature minus the mean of its class
    means_arr = np.stack(means)
    centered = features - means_arr[labels]

    if shrinkage:
        # Data are already class-centred: no additional global mean to subtract
        cov = LedoitWolf(assume_centered=True).fit(centered).covariance_
    else:
        n_classes_present = int(np.sum(~np.isnan(means_arr[:, 0])))
        dof = max(len(centered) - n_classes_present, 1)
        cov = centered.T @ centered / dof

    inv_cov = np.linalg.pinv(cov + np.eye(dim) * 1e-6)

    print(
        f"ID statistics: {len(features)} images, {dim} dimensions, "
        f"shared within-class covariance ({'Ledoit-Wolf' if shrinkage else 'sample'})"
    )
    return means, inv_cov


def mahalanobis_distances(
    features: np.ndarray,
    means: list,
    inv_cov: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Mahalanobis distance to the closest class for each row of `features`.
    Classes with NaN means (no training images) are ignored. Vectorised and
    computed in chunks."""
    means_arr = np.stack(means)
    valid = ~np.isnan(means_arr).any(axis=1)
    if not valid.any():
        raise ValueError("No class with a valid mean: cannot compute the Mahalanobis distance.")
    means_arr = means_arr[valid]  # [C, D]

    features = np.asarray(features, dtype=np.float64)
    out = np.empty(len(features))
    for start in range(0, len(features), chunk_size):
        chunk = features[start:start + chunk_size]
        diffs = chunk[:, None, :] - means_arr[None, :, :]          # [n, C, D]
        sq = np.einsum("ncd,de,nce->nc", diffs, inv_cov, diffs)    # [n, C]
        out[start:start + chunk_size] = np.sqrt(np.clip(sq, 0, None)).min(axis=1)
    return out


def compute_ood_threshold(
    model,
    val_ds,
    device,
    means: list,
    inv_cov: np.ndarray,
    percentile: float = None,
    batch_size: int = 32,
):
    """Threshold = percentile of the per-sample mean distances on the
    validation set (known classes only). val_ds already uses val_transform."""
    percentile = percentile if percentile is not None else config.OOD_PERCENTILE

    records = extract_features_and_logits(model, val_ds, device, batch_size)
    features = np.stack([r["feature"] for r in records])
    sample_ids = [r["sample_id"] for r in records]

    distances = mahalanobis_distances(features, means, inv_cov)
    df = pd.DataFrame({"sample_id": sample_ids, "mahalanobis": distances})
    sample_distances = df.groupby("sample_id")["mahalanobis"].mean()

    threshold = float(np.percentile(sample_distances, percentile))
    print(f"Mean distance of known samples (validation): {sample_distances.mean():.3f}")
    return threshold, sample_distances


# -- OOD dataset analysis -------------------------------------------------------

def analyze_ood_dataset(
    model,
    ood_ds,
    device,
    means: list,
    inv_cov: np.ndarray,
    odin_temperature: float = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Single forward pass computing MSP, entropy, energy, ODIN (high
    temperature, no input perturbation) and Mahalanobis distance for every
    image. Features are kept in the `feature_vector` column for clustering.

    Conventions: msp and odin are CONFIDENCES (high = likely known);
    entropy, energy and mahalanobis are already oriented (high = likely new).
    """
    odin_temperature = odin_temperature or config.ODIN_TEMPERATURE

    records = extract_features_and_logits(model, ood_ds, device, batch_size)
    logits_all = np.stack([r["logits"] for r in records]).astype(np.float64)
    probs_all = np.stack([r["probs"] for r in records]).astype(np.float64)
    features_all = np.stack([r["feature"] for r in records])
    sample_ids = [r["sample_id"] for r in records]

    msp = probs_all.max(axis=1)
    preds = probs_all.argmax(axis=1)
    entropy = -(probs_all * np.log(probs_all + 1e-12)).sum(axis=1)
    energy = -logsumexp(logits_all, axis=1)
    odin_score = softmax(logits_all / odin_temperature, axis=1).max(axis=1)
    mahal = mahalanobis_distances(features_all, means, inv_cov)

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "pred_class_idx": preds,
            "msp": msp,
            "entropy": entropy,
            "energy": energy,
            "odin": odin_score,
            "mahalanobis": mahal,
        }
    )
    df["feature_vector"] = list(features_all)
    return df
