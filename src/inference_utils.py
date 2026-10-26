"""
Temperature scaling and sample-level late fusion of the PPL + XPL images.

Usage (as in run_cv.py):

    records = collect_logits(model, val_ds, device)
    T = fit_temperature(logits, labels)
    probs_cal = apply_temperature(logits, T)
    sample_df = aggregate_by_sample(records_with_calibrated_probs)
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def collect_logits(model, dataset, device, batch_size: int = 32):
    """Run the model over the dataset and collect per-image predictions.

    Iterates over the dataset directly so that records stay aligned with
    dataset.df even if some images fail to load. For each valid image returns:
    index, sample_id, polarization, label, logits, probs.
    """
    model.eval()
    records = []
    n = len(dataset)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            imgs, labels, idxs = [], [], []
            for i in range(start, end):
                img, label = dataset[i]
                if label < 0:
                    continue
                imgs.append(img)
                labels.append(label)
                idxs.append(i)

            if not imgs:
                continue

            batch = torch.stack(imgs).to(device)
            outputs = model(batch)
            logits = (outputs[0] if isinstance(outputs, (tuple, list)) else outputs).float()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            logits_np = logits.cpu().numpy()

            for j, i in enumerate(idxs):
                records.append(
                    {
                        "index": i,
                        "sample_id": dataset.df.loc[i, "sample_id"],
                        "polarization": dataset.df.loc[i, "polarization"],
                        "label": labels[j],
                        "logits": logits_np[j],
                        "probs": probs[j],
                    }
                )

    return records


# -- Calibration ----------------------------------------------------------------

def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of logits scaled by 1/temperature."""
    logits = np.asarray(logits, dtype=np.float64)
    lp = log_softmax(logits / temperature, axis=1)
    return float(-lp[np.arange(len(labels)), np.asarray(labels)].mean())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 20.0,
    verbose: bool = True,
) -> float:
    """Temperature scaling (Guo et al., 2017): find T minimising the NLL.

    Optimises beta = 1/T in [1/t_max, 1/t_min]. The NLL is convex in beta,
    so a bounded scalar search deterministically finds the global minimum.
    T < 1 means the model is under-confident, T > 1 over-confident.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    def objective(beta):
        return nll(logits, labels, temperature=1.0 / beta)

    res = minimize_scalar(
        objective,
        bounds=(1.0 / t_max, 1.0 / t_min),
        method="bounded",
        options={"xatol": 1e-5},
    )
    T = float(1.0 / res.x)

    if verbose:
        print(f"  Validation NLL with T=1:     {nll(logits, labels, 1.0):.4f}")
        print(f"  Validation NLL with T={T:.3f}: {nll(logits, labels, T):.4f}")
        if T <= t_min * 1.01 or T >= t_max * 0.99:
            print(
                f"  WARNING: T={T:.3f} is at the boundary of [{t_min}, {t_max}]; "
                "calibration is unreliable (check the validation logits)."
            )
    return T


# -- Sample-level fusion --------------------------------------------------------

def aggregate_by_sample(records: list) -> pd.DataFrame:
    """Late fusion over all images (PPL + XPL) sharing a sample_id.

    Computes soft voting (mean of probabilities) and hard voting (majority of
    argmax predictions). Hard-voting ties are broken by the highest mean
    probability among the tied classes.
    """
    by_sample = defaultdict(list)
    for r in records:
        by_sample[r["sample_id"]].append(r)

    rows = []
    for sid, items in by_sample.items():
        true_label = items[0]["label"]
        mean_probs = np.stack([it["probs"] for it in items]).mean(axis=0)
        soft_pred = int(mean_probs.argmax())

        votes = pd.Series([int(it["probs"].argmax()) for it in items])
        vote_counts = votes.value_counts()
        tied = vote_counts[vote_counts == vote_counts.max()].index.tolist()
        if len(tied) == 1:
            hard_pred = int(tied[0])
        else:
            hard_pred = int(tied[int(np.argmax([mean_probs[c] for c in tied]))])

        rows.append(
            {
                "sample_id": sid,
                "true_label": true_label,
                "n_images": len(items),
                "polarizations": "+".join(sorted(set(it["polarization"] for it in items))),
                "soft_pred": soft_pred,
                "hard_pred": hard_pred,
                "soft_confidence": float(mean_probs[soft_pred]),
            }
        )

    return pd.DataFrame(rows)
