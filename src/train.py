"""
Training loop: class-weighted cross-entropy with label smoothing (plus the
token keep-ratio loss of the pruning modules when the model is the DynamicViT),
AdamW with linear warmup and cosine decay (one scheduler step per batch), mixed
precision on GPU. The same recipe is used for all the architectures (see
config.ARCH). Called by run_cv.py and run_loco_ood.py.
"""

import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import config

IS_DVIT = config.ARCH == "dvit"


def build_param_groups(model: torch.nn.Module, weight_decay: float):
    """Split parameters into two groups: with weight decay (weight matrices,
    ndim >= 2) and without (bias, LayerNorm, cls_token, pos_embed), following
    standard Vision Transformer practice (DeiT, timm)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "cls_token" in name
            or "pos_embed" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    print(
        f"Weight decay: {len(decay)} tensors decayed (weight_decay={weight_decay}), "
        f"{len(no_decay)} excluded (bias/norm/cls_token/pos_embed, weight_decay=0.0)"
    )
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int):
    """Step-wise scheduler: linear warmup for LR_WARMUP_EPOCHS, then cosine
    decay down to base_lr * LR_MIN_FACTOR. Call .step() after every batch."""
    warmup_steps = config.LR_WARMUP_EPOCHS * steps_per_epoch
    total_steps = config.NUM_EPOCHS * steps_per_epoch
    min_lr_factor = config.LR_MIN_FACTOR

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_ratio_loss(decision_masks, target_ratios: torch.Tensor) -> torch.Tensor:
    """MSE between the actual and target token keep ratio at each pruning stage."""
    loss = 0.0
    for mask, target in zip(decision_masks, target_ratios):
        actual_ratio = mask.mean(dim=1)
        loss += F.mse_loss(actual_ratio, torch.full_like(actual_ratio, target))
    return loss / len(decision_masks)


def get_class_weights(dataset, device: torch.device) -> torch.Tensor:
    """'Balanced' class weights from the labels of the training set."""
    all_labels = dataset.labels.values
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(all_labels),
        y=all_labels,
    )
    return torch.tensor(weights, dtype=torch.float).to(device)


def train(
    model,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device: torch.device,
    best_model_path: str,
    metrics_csv: str,
):
    """Train the model, saving the weights with the best validation accuracy
    to best_model_path and the per-epoch metrics to metrics_csv."""
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    criterion_cls = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
    target_ratios = torch.tensor(config.TOKEN_RATIO, device=device) if IS_DVIT else None

    param_groups = build_param_groups(model, config.WEIGHT_DECAY)
    optimizer = optim.AdamW(param_groups, lr=config.LEARNING_RATE)
    scheduler = build_scheduler(optimizer, steps_per_epoch=len(train_loader))

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)

    best_val_acc = 0.0
    epochs_no_improve = 0

    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        if IS_DVIT:
            model.distill = True  # makes DynamicViT return the pruning decisions

        total_loss = correct = total = 0
        n_batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(amp_device, enabled=use_amp):
                if IS_DVIT:
                    logits, _aux, _prev_decision, decision_masks = model(images)
                    loss = criterion_cls(logits, labels)
                    loss = loss + config.LAMBDA_RATIO * compute_ratio_loss(
                        decision_masks, target_ratios)
                else:
                    out = model(images)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = criterion_cls(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100.0 * correct / max(total, 1)
        train_loss = total_loss / max(n_batches, 1)

        # -- Validation --------------------------------------------------------
        model.eval()
        val_correct = val_total = 0
        val_loss_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, labels in val_loader:
                if images.size(0) == 0:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                val_loss_total += criterion_cls(logits, labels).item()
                val_batches += 1
                _, predicted = logits.max(1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_loss = val_loss_total / max(val_batches, 1)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"\nEpoch {epoch+1}  |  lr={current_lr:.2e}  |  train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%"
            f"  |  val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%"
        )

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("  Saved best model")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.PATIENCE:
            print("Early stopping triggered")
            break

    print(f"\nTraining complete - best val acc: {best_val_acc:.2f}%")
    return model
