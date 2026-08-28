"""
Evaluation Metrics Module for Breast Ultrasound (BUSI) AI Pipeline.

Provides metric computation for:
1. Binary Segmentation (Dice, IoU, Pixel Precision, Pixel Recall/Sensitivity, Pixel Specificity)
2. Binary Classification (ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1, Balanced Accuracy, Confusion Matrix)

Design:
- Completely decoupled from training code.
- Operates on NumPy arrays or PyTorch Tensors.
- Robust to edge cases (e.g. all zeros, zero division epsilon handling).
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ==============================================================================
# SEGMENTATION METRICS
# ==============================================================================

def compute_dice_coefficient(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Compute Dice Similarity Coefficient (DSC) between binary masks.
    Formula: 2 * |A intersect B| / (|A| + |B| + eps)
    """
    pred_bin = (pred_mask > 0.5).astype(np.float32).flatten()
    gt_bin = (gt_mask > 0.5).astype(np.float32).flatten()

    intersection = np.sum(pred_bin * gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)

    if total == 0:
        return 1.0

    dice = (2.0 * intersection + eps) / (total + eps)
    return float(dice)


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Compute Jaccard Index / Intersection over Union (IoU).
    Formula: |A intersect B| / (|A union B| + eps)
    """
    pred_bin = (pred_mask > 0.5).astype(np.float32).flatten()
    gt_bin = (gt_mask > 0.5).astype(np.float32).flatten()

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin) - intersection

    if union == 0:
        return 1.0

    iou = (intersection + eps) / (union + eps)
    return float(iou)


def compute_pixel_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Compute pixel-level Precision, Recall (Sensitivity), and Specificity.
    """
    pred_bin = (pred_mask > 0.5).astype(np.bool_).flatten()
    gt_bin = (gt_mask > 0.5).astype(np.bool_).flatten()

    tp = np.sum(pred_bin & gt_bin)
    fp = np.sum(pred_bin & ~gt_bin)
    fn = np.sum(~pred_bin & gt_bin)
    tn = np.sum(~pred_bin & ~gt_bin)

    precision = float((tp + eps) / (tp + fp + eps)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = float((tp + eps) / (tp + fn + eps)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    specificity = float((tn + eps) / (tn + fp + eps)) if (tn + fp) > 0 else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_segmentation_sample(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all primary and secondary segmentation metrics for a single image/mask pair.
    """
    dice = compute_dice_coefficient(pred_mask, gt_mask)
    iou = compute_iou(pred_mask, gt_mask)
    pixel_stats = compute_pixel_metrics(pred_mask, gt_mask)

    return {
        "dice": dice,
        "iou": iou,
        "precision": pixel_stats["precision"],
        "recall": pixel_stats["recall"],
        "specificity": pixel_stats["specificity"],
    }


# ==============================================================================
# CLASSIFICATION METRICS
# ==============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full suite of classification metrics as specified in PRD §12:
    - ROC-AUC (Primary summary metric)
    - PR-AUC
    - Sensitivity / Recall (Malignant=1)
    - Specificity
    - Precision
    - F1-Score
    - Balanced Accuracy
    - Confusion Matrix (Absolute & Row-Normalized)
    """
    y_true = np.asarray(y_true, dtype=int).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float64).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    try:
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        else:
            roc_auc = float("nan")
    except Exception:
        roc_auc = float("nan")

    # PR-AUC
    try:
        if len(np.unique(y_true)) > 1:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(rec_curve, prec_curve))
        else:
            pr_auc = float("nan")
    except Exception:
        pr_auc = float("nan")

    # Sensitivity / Recall
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))

    # Precision
    precision = float(precision_score(y_true, y_pred, zero_division=0))

    # F1 Score
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Balanced Accuracy
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Normalized Confusion Matrix (row-normalized by true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    }
