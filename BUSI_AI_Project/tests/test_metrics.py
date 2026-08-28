"""
Unit Tests for Segmentation Metrics, Classification Metrics, and Loss Functions.

Validates:
- Dice similarity coefficient and IoU on known toy inputs.
- Pixel precision, recall, and specificity.
- Classification ROC-AUC, PR-AUC, sensitivity, specificity, balanced accuracy.
- Loss function mathematical sanity and differentiability.
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.losses import CombinedBCEDiceLoss, DiceLoss
from src.metrics import (
    compute_dice_coefficient,
    compute_iou,
    compute_pixel_metrics,
    evaluate_classification,
    evaluate_segmentation_sample,
)


def test_dice_and_iou_perfect_match():
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[2:6, 2:6] = 1.0
    pred = gt.copy()

    dice = compute_dice_coefficient(pred, gt)
    iou = compute_iou(pred, gt)

    assert pytest.approx(dice, rel=1e-5) == 1.0
    assert pytest.approx(iou, rel=1e-5) == 1.0


def test_dice_and_iou_complete_mismatch():
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[0:3, 0:3] = 1.0
    pred = np.zeros((10, 10), dtype=np.float32)
    pred[5:8, 5:8] = 1.0

    dice = compute_dice_coefficient(pred, gt)
    iou = compute_iou(pred, gt)

    assert dice < 1e-4
    assert iou < 1e-4


def test_dice_and_iou_partial_overlap():
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[0:4, 0:4] = 1.0  # Area = 16

    pred = np.zeros((10, 10), dtype=np.float32)
    pred[2:6, 0:4] = 1.0  # Area = 16
    # Overlap is row 2,3 col 0..3 -> 2*4 = 8 pixels

    # Dice = 2 * 8 / (16 + 16) = 16 / 32 = 0.50
    # IoU = 8 / (16 + 16 - 8) = 8 / 24 = 0.3333...
    dice = compute_dice_coefficient(pred, gt)
    iou = compute_iou(pred, gt)

    assert pytest.approx(dice, rel=1e-3) == 0.50
    assert pytest.approx(iou, rel=1e-3) == 1.0 / 3.0


def test_pixel_precision_recall_specificity():
    gt = np.zeros((10, 10), dtype=np.float32)
    gt[0:2, 0:2] = 1.0  # 4 positives

    pred = np.zeros((10, 10), dtype=np.float32)
    pred[0:2, 0:4] = 1.0  # 8 predicted positive (4 TP, 4 FP, 0 FN, 92 TN)

    metrics = compute_pixel_metrics(pred, gt)
    assert metrics["tp"] == 4
    assert metrics["fp"] == 4
    assert metrics["fn"] == 0
    assert metrics["tn"] == 92
    assert pytest.approx(metrics["precision"], rel=1e-3) == 0.50
    assert pytest.approx(metrics["recall"], rel=1e-3) == 1.00
    assert pytest.approx(metrics["specificity"], rel=1e-3) == 92.0 / 96.0


def test_classification_metrics_perfect():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.05, 0.9, 0.85, 0.95])

    res = evaluate_classification(y_true, y_prob, threshold=0.5)
    assert pytest.approx(res["roc_auc"], rel=1e-4) == 1.0
    assert pytest.approx(res["pr_auc"], rel=1e-4) == 1.0
    assert pytest.approx(res["sensitivity"], rel=1e-4) == 1.0
    assert pytest.approx(res["specificity"], rel=1e-4) == 1.0
    assert pytest.approx(res["f1"], rel=1e-4) == 1.0
    assert pytest.approx(res["balanced_accuracy"], rel=1e-4) == 1.0


def test_dice_loss_backprop():
    loss_fn = DiceLoss(from_logits=True)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()

    loss = loss_fn(logits, target)
    assert loss.item() >= 0.0
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_combined_bce_dice_loss():
    loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
    target = torch.ones(2, 1, 16, 16)

    loss = loss_fn(logits, target)
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
