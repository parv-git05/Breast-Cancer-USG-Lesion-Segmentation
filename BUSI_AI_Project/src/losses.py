"""
Loss Functions for Binary Lesion Segmentation.

Implements:
1. Soft Dice Loss (differentiable, smooth sigmoid formulation)
2. BCEWithLogitsLoss (numerically stable binary cross-entropy)
3. Combined Dice + BCE Loss with configurable weighting
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss operating on logits or probabilities.
    Dice = 2 * (pred * target).sum() / (pred.sum() + target.sum() + eps)
    Loss = 1.0 - Dice
    """
    def __init__(self, from_logits: bool = True, eps: float = 1e-6):
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits (if from_logits=True) or probabilities of shape (B, 1, H, W).
            target: Binary target tensor of shape (B, 1, H, W) with values {0.0, 1.0}.
        """
        if self.from_logits:
            probs = torch.sigmoid(pred)
        else:
            probs = pred

        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (probs_flat * target_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss:
    Total Loss = bce_weight * BCE(logits, target) + dice_weight * Dice(logits, target)
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
        eps: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(from_logits=True, eps=eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice
