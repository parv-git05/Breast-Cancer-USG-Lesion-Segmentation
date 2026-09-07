"""
Visualization Module for Ultrasound Lesion Segmentation.

Features:
- Overlay creation: Grayscale ultrasound image + Ground-Truth mask (Green) + Predicted mask (Red/Yellow).
- Batch grid generation for train / val / test sanity checking (Gate 3 verification).
- Failure case visualization (worst Dice score predictions).
- Non-interactive rendering: outputs directly to PNG files in outputs/reports/.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import BUSIDataset
from src.transforms import build_transforms
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize BUSI dataset samples or predictions.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/split.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split to visualize."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=16,
        help="Number of samples to visualize in grid."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (defaults to outputs/reports/<split>_samples.png)."
    )
    return parser.parse_args()


def create_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create an RGB overlay of image and binary masks.
    - Image: Grayscale normalized [0, 1]
    - Ground Truth Mask: Green tint
    - Predicted Mask: Red tint (Overlap produces Yellow)
    """
    img = np.squeeze(image).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    rgb = np.stack([img, img, img], axis=-1)
    gt = np.squeeze(gt_mask) > 0.5

    if pred_mask is not None:
        pred = np.squeeze(pred_mask) > 0.5
        rgb[pred, 0] = np.clip(rgb[pred, 0] * (1.0 - alpha) + 1.0 * alpha, 0, 1)
        rgb[pred, 1] = rgb[pred, 1] * (1.0 - alpha)
        rgb[pred, 2] = rgb[pred, 2] * (1.0 - alpha)

    rgb[gt, 1] = np.clip(rgb[gt, 1] * (1.0 - alpha) + 1.0 * alpha, 0, 1)
    if pred_mask is None:
        rgb[gt, 0] = rgb[gt, 0] * (1.0 - alpha)
        rgb[gt, 2] = rgb[gt, 2] * (1.0 - alpha)

    return rgb


def visualize_sample_grid(
    dataset: BUSIDataset,
    num_samples: int = 16,
    output_path: str = "outputs/reports/train_samples.png",
    title: str = "Dataset Sample Sanity Inspection"
) -> None:
    """
    Generate a grid of (Image, GT Mask, Overlay) for dataset validation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_samples = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    cols = 4
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        sample = dataset[idx]
        img_np = sample["image"].squeeze(0).numpy()
        mask_np = sample["mask"].squeeze(0).numpy()
        cls_name = sample["class_name"]

        overlay = create_overlay(img_np, mask_np)

        axes[r, c].imshow(overlay)
        axes[r, c].set_title(f"[{cls_name}] idx:{idx}", fontsize=9)
        axes[r, c].axis("off")

    for i in range(num_samples, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Sample grid saved to: {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    
    splits_dir = config["data"]["splits_dir"]
    split_csv = os.path.join(splits_dir, f"{args.split}.csv")
    
    if not os.path.exists(split_csv):
        raise FileNotFoundError(f"Split file not found: {split_csv}. Please run src/split.py first.")

    is_train = (args.split == "train")
    transform = build_transforms(config, is_train=is_train)
    dataset = BUSIDataset(split_csv, transform=transform)

    out_file = args.output or f"outputs/reports/{args.split}_samples.png"
    visualize_sample_grid(
        dataset=dataset,
        num_samples=args.n,
        output_path=out_file,
        title=f"BUSI {args.split.upper()} Set Samples (Image + GT Mask Overlay)"
    )


if __name__ == "__main__":
    main()
