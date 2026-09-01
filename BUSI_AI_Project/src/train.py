"""
Stratified 3-Fold Cross-Validation Training and Evaluation Pipeline
for EXP-SEG-01 Plain U-Net Baseline.

Features:
- Config-driven 3-fold cross-validation loop with reproducible seeding.
- Fresh U-Net model initialized from scratch for EACH fold.
- Zero model weight leakage between folds.
- Combined BCE + Dice Loss (50/50 weighting).
- Adam optimizer with Cosine Annealing learning rate schedule.
- Validation checkpoint selection based on highest Validation Dice score per fold.
- Early stopping per fold.
- Diagnostic artifact generation PER FOLD:
    * Training and validation loss / metric curves.
    * Validation Dice and IoU distribution histograms.
    * Pixel-level confusion matrix.
    * Representative best, median, and failure-case visualization grids.
    * Numerical history CSV/JSON.
- Aggregate 3-Fold CV Summary Report:
    * Out-of-fold pooled metrics (628 images total): Dice (mean ± std), IoU (mean ± std).
    * Pixel precision, pixel recall/sensitivity, pixel specificity.
    * Fold-wise breakdown of best epoch, training duration, and scores.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import BUSIDataset, get_fold_dataloaders
from src.losses import CombinedBCEDiceLoss
from src.metrics import (
    compute_dice_coefficient,
    compute_iou,
    compute_pixel_metrics,
    evaluate_segmentation_sample,
)
from src.models.unet import get_unet
from src.transforms import build_transforms
from src.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed
from src.visualize import create_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline Plain U-Net using 3-Fold CV on BUSI dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_seg_01.yaml",
        help="Path to experiment YAML configuration file."
    )
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch and return average training loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, num_batches)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Validate on a validation fold split.
    Returns: (val_loss, val_dice_mean, val_iou_mean)
    """
    model.eval()
    total_loss = 0.0
    dice_scores: List[float] = []
    iou_scores: List[float] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            gt_masks = masks.cpu().numpy()

            for i in range(len(images)):
                pred_bin = (probs[i, 0] > 0.5).astype(np.float32)
                gt_bin = gt_masks[i, 0]

                dice = compute_dice_coefficient(pred_bin, gt_bin)
                iou = compute_iou(pred_bin, gt_bin)

                dice_scores.append(dice)
                iou_scores.append(iou)

    mean_loss = total_loss / max(1, len(loader))
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return mean_loss, mean_dice, mean_iou


def evaluate_and_generate_diagnostics(
    model: nn.Module,
    val_dataset: BUSIDataset,
    report_dir: str,
    fold_idx: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Perform deep validation evaluation using the best checkpoint of a fold:
    - Computes per-image Dice, IoU, Precision, Recall, Specificity, and pixel confusion counts.
    - Generates distribution histograms, pixel confusion matrix, and visual grids.
    """
    os.makedirs(report_dir, exist_ok=True)
    model.eval()

    val_records = []
    val_preds = []
    val_gts = []
    val_imgs = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            img_t = sample["image"].unsqueeze(0).to(device)  # (1, 1, H, W)
            gt_t = sample["mask"].numpy()                   # (1, H, W)

            logits = model(img_t)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (1, H, W)
            pred_bin = (probs[0] > 0.5).astype(np.float32)
            gt_bin = gt_t[0]

            metrics = evaluate_segmentation_sample(pred_bin, gt_bin)
            pix_stats = compute_pixel_metrics(pred_bin, gt_bin)

            total_tp += pix_stats["tp"]
            total_fp += pix_stats["fp"]
            total_fn += pix_stats["fn"]
            total_tn += pix_stats["tn"]

            record = {
                "fold": fold_idx,
                "index": i,
                "image_path": sample["image_path"],
                "class_name": sample["class_name"],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "tp_pixels": pix_stats["tp"],
                "fp_pixels": pix_stats["fp"],
                "fn_pixels": pix_stats["fn"],
                "tn_pixels": pix_stats["tn"],
            }
            val_records.append(record)

            val_imgs.append(sample["image"].squeeze(0).numpy())
            val_gts.append(gt_bin)
            val_preds.append(pred_bin)

    df_results = pd.DataFrame(val_records)
    csv_path = os.path.join(report_dir, "val_predictions.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[INFO] Fold {fold_idx} per-image validation metrics saved to: {csv_path}")

    # 1. Distribution Plots (Dice and IoU)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df_results["dice"], bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0].axvline(df_results["dice"].mean(), color="red", linestyle="--", label=f"Mean Dice: {df_results['dice'].mean():.4f}")
    axes[0].axvline(df_results["dice"].median(), color="orange", linestyle=":", label=f"Median Dice: {df_results['dice'].median():.4f}")
    axes[0].set_title(f"Fold {fold_idx} Validation Dice Distribution (N={len(df_results)})", fontsize=12)
    axes[0].set_xlabel("Dice Similarity Coefficient")
    axes[0].set_ylabel("Frequency (Images)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(df_results["iou"], bins=20, color="#2e7d32", edgecolor="black", alpha=0.8)
    axes[1].axvline(df_results["iou"].mean(), color="red", linestyle="--", label=f"Mean IoU: {df_results['iou'].mean():.4f}")
    axes[1].axvline(df_results["iou"].median(), color="orange", linestyle=":", label=f"Median IoU: {df_results['iou'].median():.4f}")
    axes[1].set_title(f"Fold {fold_idx} Validation IoU Distribution (N={len(df_results)})", fontsize=12)
    axes[1].set_xlabel("Intersection over Union (IoU)")
    axes[1].set_ylabel("Frequency (Images)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    dist_plot_path = os.path.join(report_dir, "val_metric_distributions.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close(fig)

    # 2. Pixel-Level Confusion Matrix
    cm_pixels = np.array([[total_tn, total_fp], [total_fn, total_tp]], dtype=np.int64)
    cm_pixels_norm = cm_pixels.astype(float) / max(1.0, float(cm_pixels.sum(axis=1, keepdims=True).min()))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pixels_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Background (0)", "Lesion (1)"])
    ax.set_yticklabels(["Background (0)", "Lesion (1)"])
    ax.set_xlabel("Predicted Pixel Label", fontsize=11)
    ax.set_ylabel("True Pixel Label", fontsize=11)
    ax.set_title(f"Fold {fold_idx} Pixel-Level Segmentation Confusion Matrix\n(Normalized by True Class)", fontsize=12)

    for r in range(2):
        for c in range(2):
            val_pct = cm_pixels_norm[r, c] * 100
            count_str = f"{cm_pixels[r, c]:,}"
            color = "white" if val_pct > 50 else "black"
            ax.text(c, r, f"{val_pct:.2f}%\n({count_str} px)", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = os.path.join(report_dir, "val_pixel_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # 3. Qualitative Visualizations: Best, Median, and Failure cases
    sorted_indices = df_results.sort_values(by="dice", ascending=False)["index"].tolist()
    best_4 = sorted_indices[:4]
    worst_4 = sorted_indices[-4:]
    mid_start = max(0, (len(sorted_indices) // 2) - 2)
    median_4 = sorted_indices[mid_start : mid_start + 4]

    def save_sample_grid(indices: List[int], title_prefix: str, save_filename: str):
        fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))
        if len(indices) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, sample_idx in enumerate(indices):
            rec = df_results[df_results["index"] == sample_idx].iloc[0]
            img = val_imgs[sample_idx]
            gt = val_gts[sample_idx]
            pred = val_preds[sample_idx]
            overlay = create_overlay(img, gt, pred)

            axes[row_idx, 0].imshow(img, cmap="gray")
            axes[row_idx, 0].set_title(f"Original [{rec['class_name']}]\nidx:{sample_idx}", fontsize=10)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(gt, cmap="gray")
            axes[row_idx, 1].set_title("Ground Truth Mask", fontsize=10)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(pred, cmap="gray")
            axes[row_idx, 2].set_title("U-Net Prediction", fontsize=10)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(overlay)
            axes[row_idx, 3].set_title(f"Overlay (Green:GT, Red:Pred)\nDice={rec['dice']:.4f} | IoU={rec['iou']:.4f}", fontsize=10)
            axes[row_idx, 3].axis("off")

        fig.suptitle(f"{title_prefix} — Fold {fold_idx} Validation", fontsize=14, y=0.99)
        plt.tight_layout()
        save_path = os.path.join(report_dir, save_filename)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    save_sample_grid(best_4, f"Fold {fold_idx} Best Cases (Highest Dice)", "val_best_samples.png")
    save_sample_grid(median_4, f"Fold {fold_idx} Median Cases", "val_median_samples.png")
    save_sample_grid(worst_4, f"Fold {fold_idx} Representative Failure Cases (Lowest Dice)", "val_worst_failure_samples.png")

    return df_results


def plot_training_curves(history: Dict[str, List], output_path: str, fold_idx: int):
    """Generate and save training/validation loss and metric curves per fold."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["epoch"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_title(f"Fold {fold_idx} Training & Validation Loss (BCE + Dice)", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Metric Curves
    axes[1].plot(epochs, history["val_dice"], label="Val Dice (DSC)", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, history["val_iou"], label="Val IoU (Jaccard)", color="#ff7f0e", linewidth=2, linestyle="--")
    best_idx = int(np.argmax(history["val_dice"]))
    best_dice = history["val_dice"][best_idx]
    axes[1].scatter([best_idx + 1], [best_dice], color="red", s=80, zorder=5, label=f"Best Val Dice: {best_dice:.4f} (Ep {best_idx + 1})")
    axes[1].set_title(f"Fold {fold_idx} Validation Segmentation Metrics vs Epoch", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def train_single_fold(
    fold_idx: int,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Train a FRESH Plain U-Net model from scratch on fold_idx.
    Returns: (val_eval_df, fold_summary_dict)
    """
    seed = config.get("data", {}).get("seed", 42) + fold_idx  # Seed variant for exact reproducibility
    set_seed(seed=seed, deterministic=config.get("hardware", {}).get("deterministic", True))

    print("-" * 80)
    print(f"STARTING TRAINING: FOLD {fold_idx} / 3 (Seed: {seed})")
    print("-" * 80)

    # DataLoaders for this fold
    train_loader, val_loader = get_fold_dataloaders(config, fold_idx=fold_idx)
    val_dataset = val_loader.dataset

    print(f"[INFO] Fold {fold_idx} Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"[INFO] Fold {fold_idx} Val samples:   {len(val_loader.dataset)} ({len(val_loader)} batches)")

    # Model initialization: FRESH MODEL FROM SCRATCH (Zero weight carryover)
    model = get_unet(config).to(device)

    loss_cfg = config.get("training", {})
    bce_weight = loss_cfg.get("bce_weight", 0.5)
    dice_weight = loss_cfg.get("dice_weight", 0.5)
    criterion = CombinedBCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

    lr = float(loss_cfg.get("lr", 1e-3))
    epochs = int(loss_cfg.get("epochs", 100))
    patience = int(loss_cfg.get("early_stopping_patience", 20))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Directories per fold
    base_chk_dir = config.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints/exp_seg_01")
    base_log_dir = config.get("outputs", {}).get("log_dir", "outputs/logs/exp_seg_01")
    base_rep_dir = config.get("outputs", {}).get("report_dir", "outputs/reports/exp_seg_01")

    fold_chk_dir = os.path.join(base_chk_dir, f"fold_{fold_idx}")
    fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_idx}")
    fold_rep_dir = os.path.join(base_rep_dir, f"fold_{fold_idx}")

    os.makedirs(fold_chk_dir, exist_ok=True)
    os.makedirs(fold_log_dir, exist_ok=True)
    os.makedirs(fold_rep_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(fold_chk_dir, "best_model.pt")

    history: Dict[str, List] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "lr": [],
        "epoch_time_sec": []
    }

    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.perf_counter() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        is_best = val_dice > best_val_dice
        status_flag = ""

        if is_best:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_epoch = epoch
            epochs_no_improve = 0
            status_flag = " * [BEST]"

            save_checkpoint(
                state={
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                },
                checkpoint_path=best_checkpoint_path,
                config=config,
            )
        else:
            epochs_no_improve += 1

        print(
            f"Fold {fold_idx} | Ep [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.2f}s{status_flag}",
            flush=True
        )

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING FOLD {fold_idx}] Validation Dice did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.")
            break

    total_duration = time.perf_counter() - t_start

    # Save training logs & plots per fold
    df_hist = pd.DataFrame(history)
    hist_csv_path = os.path.join(fold_log_dir, "training_history.csv")
    df_hist.to_csv(hist_csv_path, index=False)

    curves_path = os.path.join(fold_rep_dir, "training_curves.png")
    plot_training_curves(history, curves_path, fold_idx=fold_idx)

    # Load BEST model checkpoint for diagnostics
    best_checkpoint = load_checkpoint(best_checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_eval_df = evaluate_and_generate_diagnostics(
        model=model,
        val_dataset=val_dataset,
        report_dir=fold_rep_dir,
        fold_idx=fold_idx,
        device=device,
    )

    fold_summary = {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_val_iou),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "total_epochs_trained": len(history["epoch"]),
        "training_duration_seconds": float(total_duration),
        "checkpoint_path": best_checkpoint_path,
        "val_dice_mean": float(val_eval_df["dice"].mean()),
        "val_dice_std": float(val_eval_df["dice"].std()),
        "val_iou_mean": float(val_eval_df["iou"].mean()),
        "val_iou_std": float(val_eval_df["iou"].std()),
    }

    with open(os.path.join(fold_rep_dir, "metric_summary.json"), "w", encoding="utf-8") as f:
        json.dump(fold_summary, f, indent=2)

    return val_eval_df, fold_summary


def main():
    args = parse_args()
    config = load_config(args.config)

    device = get_device(config.get("hardware", {}).get("device", "auto"))
    print("=" * 80)
    print("STARTING STRATIFIED 3-FOLD CV EXPERIMENT: EXP-SEG-01 (Plain U-Net Baseline)")
    print(f"Device: {device}")
    print("=" * 80)

    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        print(f"[INFO] PyTorch CPU threads set to: {torch.get_num_threads()}")

    all_val_records: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, Any]] = []

    total_cv_start_time = time.perf_counter()

    for fold_idx in range(1, 4):
        val_df, summary = train_single_fold(fold_idx=fold_idx, config=config, device=device)
        all_val_records.append(val_df)
        fold_summaries.append(summary)

    total_cv_duration = time.perf_counter() - total_cv_start_time

    # Combine out-of-fold validation predictions (N=628)
    combined_val_df = pd.concat(all_val_records, ignore_index=True)

    base_rep_dir = config.get("outputs", {}).get("report_dir", "outputs/reports/exp_seg_01")
    os.makedirs(base_rep_dir, exist_ok=True)

    combined_csv_path = os.path.join(base_rep_dir, "cv_out_of_fold_predictions.csv")
    combined_val_df.to_csv(combined_csv_path, index=False)

    # Compute overall pooled metrics across all 628 out-of-fold predictions
    overall_dice_mean = float(combined_val_df["dice"].mean())
    overall_dice_std = float(combined_val_df["dice"].std())
    overall_iou_mean = float(combined_val_df["iou"].mean())
    overall_iou_std = float(combined_val_df["iou"].std())

    total_tp = int(combined_val_df["tp_pixels"].sum())
    total_fp = int(combined_val_df["fp_pixels"].sum())
    total_fn = int(combined_val_df["fn_pixels"].sum())
    total_tn = int(combined_val_df["tn_pixels"].sum())

    pixel_precision = total_tp / max(1, (total_tp + total_fp))
    pixel_recall = total_tp / max(1, (total_tp + total_fn))
    pixel_specificity = total_tn / max(1, (total_tn + total_fp))

    cv_summary_data = {
        "experiment": "EXP-SEG-01",
        "model_architecture": "Plain U-Net (Depth=4, BaseChannels=64)",
        "protocol": "3-Fold Stratified Image-Level Cross-Validation (Not patient-wise)",
        "total_images_evaluated": len(combined_val_df),
        "overall_metrics": {
            "dice_mean": overall_dice_mean,
            "dice_std": overall_dice_std,
            "iou_mean": overall_iou_mean,
            "iou_std": overall_iou_std,
            "pixel_precision": float(pixel_precision),
            "pixel_recall_sensitivity": float(pixel_recall),
            "pixel_specificity": float(pixel_specificity),
            "total_tp_pixels": total_tp,
            "total_fp_pixels": total_fp,
            "total_fn_pixels": total_fn,
            "total_tn_pixels": total_tn,
        },
        "fold_summaries": fold_summaries,
        "total_cv_duration_seconds": float(total_cv_duration),
    }

    summary_json_path = os.path.join(base_rep_dir, "cv_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary_data, f, indent=2)

    # Generate Markdown Summary Report
    md_report_path = os.path.join(base_rep_dir, "cv_summary_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# EXP-SEG-01 Plain U-Net: Stratified 3-Fold Cross-Validation Official Report\n\n")
        f.write("## Overview & Protocol\n")
        f.write("- **Model**: Plain U-Net (Depth=4, BaseChannels=64)\n")
        f.write("- **Evaluation Protocol**: 3-Fold Stratified Image-Level Cross-Validation (Not patient-wise)\n")
        f.write(f"- **Total Evaluated Out-of-Fold Images**: {len(combined_val_df)} (420 benign, 208 malignant)\n")
        f.write(f"- **Total Training Duration**: {total_cv_duration / 60:.2f} minutes\n\n")
        f.write("## Primary Out-of-Fold Segmentation Performance\n\n")
        f.write("| Metric | Mean ± Std / Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Dice Coefficient (DSC)** | **{overall_dice_mean:.4f} ± {overall_dice_std:.4f}** |\n")
        f.write(f"| **Intersection over Union (IoU)** | **{overall_iou_mean:.4f} ± {overall_iou_std:.4f}** |\n")
        f.write(f"| **Pixel Precision** | {pixel_precision:.4f} |\n")
        f.write(f"| **Pixel Recall (Sensitivity)** | {pixel_recall:.4f} |\n")
        f.write(f"| **Pixel Specificity** | {pixel_specificity:.4f} |\n\n")
        f.write("## Fold-Wise Results Breakdown\n\n")
        f.write("| Fold | Val Images | Best Epoch | Val Dice (Mean ± Std) | Val IoU (Mean ± Std) | Duration (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for s in fold_summaries:
            f.write(f"| Fold {s['fold']} | {s['fold']} | Ep {s['best_epoch']} | {s['val_dice_mean']:.4f} ± {s['val_dice_std']:.4f} | {s['val_iou_mean']:.4f} ± {s['val_iou_std']:.4f} | {s['training_duration_seconds']:.1f}s |\n")

    mins, secs = divmod(total_cv_duration, 60)
    print("\n" + "=" * 80)
    print("STRATIFIED 3-FOLD CROSS-VALIDATION COMPLETE")
    print(f"Total Duration: {int(mins)}m {secs:.2f}s")
    print("-" * 80)
    print(f"Overall Dice (DSC): {overall_dice_mean:.4f} ± {overall_dice_std:.4f}")
    print(f"Overall IoU:        {overall_iou_mean:.4f} ± {overall_iou_std:.4f}")
    print(f"Pixel Precision:    {pixel_precision:.4f}")
    print(f"Pixel Recall:       {pixel_recall:.4f}")
    print(f"Pixel Specificity:  {pixel_specificity:.4f}")
    print("-" * 80)
    print("Fold-wise Breakdown:")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}: Best Ep {s['best_epoch']} | Dice: {s['val_dice_mean']:.4f} ± {s['val_dice_std']:.4f} | IoU: {s['val_iou_mean']:.4f} ± {s['val_iou_std']:.4f}")
    print("=" * 80)
    print(f"Summary JSON saved to: {summary_json_path}")
    print(f"Summary MD report saved to: {md_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
