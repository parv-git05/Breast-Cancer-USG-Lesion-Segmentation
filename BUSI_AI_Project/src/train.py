"""
Training and Validation Pipeline for EXP-SEG-01 Plain U-Net Baseline.

Features:
- Config-driven training loop with reproducible seeding.
- Combined BCE + Dice Loss (50/50 weighting).
- Adam optimizer with Cosine Annealing learning rate schedule.
- Validation checkpoint selection based on highest Validation Dice score.
- Early stopping with configurable patience.
- Zero Leakage: Test set is NEVER loaded or evaluated during training.
- Diagnostic artifact generation:
    * Training and validation loss/metric curves.
    * Validation Dice and IoU distribution histograms.
    * Pixel-level confusion matrix.
    * Representative best, median, and failure-case visualization grids.
    * Numerical history CSV/JSON.
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
from src.dataset import BUSIDataset
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
    parser = argparse.ArgumentParser(description="Train baseline Plain U-Net on BUSI dataset.")
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
    Validate on the validation split.
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
    device: torch.device,
) -> pd.DataFrame:
    """
    Perform deep validation evaluation using the best checkpoint:
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
    print(f"[INFO] Validation per-image metrics saved to: {csv_path}")

    # 1. Distribution Plots (Dice and IoU)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df_results["dice"], bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0].axvline(df_results["dice"].mean(), color="red", linestyle="--", label=f"Mean Dice: {df_results['dice'].mean():.4f}")
    axes[0].axvline(df_results["dice"].median(), color="orange", linestyle=":", label=f"Median Dice: {df_results['dice'].median():.4f}")
    axes[0].set_title("Validation Dice Distribution (N=94)", fontsize=12)
    axes[0].set_xlabel("Dice Similarity Coefficient")
    axes[0].set_ylabel("Frequency (Images)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(df_results["iou"], bins=20, color="#2e7d32", edgecolor="black", alpha=0.8)
    axes[1].axvline(df_results["iou"].mean(), color="red", linestyle="--", label=f"Mean IoU: {df_results['iou'].mean():.4f}")
    axes[1].axvline(df_results["iou"].median(), color="orange", linestyle=":", label=f"Median IoU: {df_results['iou'].median():.4f}")
    axes[1].set_title("Validation IoU Distribution (N=94)", fontsize=12)
    axes[1].set_xlabel("Intersection over Union (IoU)")
    axes[1].set_ylabel("Frequency (Images)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    dist_plot_path = os.path.join(report_dir, "val_metric_distributions.png")
    plt.savefig(dist_plot_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Validation distributions plot saved to: {dist_plot_path}")

    # 2. Pixel-Level Confusion Matrix (Clearly labeled as Pixel-level)
    cm_pixels = np.array([[total_tn, total_fp], [total_fn, total_tp]], dtype=np.int64)
    cm_pixels_norm = cm_pixels.astype(float) / cm_pixels.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pixels_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Background (0)", "Lesion (1)"])
    ax.set_yticklabels(["Background (0)", "Lesion (1)"])
    ax.set_xlabel("Predicted Pixel Label", fontsize=11)
    ax.set_ylabel("True Pixel Label", fontsize=11)
    ax.set_title("Pixel-Level Segmentation Confusion Matrix\n(Normalized by True Class)", fontsize=12)

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
    print(f"[INFO] Pixel confusion matrix saved to: {cm_path}")

    # 3. Qualitative Visualizations: Best, Median, and Failure cases
    sorted_indices = df_results.sort_values(by="dice", ascending=False)["index"].tolist()
    best_4 = sorted_indices[:4]
    worst_4 = sorted_indices[-4:]
    mid_start = (len(sorted_indices) // 2) - 2
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

        fig.suptitle(f"{title_prefix} — EXP-SEG-01 Validation", fontsize=14, y=0.99)
        plt.tight_layout()
        save_path = os.path.join(report_dir, save_filename)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Sample grid saved to: {save_path}")

    save_sample_grid(best_4, "Representative Best Cases (Highest Dice)", "val_best_samples.png")
    save_sample_grid(median_4, "Representative Median Cases", "val_median_samples.png")
    save_sample_grid(worst_4, "Representative Failure Cases (Lowest Dice)", "val_worst_failure_samples.png")

    return df_results


def plot_training_curves(history: Dict[str, List], output_path: str):
    """Generate and save clean training and validation metric curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history["epoch"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_title("Training and Validation Loss (BCE + Dice Combined)", fontsize=12)
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
    axes[1].set_title("Validation Segmentation Metrics vs Epoch", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric Score")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Training curves plot saved to: {output_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    # 1. Setup reproducibility & hardware
    seed = config.get("data", {}).get("split_seed", 42)
    set_seed(seed=seed, deterministic=config.get("hardware", {}).get("deterministic", True))
    device = get_device(config.get("hardware", {}).get("device", "auto"))
    print(f"[INFO] Training on Device: {device} | Random Seed: {seed}")

    # Set PyTorch thread count for CPU optimization
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        print(f"[INFO] PyTorch CPU threads set to: {torch.get_num_threads()}")

    # 2. Setup DataLoaders (Train and Val ONLY - Zero Test Leakage)
    splits_dir = config["data"]["splits_dir"]
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"Splits not found in {splits_dir}. Run src/split.py first.")

    train_transform = build_transforms(config, is_train=True)
    val_transform = build_transforms(config, is_train=False)

    train_dataset = BUSIDataset(train_csv, transform=train_transform)
    val_dataset = BUSIDataset(val_csv, transform=val_transform)

    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("hardware", {}).get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[INFO] Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"[INFO] Val samples:   {len(val_dataset)} ({len(val_loader)} batches)")

    # 3. Model, Loss, Optimizer, Scheduler
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

    # Directories
    checkpoint_dir = config.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints/exp_seg_01")
    log_dir = config.get("outputs", {}).get("log_dir", "outputs/logs/exp_seg_01")
    report_dir = config.get("outputs", {}).get("report_dir", "outputs/reports/exp_seg_01")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pt")

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

    print("=" * 80)
    print("STARTING EXPERIMENT: EXP-SEG-01 (Plain U-Net Baseline)")
    print("=" * 80)
    start_total_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.perf_counter() - t0

        # Record history
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
            status_flag = " ⭐ [BEST]"

            # Save best checkpoint
            save_checkpoint(
                state={
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

        # Always save latest checkpoint
        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou,
            },
            checkpoint_path=latest_checkpoint_path,
            config=config,
        )

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.2f}s{status_flag}"
        )

        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING] Validation Dice did not improve for {patience} consecutive epochs. Stopping at epoch {epoch}.")
            break

    total_duration = time.perf_counter() - start_total_time
    mins, secs = divmod(total_duration, 60)
    print("=" * 80)
    print(f"TRAINING COMPLETE in {int(mins)}m {secs:.2f}s")
    print(f"Best Validation Epoch: {best_epoch}")
    print(f"Best Validation Dice:  {best_val_dice:.4f}")
    print(f"Best Validation IoU:   {best_val_iou:.4f}")
    print(f"Best Checkpoint Saved: {best_checkpoint_path}")
    print("=" * 80)

    # 4. Save history artifacts
    df_hist = pd.DataFrame(history)
    hist_csv_path = os.path.join(log_dir, "training_history.csv")
    hist_json_path = os.path.join(log_dir, "training_history.json")
    df_hist.to_csv(hist_csv_path, index=False)
    df_hist.to_csv(os.path.join(report_dir, "training_history.csv"), index=False)
    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 5. Plot training curves
    curves_path = os.path.join(report_dir, "training_curves.png")
    plot_training_curves(history, curves_path)

    # 6. Deep Diagnostic Visualizations using BEST model checkpoint
    print("\n[INFO] Loading best model checkpoint for validation diagnostics...")
    best_checkpoint = load_checkpoint(best_checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_eval_df = evaluate_and_generate_diagnostics(
        model=model,
        val_dataset=val_dataset,
        report_dir=report_dir,
        device=device,
    )

    # 7. Write final summary JSON
    summary_data = {
        "experiment": "EXP-SEG-01",
        "model_architecture": "Plain U-Net (Depth=4, BaseChannels=64)",
        "best_epoch": best_epoch,
        "best_val_dice": float(best_val_dice),
        "best_val_iou": float(best_val_iou),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "total_epochs_trained": len(history["epoch"]),
        "total_training_duration_seconds": float(total_duration),
        "checkpoint_path": best_checkpoint_path,
        "val_dice_distribution": {
            "mean": float(val_eval_df["dice"].mean()),
            "std": float(val_eval_df["dice"].std()),
            "median": float(val_eval_df["dice"].median()),
            "min": float(val_eval_df["dice"].min()),
            "max": float(val_eval_df["dice"].max()),
        },
        "val_iou_distribution": {
            "mean": float(val_eval_df["iou"].mean()),
            "std": float(val_eval_df["iou"].std()),
            "median": float(val_eval_df["iou"].median()),
            "min": float(val_eval_df["iou"].min()),
            "max": float(val_eval_df["iou"].max()),
        },
    }

    summary_json_path = os.path.join(report_dir, "metric_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[INFO] Experiment summary saved to: {summary_json_path}")


if __name__ == "__main__":
    main()
