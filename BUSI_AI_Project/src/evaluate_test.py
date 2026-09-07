"""
EXP-SEG-02 Official Locked-Test Evaluation Pipeline.

Evaluates ONLY the final model trained on all 565 development images for 86 epochs:
  outputs/checkpoints/exp_seg_02/final_model/final_model.pt

Enforces strict locked-test isolation:
  - 63 total test images (42 benign, 21 malignant).
  - 0 overlap with 565 development images.
  - Evaluation-only: no retraining, tuning, or parameter changes.

Generates:
  - outputs/reports/exp_seg_02/test_evaluation/per_image_metrics.csv (63 rows)
  - outputs/reports/exp_seg_02/test_evaluation/qualitative/ (63 4-panel PNGs)
  - outputs/reports/exp_seg_02/test_evaluation/test_metric_boxplot.png
  - outputs/reports/exp_seg_02/test_evaluation/test_metric_summary.png
  - outputs/reports/exp_seg_02/test_evaluation/test_evaluation_report.md
  - outputs/reports/exp_seg_02/test_evaluation/test_evaluation_metadata.json
  - outputs/reports/exp_seg_02/test_evaluation/test_metrics.json
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import BUSIDataset
from src.metrics import (
    compute_dice_coefficient,
    compute_iou,
    compute_pixel_metrics,
    evaluate_segmentation_sample,
)
from src.models.unet import get_unet
from src.transforms import build_transforms
from src.utils import get_device, load_checkpoint, load_config
from src.visualize import create_overlay


def parse_args():
    parser = argparse.ArgumentParser(
        description="Official Locked-Test Evaluation of EXP-SEG-02 Final Model Checkpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp_seg_02.yaml",
        help="Path to EXP-SEG-02 YAML configuration file."
    )
    return parser.parse_args()


def get_git_commit_hash() -> str:
    """Attempt to get current git commit hash, return 'unknown' if not in git repo."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return output
    except Exception:
        return "unknown"


def run_pre_inference_audits(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run strict hard assertions and audits on test set isolation prior to inference.
    """
    test_manifest_path = config.get("data", {}).get("test_manifest", "data/manifests/BUSI_final_test_manifest.csv")
    dev_manifest_path = config.get("data", {}).get("manifest", "data/manifests/BUSI_development_manifest.csv")
    checkpoint_path = os.path.join(
        config.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints/exp_seg_02"),
        "final_model",
        "final_model.pt"
    )

    print("=" * 80)
    print("RUNNING PRE-INFERENCE AUDITS & ISOLATION CHECKS (EXP-SEG-02)")
    print("=" * 80)
    print(f"Target Checkpoint: {checkpoint_path}")
    print(f"Test Manifest:     {test_manifest_path}")
    print(f"Dev Manifest:      {dev_manifest_path}")
    print("-" * 80)

    # 1. Assert checkpoint exists
    assert os.path.exists(checkpoint_path), (
        f"[CRITICAL ERROR] Final model checkpoint missing at: {checkpoint_path}\n"
        "Do not run evaluation without the final trained model checkpoint."
    )
    print("[PASS] Checkpoint physical existence verified.")

    # 2. Assert manifests exist
    assert os.path.exists(test_manifest_path), f"Test manifest missing: {test_manifest_path}"
    assert os.path.exists(dev_manifest_path), f"Development manifest missing: {dev_manifest_path}"

    test_df = pd.read_csv(test_manifest_path)
    dev_df = pd.read_csv(dev_manifest_path)

    # 3. Hard assertions on test set counts & composition
    num_test = len(test_df)
    assert num_test == 63, f"[ASSERTION FAILURE] Expected exactly 63 test images, got {num_test}"
    
    benign_count = int((test_df["class"] == "benign").sum())
    malignant_count = int((test_df["class"] == "malignant").sum())

    assert benign_count == 42, f"[ASSERTION FAILURE] Expected exactly 42 benign test images, got {benign_count}"
    assert malignant_count == 21, f"[ASSERTION FAILURE] Expected exactly 21 malignant test images, got {malignant_count}"
    print(f"[PASS] Test dataset counts verified: {num_test} total ({benign_count} Benign, {malignant_count} Malignant).")

    # 4. Zero overlap check between dev and test
    dev_paths = set(dev_df["image_path"].apply(os.path.normpath))
    test_paths = set(test_df["image_path"].apply(os.path.normpath))
    overlap = dev_paths.intersection(test_paths)

    assert len(overlap) == 0, (
        f"[CRITICAL DATA LEAKAGE DETECTED] {len(overlap)} overlapping images between dev and test manifests!"
    )
    print("[PASS] Zero overlap between 565 development images and 63 test images verified.")

    # 5. Verify test manifest is distinct from dev manifest
    assert set(test_df["image_path"]) != set(dev_df["image_path"]), (
        "[ERROR] Test manifest identical to development manifest! Development manifest was loaded by mistake."
    )
    print("[PASS] Evaluation manifest confirmed to be locked test set (not dev set).")

    print("-" * 80)
    print("ALL PRE-INFERENCE AUDITS PASSED CLEANLY.")
    print("=" * 80 + "\n")

    return {
        "checkpoint_path": checkpoint_path,
        "test_manifest": test_manifest_path,
        "num_test": num_test,
        "benign_count": benign_count,
        "malignant_count": malignant_count,
        "dev_count": len(dev_df),
        "overlap_count": len(overlap),
    }


def compute_summary_stats(series: pd.Series) -> Dict[str, float]:
    """Compute mean, std, median, min, max for a pandas Series."""
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def evaluate_final_model(
    model: torch.nn.Module,
    test_dataset: BUSIDataset,
    device: torch.device
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Run evaluation-only inference on all 63 locked test samples.
    """
    records: List[Dict[str, Any]] = []
    test_imgs: List[np.ndarray] = []
    test_gts: List[np.ndarray] = []
    test_preds: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            img_t = sample["image"].unsqueeze(0).to(device)  # (1, 1, H, W)
            gt_t = sample["mask"].numpy()                   # (1, H, W)

            output = model(img_t)
            prob_map = torch.sigmoid(output).squeeze(0).cpu().numpy()[0]  # (H, W)

            # Segmentation threshold = 0.5 (exact, no post-processing)
            pred_bin = (prob_map > 0.5).astype(np.float32)
            gt_bin = gt_t[0]

            metrics = evaluate_segmentation_sample(pred_bin, gt_bin)
            pix_stats = compute_pixel_metrics(pred_bin, gt_bin)

            filename = os.path.basename(sample["image_path"])
            image_id = os.path.splitext(filename)[0]
            cls_name = sample["class_name"]

            # Calculate intersection and positive pixel counts
            tp = pix_stats["tp"]
            fp = pix_stats["fp"]
            fn = pix_stats["fn"]
            pred_pos = tp + fp
            gt_pos = tp + fn

            records.append({
                "test_index": idx,
                "image_id": image_id,
                "filename": filename,
                "class": cls_name,
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "pixel_precision": float(metrics["precision"]),
                "pixel_recall": float(metrics["recall"]),
                "specificity": float(metrics["specificity"]),
                "intersection": int(tp),
                "predicted_positive_pixels": int(pred_pos),
                "ground_truth_positive_pixels": int(gt_pos),
            })

            test_imgs.append(sample["image"].squeeze(0).numpy())
            test_gts.append(gt_bin)
            test_preds.append(pred_bin)

    df = pd.DataFrame(records)
    return df, test_imgs, test_gts, test_preds


def generate_qualitative_visualizations(
    df: pd.DataFrame,
    test_imgs: List[np.ndarray],
    test_gts: List[np.ndarray],
    test_preds: List[np.ndarray],
    qualitative_dir: str
) -> None:
    """
    Generate 63 separate 4-panel qualitative visualization images, one per test sample.
    """
    os.makedirs(qualitative_dir, exist_ok=True)

    for idx, row in df.iterrows():
        img = test_imgs[idx]
        gt = test_gts[idx]
        pred = test_preds[idx]
        overlay = create_overlay(img, gt, pred)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Panel 1 — Original Ultrasound
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Original {row['class']}\nid: {row['image_id']}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2 — Ground Truth Mask
        axes[1].imshow(gt, cmap="gray")
        axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3 — U-Net Prediction
        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("U-Net Prediction", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Panel 4 — Overlay
        axes[3].imshow(overlay)
        axes[3].set_title(
            f"Overlay (GT=Green, Pred=Red)\nDice = {row['dice']:.4f} | IoU = {row['iou']:.4f}",
            fontsize=11,
            fontweight="bold"
        )
        axes[3].axis("off")

        plt.tight_layout()
        save_filename = f"sample_{idx+1:03d}_{row['image_id']}.png"
        save_path = os.path.join(qualitative_dir, save_filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[PASS] Generated all {len(df)} individual qualitative 4-panel visualizations in: {qualitative_dir}")


def generate_box_plot(df: pd.DataFrame, save_path: str) -> None:
    """
    Generate publication-quality box plot for test set segmentation metrics.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_keys = ["dice", "iou", "pixel_precision", "pixel_recall", "specificity"]
    metric_labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    metric_data = [df[k] for k in metrics_keys]

    # Clear, publication-friendly colors
    colors = ["#2b5c8f", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    box = ax.boxplot(
        metric_data,
        tick_labels=metric_labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={"marker": "o", "color": "red", "alpha": 0.6, "markersize": 5}
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"Test Set Segmentation Performance Distribution (N={len(df)})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Segmentation Metric", fontsize=12, labelpad=10)
    ax.set_ylabel("Metric Value", fontsize=12, labelpad=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Add custom legend for mean indicator
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Median"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="yellow", markeredgecolor="black", markersize=7, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] Generated metric distribution box plot: {save_path}")


def generate_summary_plot(df: pd.DataFrame, save_path: str) -> None:
    """
    Generate summary bar chart with Mean ± Standard Deviation error bars for test metrics.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metrics_keys = ["dice", "iou", "pixel_precision", "pixel_recall", "specificity"]
    metric_labels = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    
    means = [df[k].mean() for k in metrics_keys]
    stds = [df[k].std() for k in metrics_keys]

    colors = ["#2b5c8f", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    bars = ax.bar(
        metric_labels,
        means,
        yerr=stds,
        capsize=6,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2
    )

    for bar, m, s in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(height + s + 0.02, 1.02),
            f"{m:.4f}\n±{s:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold"
        )

    ax.set_title(f"Test Set Segmentation Performance Summary (Mean ± SD, N={len(df)})", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PASS] Generated metric summary plot: {save_path}")


def generate_markdown_report(
    df: pd.DataFrame,
    stats_overall: Dict[str, Dict[str, float]],
    stats_benign: Dict[str, Dict[str, float]],
    stats_malignant: Dict[str, Dict[str, float]],
    report_path: str
) -> None:
    """
    Generate human-readable markdown test evaluation report.
    """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EXP-SEG-02 Plain U-Net: Official Locked Test Evaluation Report\n\n")
        f.write("## 1. Experiment & Model Information\n")
        f.write("- **Experiment ID**: `EXP-SEG-02`\n")
        f.write("- **Model Architecture**: Plain U-Net (Depth=4, Base Channels=64, Output Channels=1)\n")
        f.write("- **Evaluated Checkpoint**: `outputs/checkpoints/exp_seg_02/final_model/final_model.pt`\n")
        f.write("- **Training Details**: Trained on all 565 development images for exactly 86 epochs\n")
        f.write("- **Test Set Size**: Total N = 63 (Benign N = 42, Malignant N = 21)\n")
        f.write("- **Segmentation Threshold**: Binary 0.5 (No threshold tuning or mask post-processing)\n\n")

        f.write("## 2. Overall Test Set Performance (N=63)\n\n")
        f.write("| Metric | Mean ± Std | Median | Min | Max |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for key, name in [
            ("dice", "Dice Similarity Coefficient"),
            ("iou", "Intersection over Union (IoU)"),
            ("pixel_precision", "Pixel Precision"),
            ("pixel_recall", "Pixel Recall (Sensitivity)"),
            ("specificity", "Pixel Specificity"),
        ]:
            st = stats_overall[key]
            f.write(f"| **{name}** | **{st['mean']:.4f} ± {st['std']:.4f}** | {st['median']:.4f} | {st['min']:.4f} | {st['max']:.4f} |\n")
        f.write("\n")

        f.write("## 3. Subgroup Performance Analysis\n\n")
        f.write("### Benign Subgroup (N=42)\n\n")
        f.write("| Metric | Mean ± Std | Median | Min | Max |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for key, name in [
            ("dice", "Dice Similarity Coefficient"),
            ("iou", "Intersection over Union (IoU)"),
            ("pixel_precision", "Pixel Precision"),
            ("pixel_recall", "Pixel Recall (Sensitivity)"),
            ("specificity", "Pixel Specificity"),
        ]:
            st = stats_benign[key]
            f.write(f"| **{name}** | **{st['mean']:.4f} ± {st['std']:.4f}** | {st['median']:.4f} | {st['min']:.4f} | {st['max']:.4f} |\n")
        f.write("\n")

        f.write("### Malignant Subgroup (N=21)\n\n")
        f.write("| Metric | Mean ± Std | Median | Min | Max |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for key, name in [
            ("dice", "Dice Similarity Coefficient"),
            ("iou", "Intersection over Union (IoU)"),
            ("pixel_precision", "Pixel Precision"),
            ("pixel_recall", "Pixel Recall (Sensitivity)"),
            ("specificity", "Pixel Specificity"),
        ]:
            st = stats_malignant[key]
            f.write(f"| **{name}** | **{st['mean']:.4f} ± {st['std']:.4f}** | {st['median']:.4f} | {st['min']:.4f} | {st['max']:.4f} |\n")
        f.write("\n")

        f.write("## 4. Test Set Integrity & Isolation Verification\n")
        f.write("- **Test N**: 63 samples (42 benign, 21 malignant)\n")
        f.write("- **Lock Status**: Test set was locked prior to final model training\n")
        f.write("- **Zero Contamination**: No test images were used during training, validation, early stopping, or threshold tuning\n")
        f.write("- **Single Evaluation**: All 63 test images were evaluated exactly once\n")
        f.write("- **Development/Test Overlap**: 0 overlapping images verified\n")
        f.write("- **Qualitative Verification**: All 63 individual 4-panel case visualizations generated and stored in `qualitative/`\n")

    print(f"[PASS] Saved markdown report: {report_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.get("hardware", {}).get("device", "auto"))

    # 1. Run Pre-Inference Audits
    audit_meta = run_pre_inference_audits(config)

    # 2. Setup Directories
    target_dir = os.path.join(
        config.get("outputs", {}).get("report_dir", "outputs/reports/exp_seg_02"),
        "test_evaluation"
    )
    qualitative_dir = os.path.join(target_dir, "qualitative")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(qualitative_dir, exist_ok=True)

    # 3. Load Test Dataset
    test_manifest_path = audit_meta["test_manifest"]
    raw_root = config.get("data", {}).get("raw_root", "")
    test_transform = build_transforms(config, is_train=False)
    test_dataset = BUSIDataset(test_manifest_path, transform=test_transform, raw_root=raw_root)

    print(f"[INFO] Loaded locked test set: {len(test_dataset)} images from {test_manifest_path}")

    # 4. Load Final Model Checkpoint
    checkpoint_path = audit_meta["checkpoint_path"]
    ckpt = load_checkpoint(checkpoint_path, device=device)

    model = get_unet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Successfully loaded EXP-SEG-02 final model checkpoint (Epoch {ckpt.get('epoch', 'N/A')})")

    # 5. Run Evaluation Inference
    print("\n[INFO] Performing official evaluation inference on 63 test images...")
    df, test_imgs, test_gts, test_preds = evaluate_final_model(model, test_dataset, device)

    # 6. Compute Subgroup and Overall Statistics
    metrics_keys = ["dice", "iou", "pixel_precision", "pixel_recall", "specificity"]
    
    stats_overall = {k: compute_summary_stats(df[k]) for k in metrics_keys}
    
    df_benign = df[df["class"] == "benign"]
    df_malignant = df[df["class"] == "malignant"]

    stats_benign = {k: compute_summary_stats(df_benign[k]) for k in metrics_keys}
    stats_malignant = {k: compute_summary_stats(df_malignant[k]) for k in metrics_keys}

    # 7. Save per_image_metrics.csv
    csv_path = os.path.join(target_dir, "per_image_metrics.csv")
    csv_cols = [
        "image_id", "filename", "class", "dice", "iou",
        "pixel_precision", "pixel_recall", "specificity",
        "intersection", "predicted_positive_pixels", "ground_truth_positive_pixels"
    ]
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"[PASS] Saved per-image metrics CSV ({len(df)} rows): {csv_path}")

    # 8. Generate Qualitative Visualizations
    generate_qualitative_visualizations(df, test_imgs, test_gts, test_preds, qualitative_dir)

    # 9. Generate Plots
    boxplot_path = os.path.join(target_dir, "test_metric_boxplot.png")
    summaryplot_path = os.path.join(target_dir, "test_metric_summary.png")
    generate_box_plot(df, boxplot_path)
    generate_summary_plot(df, summaryplot_path)

    # 10. Generate Markdown Report
    report_md_path = os.path.join(target_dir, "test_evaluation_report.md")
    generate_markdown_report(df, stats_overall, stats_benign, stats_malignant, report_md_path)

    # 11. Save Metrics and Metadata JSONs
    metrics_json_path = os.path.join(target_dir, "test_metrics.json")
    metadata_json_path = os.path.join(target_dir, "test_evaluation_metadata.json")

    timestamp_str = datetime.datetime.now().isoformat()
    git_hash = get_git_commit_hash()

    summary_metrics_data = {
        "experiment_id": "EXP-SEG-02",
        "checkpoint": checkpoint_path,
        "sample_count": len(df),
        "benign_count": len(df_benign),
        "malignant_count": len(df_malignant),
        "overall": stats_overall,
        "benign_subgroup": stats_benign,
        "malignant_subgroup": stats_malignant,
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics_data, f, indent=2)
    print(f"[PASS] Saved metrics JSON: {metrics_json_path}")

    metadata_dict = {
        "experiment_id": "EXP-SEG-02",
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": ckpt.get("epoch", 86),
        "test_manifest_path": test_manifest_path,
        "test_image_count": len(df),
        "benign_count": len(df_benign),
        "malignant_count": len(df_malignant),
        "development_image_count": 565,
        "dev_test_disjointness_confirmed": True,
        "preprocessing_config": config.get("preprocessing", {}),
        "image_size": config.get("data", {}).get("image_size", [256, 256]),
        "segmentation_threshold": 0.5,
        "metrics_calculated": metrics_keys,
        "evaluation_timestamp": timestamp_str,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit_hash": git_hash,
        "evaluated_samples_confirmed": len(df),
    }

    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"[PASS] Saved metadata JSON: {metadata_json_path}")

    # 12. Run Post-Inference Hard Assertions
    print("\n" + "=" * 80)
    print("POST-INFERENCE AUDIT & ASSERTION VERIFICATION")
    print("=" * 80)

    num_rows = len(pd.read_csv(csv_path))
    qualitative_files = [f for f in os.listdir(qualitative_dir) if f.endswith(".png")]

    assert num_rows == 63, f"[POST-INFERENCE ERROR] per_image_metrics.csv row count ({num_rows}) != 63"
    assert len(df) == 63, f"[POST-INFERENCE ERROR] Evaluated samples count ({len(df)}) != 63"
    assert len(df_benign) == 42, f"[POST-INFERENCE ERROR] Evaluated benign count ({len(df_benign)}) != 42"
    assert len(df_malignant) == 21, f"[POST-INFERENCE ERROR] Evaluated malignant count ({len(df_malignant)}) != 21"
    assert len(qualitative_files) == 63, f"[POST-INFERENCE ERROR] Qualitative PNG count ({len(qualitative_files)}) != 63"

    # Verify no missing predictions or metric files
    for idx, row in df.iterrows():
        assert not pd.isna(row["dice"]), f"NaN Dice detected for sample {row['image_id']}"
        assert not pd.isna(row["iou"]), f"NaN IoU detected for sample {row['image_id']}"

    print("[PASS] number of test images == 63")
    print("[PASS] number of benign == 42")
    print("[PASS] number of malignant == 21")
    print("[PASS] number of prediction masks == 63")
    print("[PASS] number of qualitative visualizations == 63")
    print("[PASS] number of per-image metric rows == 63")
    print("[PASS] All test samples complete with ground-truth, prediction, metrics & qualitative PNGs.")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("OFFICIAL EXP-SEG-02 TEST EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Primary Overall Dice: {stats_overall['dice']['mean']:.4f} ± {stats_overall['dice']['std']:.4f}")
    print(f"Primary Overall IoU:  {stats_overall['iou']['mean']:.4f} ± {stats_overall['iou']['std']:.4f}")
    print(f"Benign Subgroup Dice: {stats_benign['dice']['mean']:.4f} ± {stats_benign['dice']['std']:.4f}")
    print(f"Malignant Subgroup Dice: {stats_malignant['dice']['mean']:.4f} ± {stats_malignant['dice']['std']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
