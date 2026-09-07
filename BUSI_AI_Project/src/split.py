"""
Dataset Splitting Module for Breast Ultrasound (BUSI) AI Pipeline (EXP-SEG-02).

Implements the official Two-Stage Stratified Protocol:
1. Stage 1: Fixed Stratified Holdout Test Set (~10%, exactly 63 images: 42 benign, 21 malignant)
   and Development Set (~90%, exactly 565 images: 378 benign, 187 malignant).
2. Stage 2: Stratified 3-Fold Cross-Validation performed STRICTLY on the 565 development images.
3. Physical Test Set Copy: Copies the 63 test images and masks to data/test_set/images/ and
   data/test_set/masks/ for clear physical isolation and convenience, while the manifest
   remains the authoritative source of truth.
4. Programmatic Quality Gate 2 assertions (complete isolation, disjointness, stratification balance, zero leakage).
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifest import validate_manifest
from src.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stratified holdout test split and 3-fold CV splits on development data.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/split.yaml",
        help="Path to YAML split configuration file."
    )
    return parser.parse_args()


def copy_isolated_test_set(
    test_df: pd.DataFrame,
    raw_root: str,
    test_dir: str = "data/test_set"
) -> None:
    """
    Physically copy test images and masks to an isolated directory for organization,
    while ensuring the manifest remains the source of truth.
    """
    images_dir = os.path.join(test_dir, "images")
    masks_dir = os.path.join(test_dir, "masks")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    copied_images = 0
    copied_masks = 0

    for _, row in test_df.iterrows():
        rel_img = str(row["image_path"])
        rel_mask = str(row["mask_path"])

        src_img = os.path.join(raw_root, rel_img) if raw_root else rel_img
        src_mask = os.path.join(raw_root, rel_mask) if raw_root else rel_mask

        if not os.path.exists(src_img):
            raise FileNotFoundError(f"Source test image not found: {src_img}")
        if not os.path.exists(src_mask):
            raise FileNotFoundError(f"Source test mask not found: {src_mask}")

        dst_img = os.path.join(images_dir, os.path.basename(rel_img))
        dst_mask = os.path.join(masks_dir, os.path.basename(rel_mask))

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_mask, dst_mask)

        copied_images += 1
        copied_masks += 1

    print(f"[INFO] Physically copied {copied_images} test images to: {images_dir}")
    print(f"[INFO] Physically copied {copied_masks} test masks to: {masks_dir}")


def generate_exp_seg_02_splits(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    test_size: int = 63,
    n_splits: int = 3,
    raw_root: str = "",
    manifests_dir: str = "data/manifests",
    folds_dir: str = "data/folds/exp_seg_02",
    test_dir: str = "data/test_set",
    excluded_path: Optional[str] = "data/manifests/BUSI_phase1_excluded.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[pd.DataFrame, pd.DataFrame]], Dict[str, Any]]:
    """
    Execute full two-stage split with strict Quality Gate assertions.
    """
    set_seed(seed)
    total_count = len(manifest_df)
    if total_count != 628:
        raise ValueError(f"Expected 628 Phase-1 images, got {total_count}")

    # =========================================================================
    # STAGE 1: Stratified Holdout Test Split (~10% = 63 images)
    # =========================================================================
    dev_df, test_df = train_test_split(
        manifest_df,
        test_size=test_size,
        stratify=manifest_df["class"],
        random_state=seed,
        shuffle=True
    )
    dev_df = dev_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    test_df["split"] = "test"
    dev_df["split"] = "development"

    # Verify Stage 1 counts & stratification
    test_benign = int((test_df["class"] == "benign").sum())
    test_malignant = int((test_df["class"] == "malignant").sum())
    dev_benign = int((dev_df["class"] == "benign").sum())
    dev_malignant = int((dev_df["class"] == "malignant").sum())

    assert len(test_df) == test_size, f"Test size mismatch: {len(test_df)} != {test_size}"
    assert len(dev_df) == (total_count - test_size), f"Dev size mismatch: {len(dev_df)} != {total_count - test_size}"

    # Disjointness between dev and test
    dev_imgs = set(dev_df["image_path"].apply(os.path.normpath))
    test_imgs = set(test_df["image_path"].apply(os.path.normpath))
    dev_test_overlap = dev_imgs.intersection(test_imgs)
    if len(dev_test_overlap) > 0:
        raise AssertionError(f"[GATE 2 VIOLATION] Leakage detected between Dev and Test! Overlap: {dev_test_overlap}")

    # =========================================================================
    # STAGE 2: Stratified 3-Fold CV on Development Set ONLY (565 images)
    # =========================================================================
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    val_image_sets: List[set] = []
    metadata_counts: Dict[str, Any] = {}

    overall_dev_mal_ratio = float((dev_df["class"] == "malignant").mean())

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df["class"]), start=1):
        train_df = dev_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = dev_df.iloc[val_idx].copy().reset_index(drop=True)

        train_df["fold"] = fold_idx
        val_df["fold"] = fold_idx
        train_df["split"] = "train"
        val_df["split"] = "val"

        train_fold_imgs = set(train_df["image_path"].apply(os.path.normpath))
        val_fold_imgs = set(val_df["image_path"].apply(os.path.normpath))

        # 1. Intra-fold disjointness
        intra_overlap = train_fold_imgs.intersection(val_fold_imgs)
        if len(intra_overlap) > 0:
            raise AssertionError(f"[GATE 2 VIOLATION] Intra-fold overlap in Fold {fold_idx}: {len(intra_overlap)} images")

        # 2. Test leakage into CV fold
        if len(train_fold_imgs.intersection(test_imgs)) > 0:
            raise AssertionError(f"[GATE 2 VIOLATION] Test image found in Fold {fold_idx} training set!")
        if len(val_fold_imgs.intersection(test_imgs)) > 0:
            raise AssertionError(f"[GATE 2 VIOLATION] Test image found in Fold {fold_idx} validation set!")

        # 3. Stratification ratio preservation
        val_mal_ratio = float((val_df["class"] == "malignant").mean())
        if abs(val_mal_ratio - overall_dev_mal_ratio) > 0.03:
            raise AssertionError(f"[GATE 2 VIOLATION] Fold {fold_idx} val class ratio deviation > 3%")

        folds_data.append((train_df, val_df))
        val_image_sets.append(val_fold_imgs)

        metadata_counts[f"fold_{fold_idx}"] = {
            "train": {
                "total": len(train_df),
                "benign": int((train_df["class"] == "benign").sum()),
                "malignant": int((train_df["class"] == "malignant").sum()),
                "malignant_ratio": float((train_df["class"] == "malignant").mean()),
            },
            "val": {
                "total": len(val_df),
                "benign": int((val_df["class"] == "benign").sum()),
                "malignant": int((val_df["class"] == "malignant").sum()),
                "malignant_ratio": float((val_df["class"] == "malignant").mean()),
            }
        }

    # Inter-fold validation disjointness
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            inter_val_overlap = val_image_sets[i].intersection(val_image_sets[j])
            if len(inter_val_overlap) > 0:
                raise AssertionError(f"[GATE 2 VIOLATION] Inter-fold validation overlap between Fold {i+1} and {j+1}")

    # Complete coverage of development set across validation folds
    all_val_imgs = set().union(*val_image_sets)
    if len(all_val_imgs) != len(dev_df):
        raise AssertionError(f"[GATE 2 VIOLATION] Pooled val images ({len(all_val_imgs)}) != Dev set count ({len(dev_df)})")

    # Exclusion assertion
    if excluded_path and os.path.exists(excluded_path):
        excluded_df = pd.read_csv(excluded_path)
        ex_col = "image" if "image" in excluded_df.columns else excluded_df.columns[0]
        excluded_set = set(excluded_df[ex_col].apply(os.path.normpath))
        if len(dev_imgs.intersection(excluded_set)) > 0 or len(test_imgs.intersection(excluded_set)) > 0:
            raise AssertionError("[GATE 2 VIOLATION] Excluded cases found in dev or test sets!")

    # =========================================================================
    # Write Manifests & Fold CSVs
    # =========================================================================
    os.makedirs(manifests_dir, exist_ok=True)
    dev_manifest_path = os.path.join(manifests_dir, "BUSI_development_manifest.csv")
    test_manifest_path = os.path.join(manifests_dir, "BUSI_final_test_manifest.csv")

    dev_df.to_csv(dev_manifest_path, index=False)
    test_df.to_csv(test_manifest_path, index=False)
    print(f"[INFO] Development manifest written ({len(dev_df)} rows): {dev_manifest_path}")
    print(f"[INFO] Final test manifest written ({len(test_df)} rows): {test_manifest_path}")

    # Copy isolated test files
    copy_isolated_test_set(test_df=test_df, raw_root=raw_root, test_dir=test_dir)

    os.makedirs(folds_dir, exist_ok=True)
    for fold_idx, (tr_df, va_df) in enumerate(folds_data, start=1):
        tr_path = os.path.join(folds_dir, f"fold_{fold_idx}_train.csv")
        va_path = os.path.join(folds_dir, f"fold_{fold_idx}_val.csv")
        tr_df.to_csv(tr_path, index=False)
        va_df.to_csv(va_path, index=False)
        print(f"  - Fold {fold_idx}: Train={tr_path} ({len(tr_df)} rows) | Val={va_path} ({len(va_df)} rows)")

    metadata = {
        "experiment": "EXP-SEG-02",
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "n_splits": n_splits,
        "total_phase1_images": total_count,
        "test_set": {
            "total": len(test_df),
            "benign": test_benign,
            "malignant": test_malignant,
            "malignant_ratio": float(test_malignant / len(test_df)),
            "manifest_path": test_manifest_path,
            "physical_dir": test_dir
        },
        "development_set": {
            "total": len(dev_df),
            "benign": dev_benign,
            "malignant": dev_malignant,
            "malignant_ratio": float(dev_malignant / len(dev_df)),
            "manifest_path": dev_manifest_path,
            "folds_dir": folds_dir,
            "counts": metadata_counts
        },
        "note": "Stage 1: 10% stratified locked holdout test. Stage 2: 3-fold stratified CV on development data only."
    }

    meta_path = os.path.join(folds_dir, "cv_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] EXP-SEG-02 metadata written to: {meta_path}")

    return dev_df, test_df, folds_data, metadata


def main():
    args = parse_args()
    config = load_config(args.config)

    manifest_path = config["data"]["manifest"]
    excluded_path = config["data"].get("excluded", "data/manifests/BUSI_phase1_excluded.csv")
    folds_dir = config["data"].get("folds_dir", "data/folds/exp_seg_02")
    manifests_dir = config["data"].get("manifests_dir", "data/manifests")
    test_dir = config["data"].get("test_dir", "data/test_set")
    seed = config["data"].get("seed", 42)
    n_splits = config["data"].get("n_splits", 3)
    raw_root = config["data"].get("raw_root", "")

    print("=" * 80)
    print("STEP 1: VALIDATING BASE PHASE 1 MANIFEST (GATE 1)")
    print("=" * 80)
    validate_manifest(manifest_path, excluded_path, raw_root=raw_root)
    print("[INFO] Gate 1 validation PASSED.")

    print("=" * 80)
    print("STEP 2: GENERATING EXP-SEG-02 TWO-STAGE SPLITS (GATE 2)")
    print("=" * 80)
    manifest_df = pd.read_csv(manifest_path)

    generate_exp_seg_02_splits(
        manifest_df=manifest_df,
        seed=seed,
        test_size=63,
        n_splits=n_splits,
        raw_root=raw_root,
        manifests_dir=manifests_dir,
        folds_dir=folds_dir,
        test_dir=test_dir,
        excluded_path=excluded_path
    )

    print("\n" + "=" * 80)
    print("[GATE 2 CHECK PASSED] EXP-SEG-02 Holdout Test and Development Folds Generated & Verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
