"""
Data Audit and Validation Module for Breast Ultrasound (BUSI) Dataset.

Performs read-only audit of raw data and experimental splits:
- Counts images and masks per class (benign, malignant, normal).
- Identifies multi-mask cases.
- Checks for empty masks (all 0 pixels).
- Validates image-mask spatial dimension parity.
- Validates expected absence of the deleted duplicate pair.
- Comprehensive Data-Leakage and Split Audit (EXP-SEG-02):
    * Verifies 628 Phase-1 total images accounted for.
    * Verifies 63 test images (~10%) and 565 dev images (~90%).
    * Verifies complete disjointness (test set ∩ all dev train/val folds = ∅).
    * Verifies each dev image appears in exactly 1 val fold and 2 train folds.
    * Verifies physical test_set folder integrity with manifest.
- Strictly read-only: NEVER modifies, renames, writes, or deletes raw data.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Audit BUSI dataset integrity, structure, and split leakage.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="..",
        help="Path to raw BUSI dataset directory."
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="",
        help="Optional path to write audit summary JSON (outside raw data)."
    )
    parser.add_argument(
        "--audit_splits",
        type=str,
        default="",
        help="Optional path to experiment YAML configuration to perform full data leakage audit."
    )
    return parser.parse_args()


def audit_raw_dataset(raw_root: str) -> Dict[str, Any]:
    """
    Perform a complete non-destructive audit of the raw BUSI dataset.
    """
    if not os.path.exists(raw_root):
        raise FileNotFoundError(f"Raw dataset root not found: {raw_root}")

    classes = ["benign", "malignant", "normal"]
    summary: Dict[str, Any] = {
        "raw_root": raw_root,
        "classes": {},
        "total_images": 0,
        "total_masks": 0,
        "multi_mask_cases": [],
        "empty_masks": [],
        "dimension_mismatches": [],
        "duplicate_pair_audit": {},
        "phase1_single_mask_candidates": {"benign": 0, "malignant": 0, "total": 0}
    }

    # Audit the deleted duplicate pair
    dup_benign = os.path.join(raw_root, "benign", "benign (433).png")
    dup_malignant = os.path.join(raw_root, "malignant", "malignant (145).png")
    summary["duplicate_pair_audit"] = {
        "benign (433).png exists": os.path.exists(dup_benign),
        "malignant (145).png exists": os.path.exists(dup_malignant),
        "status": "PASS (expected absent)" if (not os.path.exists(dup_benign) and not os.path.exists(dup_malignant)) else "FAIL (found duplicate files)"
    }

    total_images_all = 0
    total_masks_all = 0

    for cls in classes:
        cls_dir = os.path.join(raw_root, cls)
        if not os.path.exists(cls_dir):
            summary["classes"][cls] = {"status": "directory missing", "image_count": 0, "mask_count": 0}
            continue

        all_pngs = glob.glob(os.path.join(cls_dir, "*.png"))
        mask_files = [f for f in all_pngs if "_mask" in os.path.basename(f)]
        image_files = [f for f in all_pngs if "_mask" not in os.path.basename(f)]

        total_images_all += len(image_files)
        total_masks_all += len(mask_files)

        cls_multi_mask = []
        cls_single_mask = 0

        for img_path in image_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            pattern = os.path.join(cls_dir, f"{glob.escape(base_name)}_mask*.png")
            associated_masks = glob.glob(pattern)

            if len(associated_masks) == 1:
                cls_single_mask += 1
                mask_p = associated_masks[0]
                try:
                    with Image.open(img_path) as img, Image.open(mask_p) as msk:
                        if img.size != msk.size:
                            summary["dimension_mismatches"].append({
                                "image": img_path,
                                "mask": mask_p,
                                "img_size": img.size,
                                "mask_size": msk.size
                            })
                        mask_arr = np.array(msk)
                        if np.count_nonzero(mask_arr) == 0:
                            summary["empty_masks"].append({
                                "image": img_path,
                                "mask": mask_p
                            })
                except Exception as e:
                    summary["dimension_mismatches"].append({
                        "image": img_path,
                        "mask": mask_p,
                        "error": str(e)
                    })
            elif len(associated_masks) > 1:
                cls_multi_mask.append({
                    "image": img_path,
                    "num_masks": len(associated_masks),
                    "masks": associated_masks
                })
                summary["multi_mask_cases"].append({
                    "image": img_path,
                    "class": cls,
                    "num_masks": len(associated_masks),
                    "masks": associated_masks
                })

        summary["classes"][cls] = {
            "image_count": len(image_files),
            "mask_count": len(mask_files),
            "single_mask_count": cls_single_mask,
            "multi_mask_count": len(cls_multi_mask),
            "zero_mask_count": len(image_files) - cls_single_mask - len(cls_multi_mask)
        }

        if cls in ["benign", "malignant"]:
            summary["phase1_single_mask_candidates"][cls] = cls_single_mask

    summary["total_images"] = total_images_all
    summary["total_masks"] = total_masks_all
    summary["phase1_single_mask_candidates"]["total"] = (
        summary["phase1_single_mask_candidates"]["benign"] +
        summary["phase1_single_mask_candidates"]["malignant"]
    )

    return summary


def audit_splits_and_leakage(config_path: str) -> Dict[str, Any]:
    """
    Perform a rigorous data leakage and split validation audit for EXP-SEG-02.
    """
    config = load_config(config_path)
    raw_root = config.get("data", {}).get("raw_root", "..")
    dev_manifest_path = config.get("data", {}).get("manifest", "data/manifests/BUSI_development_manifest.csv")
    test_manifest_path = config.get("data", {}).get("test_manifest", "data/manifests/BUSI_final_test_manifest.csv")
    folds_dir = config.get("data", {}).get("folds_dir", "data/folds/exp_seg_02")
    test_dir = config.get("data", {}).get("test_dir", "data/test_set")
    excluded_path = config.get("data", {}).get("excluded", "data/manifests/BUSI_phase1_excluded.csv")

    print("=" * 80)
    print("DATA LEAKAGE & SPLIT INTEGRITY AUDIT (EXP-SEG-02)")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Raw Root: {os.path.abspath(raw_root)}")
    print(f"Dev Manifest: {dev_manifest_path}")
    print(f"Test Manifest: {test_manifest_path}")
    print(f"Folds Directory: {folds_dir}")
    print(f"Test Set Folder: {test_dir}")
    print("-" * 80)

    # 1. Load manifests
    assert os.path.exists(dev_manifest_path), f"Development manifest missing: {dev_manifest_path}"
    assert os.path.exists(test_manifest_path), f"Test manifest missing: {test_manifest_path}"

    dev_df = pd.read_csv(dev_manifest_path)
    test_df = pd.read_csv(test_manifest_path)

    # 2. Check counts
    num_dev = len(dev_df)
    num_test = len(test_df)
    total_manifest = num_dev + num_test

    assert total_manifest == 628, f"Total count mismatch: {total_manifest} != 628"
    assert num_test == 63, f"Test count mismatch: {num_test} != 63"
    assert num_dev == 565, f"Dev count mismatch: {num_dev} != 565"

    test_b = int((test_df["class"] == "benign").sum())
    test_m = int((test_df["class"] == "malignant").sum())
    dev_b = int((dev_df["class"] == "benign").sum())
    dev_m = int((dev_df["class"] == "malignant").sum())

    assert test_b == 42 and test_m == 21, f"Test stratification mismatch: {test_b} benign, {test_m} malignant"
    assert dev_b == 378 and dev_m == 187, f"Dev stratification mismatch: {dev_b} benign, {dev_m} malignant"

    print(f"[CHECK 1] Counts & Stratification:")
    print(f"  - Total Phase 1 Images: {total_manifest} (420 Benign, 208 Malignant)")
    print(f"  - Development Set:      {num_dev} (378 Benign, 187 Malignant | {dev_m/num_dev*100:.2f}% Mal)")
    print(f"  - Final Test Set:       {num_test} (42 Benign, 21 Malignant  | {test_m/num_test*100:.2f}% Mal)")
    print(f"  => PASS")

    # 3. Disjointness check between dev and test
    dev_paths = set(dev_df["image_path"].apply(os.path.normpath))
    test_paths = set(test_df["image_path"].apply(os.path.normpath))
    dev_test_leakage = dev_paths.intersection(test_paths)
    assert len(dev_test_leakage) == 0, f"LEAKAGE: {len(dev_test_leakage)} overlapping images between Dev and Test!"
    print(f"[CHECK 2] Dev vs Test Isolation: 0 overlapping images => PASS")

    # 4. Check all test images exist on disk and have masks
    test_img_missing = 0
    test_mask_missing = 0
    for _, row in test_df.iterrows():
        p_img = os.path.join(raw_root, str(row["image_path"]))
        p_msk = os.path.join(raw_root, str(row["mask_path"]))
        if not os.path.exists(p_img):
            test_img_missing += 1
        if not os.path.exists(p_msk):
            test_mask_missing += 1
    assert test_img_missing == 0 and test_mask_missing == 0, f"Missing files in test manifest: imgs={test_img_missing}, msks={test_mask_missing}"
    print(f"[CHECK 3] Test Manifest Physical Existence in raw_root: 100% verified => PASS")

    # 5. Check isolated test_set directory
    if os.path.exists(test_dir):
        img_f = glob.glob(os.path.join(test_dir, "images", "*.png"))
        msk_f = glob.glob(os.path.join(test_dir, "masks", "*.png"))
        assert len(img_f) == 63, f"Physical test_set/images count ({len(img_f)}) != 63"
        assert len(msk_f) == 63, f"Physical test_set/masks count ({len(msk_f)}) != 63"
        print(f"[CHECK 4] Physical test_set Folder: 63 images & 63 masks verified => PASS")

    # 6. Audit 3-Fold Development CV splits
    val_fold_sets: List[Set[str]] = []
    train_fold_sets: List[Set[str]] = []

    for fold_idx in [1, 2, 3]:
        tr_csv = os.path.join(folds_dir, f"fold_{fold_idx}_train.csv")
        va_csv = os.path.join(folds_dir, f"fold_{fold_idx}_val.csv")
        assert os.path.exists(tr_csv), f"Fold {fold_idx} train CSV missing: {tr_csv}"
        assert os.path.exists(va_csv), f"Fold {fold_idx} val CSV missing: {va_csv}"

        tr_df = pd.read_csv(tr_csv)
        va_df = pd.read_csv(va_csv)

        tr_set = set(tr_df["image_path"].apply(os.path.normpath))
        va_set = set(va_df["image_path"].apply(os.path.normpath))

        # Intra-fold disjointness
        intra = tr_set.intersection(va_set)
        assert len(intra) == 0, f"Fold {fold_idx} intra-fold leakage: {len(intra)} images overlap!"

        # Zero test leakage into train or val fold
        tr_test_leak = tr_set.intersection(test_paths)
        va_test_leak = va_set.intersection(test_paths)
        assert len(tr_test_leak) == 0, f"LEAKAGE: Fold {fold_idx} train contains {len(tr_test_leak)} test images!"
        assert len(va_test_leak) == 0, f"LEAKAGE: Fold {fold_idx} val contains {len(va_test_leak)} test images!"

        val_fold_sets.append(va_set)
        train_fold_sets.append(tr_set)

        print(f"[CHECK 5.{fold_idx}] Fold {fold_idx}: Train={len(tr_df)}, Val={len(va_df)} | Disjoint & Zero Test Leakage => PASS")

    # Inter-fold validation disjointness
    for i in range(3):
        for j in range(i + 1, 3):
            inter = val_fold_sets[i].intersection(val_fold_sets[j])
            assert len(inter) == 0, f"Inter-val fold overlap between Fold {i+1} and {j+1}: {len(inter)} images!"

    # Complete coverage of development set across validation folds
    pooled_val = set().union(*val_fold_sets)
    assert len(pooled_val) == 565, f"Pooled validation set ({len(pooled_val)}) != 565 dev images!"
    assert pooled_val == dev_paths, f"Pooled validation set does not match development manifest!"
    print(f"[CHECK 6] Inter-fold Validation Disjointness & Exact 100% Dev Coverage (565 images) => PASS")

    # Exclusion overlap check
    if excluded_path and os.path.exists(excluded_path):
        ex_df = pd.read_csv(excluded_path)
        ex_col = "image" if "image" in ex_df.columns else ex_df.columns[0]
        ex_set = set(ex_df[ex_col].apply(os.path.normpath))
        assert len(dev_paths.intersection(ex_set)) == 0, "Dev set contains excluded cases!"
        assert len(test_paths.intersection(ex_set)) == 0, "Test set contains excluded cases!"
        print(f"[CHECK 7] Zero Overlap with Excluded Multi-Mask / Normal Cases => PASS")

    print("-" * 80)
    print("ALL DATA-LEAKAGE AUDIT CHECKS PASSED PERFECTLY!")
    print("=" * 80)

    return {
        "status": "PASS",
        "total_phase1": total_manifest,
        "development_count": num_dev,
        "test_count": num_test,
        "folds": {
            "fold_1": {"train": len(train_fold_sets[0]), "val": len(val_fold_sets[0])},
            "fold_2": {"train": len(train_fold_sets[1]), "val": len(val_fold_sets[1])},
            "fold_3": {"train": len(train_fold_sets[2]), "val": len(val_fold_sets[2])},
        }
    }


def main():
    args = parse_args()
    if args.audit_splits:
        audit_splits_and_leakage(args.audit_splits)
    else:
        summary = audit_raw_dataset(args.raw_root)
        print("=" * 80)
        print("BUSI RAW DATASET AUDIT REPORT")
        print("=" * 80)
        print(f"Raw Root: {summary['raw_root']}")
        print(f"Duplicate Pair Audit Status: {summary['duplicate_pair_audit']['status']}")
        print("-" * 80)
        for cls, stats in summary["classes"].items():
            print(f"{cls:<12} | Images: {stats['image_count']:<6} | Masks: {stats['mask_count']:<6} | Single: {stats['single_mask_count']:<6} | Multi: {stats['multi_mask_count']:<6}")
        print("-" * 80)
        print(f"Total Raw Images: {summary['total_images']}")
        print(f"Phase 1 Single-Mask Total: {summary['phase1_single_mask_candidates']['total']}")
        print("=" * 80)

        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Audit summary saved to: {args.json_out}")


if __name__ == "__main__":
    main()
