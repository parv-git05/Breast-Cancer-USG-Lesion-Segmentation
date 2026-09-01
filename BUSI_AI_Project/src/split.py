"""
Dataset Splitting Module for Breast Ultrasound (BUSI) AI Pipeline.

Implements Stratified 3-Fold Cross-Validation:
- Stratified by lesion class (benign / malignant)
- Fixed reproducible random seed (default: 42)
- 3 mutually exclusive folds: 2 folds training (~418-419 images), 1 fold validation (~209-210 images)
- Complete out-of-fold validation coverage (628 total images)
- Programmatic Quality Gate 2 assertions (disjointness, complete coverage, stratification preservation)
- Deletes obsolete 70/15/15 split files to avoid clutter/clashes
- Outputs data/folds/ fold CSVs and cv_metadata.json
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifest import validate_manifest
from src.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stratified 3-fold cross-validation splits.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/split.yaml",
        help="Path to YAML split configuration file."
    )
    return parser.parse_args()


def generate_3fold_cv_splits(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    n_splits: int = 3,
    folds_dir: str = "data/folds",
    excluded_path: Optional[str] = "data/manifests/BUSI_phase1_excluded.csv"
) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], Dict[str, Any]]:
    """
    Generate stratified n-fold cross-validation splits and verify Quality Gate 2 criteria.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got: {n_splits}")

    set_seed(seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds_data: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    val_image_sets: List[set] = []
    
    total_manifest_count = len(manifest_df)
    overall_malignant_ratio = float((manifest_df["class"] == "malignant").mean())

    metadata_counts: Dict[str, Any] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(manifest_df, manifest_df["class"]), start=1):
        train_df = manifest_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = manifest_df.iloc[val_idx].copy().reset_index(drop=True)

        train_df["fold"] = fold_idx
        val_df["fold"] = fold_idx
        train_df["split"] = "train"
        val_df["split"] = "val"

        train_imgs = set(train_df["image_path"].apply(os.path.normpath))
        val_imgs = set(val_df["image_path"].apply(os.path.normpath))

        # 1. Intra-fold disjointness assertion
        intra_overlap = train_imgs.intersection(val_imgs)
        if len(intra_overlap) > 0:
            raise AssertionError(
                f"[GATE 2 VIOLATION] Intra-fold data leakage in Fold {fold_idx}! "
                f"{len(intra_overlap)} overlapping images found between train and val."
            )

        # 2. Stratification ratio assertion (+- 2% tolerance)
        val_mal_ratio = float((val_df["class"] == "malignant").mean())
        diff = abs(val_mal_ratio - overall_malignant_ratio)
        if diff > 0.02:
            raise AssertionError(
                f"[GATE 2 VIOLATION] Class ratio deviation in Fold {fold_idx} val exceeds 2%: "
                f"Overall={overall_malignant_ratio:.4f}, Fold Val={val_mal_ratio:.4f} (diff={diff:.4f})"
            )

        folds_data.append((train_df, val_df))
        val_image_sets.append(val_imgs)

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

    # 3. Inter-fold validation disjointness assertion
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            inter_val_overlap = val_image_sets[i].intersection(val_image_sets[j])
            if len(inter_val_overlap) > 0:
                raise AssertionError(
                    f"[GATE 2 VIOLATION] Inter-fold validation overlap between Fold {i+1} and Fold {j+1}! "
                    f"Found {len(inter_val_overlap)} shared images."
                )

    # 4. Complete coverage assertion across all validation sets
    all_val_imgs = set().union(*val_image_sets)
    if len(all_val_imgs) != total_manifest_count:
        raise AssertionError(
            f"[GATE 2 VIOLATION] Total unique validation images ({len(all_val_imgs)}) "
            f"!= Manifest count ({total_manifest_count})"
        )

    # 5. Exclusion integrity assertion
    if excluded_path and os.path.exists(excluded_path):
        excluded_df = pd.read_csv(excluded_path)
        ex_col = "image" if "image" in excluded_df.columns else excluded_df.columns[0]
        excluded_set = set(excluded_df[ex_col].apply(os.path.normpath))
        ex_overlap = all_val_imgs.intersection(excluded_set)
        if len(ex_overlap) > 0:
            raise AssertionError(f"[GATE 2 VIOLATION] Excluded cases found in CV folds: {ex_overlap}")

    metadata: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "n_splits": n_splits,
        "total_images": total_manifest_count,
        "counts": metadata_counts,
        "overall": {
            "total": total_manifest_count,
            "benign": int((manifest_df["class"] == "benign").sum()),
            "malignant": int((manifest_df["class"] == "malignant").sum()),
            "malignant_ratio": float(overall_malignant_ratio),
        },
        "note": "3-fold stratified image-level cross-validation; not patient-wise."
    }

    return folds_data, metadata


def cleanup_legacy_splits(legacy_dir: str = "data/splits") -> None:
    """Completely remove obsolete 70/15/15 split directory to prevent clashes."""
    if os.path.exists(legacy_dir):
        shutil.rmtree(legacy_dir)
        print(f"[INFO] Legacy 70/15/15 splits directory purged: {legacy_dir}")


def main():
    args = parse_args()
    config = load_config(args.config)

    manifest_path = config["data"]["manifest"]
    excluded_path = config["data"]["excluded"]
    folds_dir = config["data"].get("folds_dir", "data/folds")
    seed = config["data"].get("seed", 42)
    n_splits = config["data"].get("n_splits", 3)
    raw_root = config["data"].get("raw_root", "")

    print("=" * 80)
    print("STEP 1: VALIDATING MANIFEST (GATE 1)")
    print("=" * 80)
    validate_manifest(manifest_path, excluded_path, raw_root=raw_root)
    print("[INFO] Gate 1 validation PASSED.")

    print("=" * 80)
    print("STEP 2: GENERATING STRATIFIED 3-FOLD CV SPLITS (GATE 2)")
    print("=" * 80)
    manifest_df = pd.read_csv(manifest_path)
    
    # Purge legacy splits directory if present
    cleanup_legacy_splits("data/splits")

    folds_data, metadata = generate_3fold_cv_splits(
        manifest_df=manifest_df,
        seed=seed,
        n_splits=n_splits,
        folds_dir=folds_dir,
        excluded_path=excluded_path
    )

    os.makedirs(folds_dir, exist_ok=True)

    for fold_idx, (train_df, val_df) in enumerate(folds_data, start=1):
        train_path = os.path.join(folds_dir, f"fold_{fold_idx}_train.csv")
        val_path = os.path.join(folds_dir, f"fold_{fold_idx}_val.csv")
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        print(f"  - Fold {fold_idx}: Train={train_path} ({len(train_df)} rows) | Val={val_path} ({len(val_df)} rows)")

    meta_path = os.path.join(folds_dir, "cv_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n3-Fold CV Manifest successfully written to: {folds_dir}")
    print(f"Metadata file: {meta_path}")
    print("-" * 80)
    print("Fold Breakdown:")
    for fold_name, counts in metadata["counts"].items():
        tr_c = counts["train"]
        va_c = counts["val"]
        print(f"  [{fold_name.upper()}] Train Total: {tr_c['total']} (Benign: {tr_c['benign']}, Malignant: {tr_c['malignant']}) | Val Total: {va_c['total']} (Benign: {va_c['benign']}, Malignant: {va_c['malignant']})")
    print("-" * 80)
    print(f"Note: {metadata['note']}")
    print("[GATE 2 CHECK PASSED] 3-Fold CV splits are mutually disjoint, stratified, and verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
