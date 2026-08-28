"""
Dataset Splitting Module for Breast Ultrasound (BUSI) AI Pipeline.

Implements image-level stratified train / validation / test splitting:
- Default split ratios: 70% train, 15% val, 15% test
- Stratified by lesion class (benign / malignant)
- Fixed reproducible random seed
- Programmatic Quality Gate 2 assertions (disjointness, complete coverage, stratification preservation)
- Outputs data/splits/train.csv, val.csv, test.csv, and split_metadata.json
"""

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.manifest import validate_manifest
from src.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stratified train/val/test splits.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/split.yaml",
        help="Path to YAML split configuration file."
    )
    return parser.parse_args()


def generate_splits(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    excluded_path: Optional[str] = "data/manifests/BUSI_phase1_excluded.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate stratified train, val, test splits and verify Quality Gate 2 criteria.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got: {ratios} (sum={sum(ratios)})")

    # Set random seed
    set_seed(seed)

    # Initial split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, temp_val_test_df = train_test_split(
        manifest_df,
        test_size=val_test_ratio,
        stratify=manifest_df["class"],
        random_state=seed,
        shuffle=True
    )

    # Secondary split: val vs test (proportional)
    val_rel_ratio = val_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        temp_val_test_df,
        train_size=val_rel_ratio,
        stratify=temp_val_test_df["class"],
        random_state=seed,
        shuffle=True
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # -------------------------------------------------------------
    # QUALITY GATE 2 ASSERTIONS
    # -------------------------------------------------------------
    total_manifest_count = len(manifest_df)
    train_count = len(train_df)
    val_count = len(val_df)
    test_count = len(test_df)

    # 1. Total row sum assertion
    if train_count + val_count + test_count != total_manifest_count:
        raise AssertionError(
            f"[GATE 2 VIOLATION] Total split rows ({train_count + val_count + test_count}) "
            f"!= Manifest count ({total_manifest_count})"
        )

    # 2. Programmatic Disjointness assertion
    train_imgs = set(train_df["image_path"].apply(os.path.normpath))
    val_imgs = set(val_df["image_path"].apply(os.path.normpath))
    test_imgs = set(test_df["image_path"].apply(os.path.normpath))

    tv_overlap = train_imgs.intersection(val_imgs)
    tt_overlap = train_imgs.intersection(test_imgs)
    vt_overlap = val_imgs.intersection(test_imgs)

    if len(tv_overlap) > 0 or len(tt_overlap) > 0 or len(vt_overlap) > 0:
        raise AssertionError(
            f"[GATE 2 VIOLATION] Data leakage detected across splits! "
            f"Train-Val: {len(tv_overlap)}, Train-Test: {len(tt_overlap)}, Val-Test: {len(vt_overlap)}"
        )

    # 3. Class stratification ratio assertion (within +- 2% tolerance)
    overall_malignant_ratio = (manifest_df["class"] == "malignant").mean()
    for name, split_data in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_mal_ratio = (split_data["class"] == "malignant").mean()
        diff = abs(split_mal_ratio - overall_malignant_ratio)
        if diff > 0.02:
            raise AssertionError(
                f"[GATE 2 VIOLATION] Class ratio deviation in {name} split exceeds 2%: "
                f"Overall={overall_malignant_ratio:.4f}, Split={split_mal_ratio:.4f} (diff={diff:.4f})"
            )

    # 4. Exclusion integrity assertion
    if excluded_path and os.path.exists(excluded_path):
        excluded_df = pd.read_csv(excluded_path)
        ex_col = "image" if "image" in excluded_df.columns else excluded_df.columns[0]
        excluded_set = set(excluded_df[ex_col].apply(os.path.normpath))
        all_split_imgs = train_imgs.union(val_imgs).union(test_imgs)
        ex_overlap = all_split_imgs.intersection(excluded_set)
        if len(ex_overlap) > 0:
            raise AssertionError(f"[GATE 2 VIOLATION] Excluded cases found in splits: {ex_overlap}")

    metadata: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "seed": seed,
        "split_ratios": list(ratios),
        "total_images": total_manifest_count,
        "counts": {
            "train": {
                "total": train_count,
                "benign": int((train_df["class"] == "benign").sum()),
                "malignant": int((train_df["class"] == "malignant").sum()),
                "malignant_ratio": float((train_df["class"] == "malignant").mean())
            },
            "val": {
                "total": val_count,
                "benign": int((val_df["class"] == "benign").sum()),
                "malignant": int((val_df["class"] == "malignant").sum()),
                "malignant_ratio": float((val_df["class"] == "malignant").mean())
            },
            "test": {
                "total": test_count,
                "benign": int((test_df["class"] == "benign").sum()),
                "malignant": int((test_df["class"] == "malignant").sum()),
                "malignant_ratio": float((test_df["class"] == "malignant").mean())
            }
        },
        "overall": {
            "total": total_manifest_count,
            "benign": int((manifest_df["class"] == "benign").sum()),
            "malignant": int((manifest_df["class"] == "malignant").sum()),
            "malignant_ratio": float(overall_malignant_ratio)
        },
        "note": (
            "image-level stratified; not patient-wise "
            "(BUSI patient IDs not verified in local data)"
        )
    }

    return train_df, val_df, test_df, metadata


def main():
    args = parse_args()
    config = load_config(args.config)

    manifest_path = config["data"]["manifest"]
    excluded_path = config["data"]["excluded"]
    splits_dir = config["data"]["splits_dir"]
    seed = config["data"].get("split_seed", 42)
    ratios = tuple(config["data"].get("split_ratios", [0.70, 0.15, 0.15]))

    print("=" * 80)
    print("STEP 1: VALIDATING MANIFEST (GATE 1)")
    print("=" * 80)
    validate_manifest(manifest_path, excluded_path)
    print("[INFO] Gate 1 validation PASSED.")

    print("=" * 80)
    print("STEP 2: GENERATING STRATIFIED SPLITS (GATE 2)")
    print("=" * 80)
    manifest_df = pd.read_csv(manifest_path)
    train_df, val_df, test_df, metadata = generate_splits(
        manifest_df=manifest_df,
        seed=seed,
        ratios=ratios,
        excluded_path=excluded_path
    )

    os.makedirs(splits_dir, exist_ok=True)
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    meta_path = os.path.join(splits_dir, "split_metadata.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Splits successfully written to: {splits_dir}")
    print(f"  - Train: {train_path} ({len(train_df)} rows)")
    print(f"  - Val:   {val_path} ({len(val_df)} rows)")
    print(f"  - Test:  {test_path} ({len(test_df)} rows)")
    print(f"  - Meta:  {meta_path}")
    print("-" * 80)
    print("Exact Counts Breakdown:")
    for split_name, counts in metadata["counts"].items():
        print(f"  [{split_name.upper():<5}] Total: {counts['total']:<4} | Benign: {counts['benign']:<4} | Malignant: {counts['malignant']:<4} | Malignant Ratio: {counts['malignant_ratio']:.4f}")
    print("-" * 80)
    print(f"Note: {metadata['note']}")
    print("[GATE 2 CHECK PASSED] Splits are mutually disjoint, stratified, and verified.")
    print("=" * 80)


if __name__ == "__main__":
    main()
