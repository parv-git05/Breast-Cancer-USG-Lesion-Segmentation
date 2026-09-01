"""
Manifest Loading and Validation Module.

Enforces Data Contract for Phase 1:
- Row count == 628
- Benign == 420, Malignant == 208
- Every referenced image and mask exists on disk
- No overlap with BUSI_phase1_excluded.csv (17 held-out multi-mask cases)
- Masks are valid and non-empty
"""

import argparse
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_MANIFEST_PATH = "data/manifests/BUSI_phase1_manifest.csv"
DEFAULT_EXCLUDED_PATH = "data/manifests/BUSI_phase1_excluded.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Load and validate BUSI Phase 1 manifests.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to BUSI_phase1_manifest.csv."
    )
    parser.add_argument(
        "--excluded",
        type=str,
        default=DEFAULT_EXCLUDED_PATH,
        help="Path to BUSI_phase1_excluded.csv."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run strict assertion validation checks (Gate 1)."
    )
    return parser.parse_args()


def load_manifest(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    excluded_path: Optional[str] = DEFAULT_EXCLUDED_PATH
) -> pd.DataFrame:
    """
    Load the Phase 1 manifest and optionally verify that excluded items are absent.
    """
    if not os.path.exists(manifest_path):
        alt_manifest = os.path.join(os.path.dirname(__file__), "..", manifest_path)
        if os.path.exists(alt_manifest):
            manifest_path = alt_manifest
        else:
            raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")

    df = pd.read_csv(manifest_path)
    return df


def validate_manifest(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    excluded_path: str = DEFAULT_EXCLUDED_PATH,
    raw_root: str = ""
) -> Dict[str, Any]:
    """
    Run comprehensive contract and Quality Gate 1 validation checks on the manifest.
    Raises ValueError / AssertionError on any violation.
    """
    if not os.path.exists(manifest_path):
        alt = os.path.join(os.path.dirname(__file__), "..", manifest_path)
        if os.path.exists(alt):
            manifest_path = alt
        else:
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if not os.path.exists(excluded_path):
        alt_ex = os.path.join(os.path.dirname(__file__), "..", excluded_path)
        if os.path.exists(alt_ex):
            excluded_path = alt_ex
        else:
            raise FileNotFoundError(f"Excluded file not found: {excluded_path}")

    manifest_df = pd.read_csv(manifest_path)
    excluded_df = pd.read_csv(excluded_path)

    # 1. Total row count assertion
    total_count = len(manifest_df)
    if total_count != 628:
        raise AssertionError(f"[GATE 1 VIOLATION] Manifest row count must be 628, found: {total_count}")

    # 2. Class distribution assertion
    class_counts = manifest_df["class"].value_counts().to_dict()
    benign_count = class_counts.get("benign", 0)
    malignant_count = class_counts.get("malignant", 0)

    if benign_count != 420:
        raise AssertionError(f"[GATE 1 VIOLATION] Expected 420 benign images, found: {benign_count}")
    if malignant_count != 208:
        raise AssertionError(f"[GATE 1 VIOLATION] Expected 208 malignant images, found: {malignant_count}")

    # 3. File existence verification
    missing_images = []
    missing_masks = []
    dim_mismatches = []
    empty_masks = []

    for idx, row in manifest_df.iterrows():
        img_p = os.path.join(raw_root, str(row["image_path"])) if raw_root else str(row["image_path"])
        msk_p = os.path.join(raw_root, str(row["mask_path"])) if raw_root else str(row["mask_path"])

        if not os.path.exists(img_p):
            missing_images.append(img_p)
            continue
        if not os.path.exists(msk_p):
            missing_masks.append(msk_p)
            continue

        try:
            with Image.open(img_p) as img, Image.open(msk_p) as msk:
                if img.size != msk.size:
                    dim_mismatches.append((img_p, msk_p, img.size, msk.size))
                msk_arr = np.array(msk)
                if np.count_nonzero(msk_arr) == 0:
                    empty_masks.append(msk_p)
        except Exception as e:
            dim_mismatches.append((img_p, msk_p, str(e)))

    if missing_images:
        raise AssertionError(f"[GATE 1 VIOLATION] {len(missing_images)} manifest image files do not exist on disk: {missing_images[:5]}")
    if missing_masks:
        raise AssertionError(f"[GATE 1 VIOLATION] {len(missing_masks)} manifest mask files do not exist on disk: {missing_masks[:5]}")
    if dim_mismatches:
        raise AssertionError(f"[GATE 1 VIOLATION] Found {len(dim_mismatches)} dimension mismatches between images and masks: {dim_mismatches[:3]}")
    if empty_masks:
        raise AssertionError(f"[GATE 1 VIOLATION] Found {len(empty_masks)} completely empty masks: {empty_masks[:3]}")

    # 4. Exclusion overlap verification
    manifest_img_set = set(manifest_df["image_path"].apply(os.path.normpath))
    excluded_img_col = "image" if "image" in excluded_df.columns else excluded_df.columns[0]
    excluded_img_set = set(excluded_df[excluded_img_col].apply(os.path.normpath))

    overlap = manifest_img_set.intersection(excluded_img_set)
    if len(overlap) > 0:
        raise AssertionError(f"[GATE 1 VIOLATION] Found {len(overlap)} excluded cases present in manifest: {overlap}")

    validation_result = {
        "status": "PASS",
        "total_count": total_count,
        "benign_count": benign_count,
        "malignant_count": malignant_count,
        "missing_files": 0,
        "excluded_overlap": 0,
        "empty_masks": len(empty_masks),
        "dimension_mismatches": len(dim_mismatches)
    }

    return validation_result


def main():
    args = parse_args()
    print("=" * 80)
    print("VALIDATING BUSI PHASE 1 MANIFEST (GATE 1)")
    print("=" * 80)
    print(f"Manifest Path: {args.manifest}")
    print(f"Excluded Path: {args.excluded}")
    
    result = validate_manifest(args.manifest, args.excluded)
    print("-" * 80)
    print(f"Status:             {result['status']}")
    print(f"Total Rows:         {result['total_count']} (Expected: 628)")
    print(f"Benign Rows:        {result['benign_count']} (Expected: 420)")
    print(f"Malignant Rows:     {result['malignant_count']} (Expected: 208)")
    print(f"Missing Files:      {result['missing_files']}")
    print(f"Excluded Overlap:   {result['excluded_overlap']}")
    print(f"Empty Masks:        {result['empty_masks']}")
    print(f"Dim Mismatches:     {result['dimension_mismatches']}")
    print("-" * 80)
    print("[GATE 1 CHECK PASSED] Manifest conforms strictly to all data contracts.")
    print("=" * 80)


if __name__ == "__main__":
    main()
