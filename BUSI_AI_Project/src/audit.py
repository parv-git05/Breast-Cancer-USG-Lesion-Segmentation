"""
Data Audit and Validation Module for Breast Ultrasound (BUSI) Dataset.

Performs read-only audit of raw data:
- Counts images and masks per class (benign, malignant, normal).
- Identifies multi-mask cases.
- Checks for empty masks (all 0 pixels).
- Validates image-mask spatial dimension parity.
- Validates expected absence of the deleted duplicate pair.
- Strictly read-only: NEVER modifies, renames, writes, or deletes raw data.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Audit BUSI dataset integrity and structure.")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="D:/Dataset_BUSI_with_GT",
        help="Path to raw BUSI dataset directory."
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="",
        help="Optional path to write audit summary JSON (outside raw data)."
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


def print_audit_report(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print("BUSI RAW DATASET AUDIT REPORT")
    print("=" * 80)
    print(f"Raw Root: {summary['raw_root']}")
    print(f"Duplicate Pair Audit Status: {summary['duplicate_pair_audit']['status']}")
    print("-" * 80)
    print(f"{'Class':<12} | {'Images':<8} | {'Masks':<8} | {'Single-Mask':<12} | {'Multi-Mask':<10}")
    print("-" * 80)
    for cls, stats in summary["classes"].items():
        print(f"{cls:<12} | {stats['image_count']:<8} | {stats['mask_count']:<8} | {stats['single_mask_count']:<12} | {stats['multi_mask_count']:<10}")
    print("-" * 80)
    print(f"Total Raw Images: {summary['total_images']}")
    print(f"Total Raw Masks:  {summary['total_masks']}")
    print(f"Multi-Mask Cases: {len(summary['multi_mask_cases'])}")
    print(f"Empty Masks:      {len(summary['empty_masks'])}")
    print(f"Dim Mismatches:   {len(summary['dimension_mismatches'])}")
    print("-" * 80)
    print(f"Phase 1 Eligible Single-Mask Cases:")
    print(f"  - Benign:    {summary['phase1_single_mask_candidates']['benign']}")
    print(f"  - Malignant: {summary['phase1_single_mask_candidates']['malignant']}")
    print(f"  - Total:     {summary['phase1_single_mask_candidates']['total']}")
    print("=" * 80)


def main():
    args = parse_args()
    summary = audit_raw_dataset(args.raw_root)
    print_audit_report(summary)
    
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Audit summary saved to: {args.json_out}")


if __name__ == "__main__":
    main()
