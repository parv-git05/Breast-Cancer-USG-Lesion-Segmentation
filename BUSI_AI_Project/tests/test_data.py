"""
Smoke and Integrity Unit Tests for BUSI Data Pipeline and Quality Gates.

Validates:
- Quality Gate 1: Manifest counts, file existence, exclusion absence.
- Quality Gate 2: 3-Fold CV fold counts, complete coverage, mutual disjointness, stratification.
- Quality Gate 3: Preprocessing, DataLoader shapes, tensor ranges, lack of NaN/Inf, binary mask values.
- Reproducibility & Synchronization: Paired transform sync and val determinism.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import BUSIDataset, get_fold_dataloaders
from src.manifest import validate_manifest
from src.split import generate_3fold_cv_splits
from src.transforms import DualTransformPipeline, build_transforms
from src.utils import load_config, set_seed


MANIFEST_PATH = "data/manifests/BUSI_phase1_manifest.csv"
EXCLUDED_PATH = "data/manifests/BUSI_phase1_excluded.csv"
CONFIG_PATH = "configs/split.yaml"


def test_gate1_manifest_contract():
    """Verify Quality Gate 1: 628 single-mask cases, 420 benign, 208 malignant, 0 missing files."""
    config = load_config(CONFIG_PATH)
    raw_root = config.get("data", {}).get("raw_root", "")
    result = validate_manifest(MANIFEST_PATH, EXCLUDED_PATH, raw_root=raw_root)
    assert result["status"] == "PASS"
    assert result["total_count"] == 628
    assert result["benign_count"] == 420
    assert result["malignant_count"] == 208
    assert result["missing_files"] == 0
    assert result["excluded_overlap"] == 0
    assert result["empty_masks"] == 0
    assert result["dimension_mismatches"] == 0


def test_gate2_3fold_cv_disjointness_and_counts():
    """Verify Quality Gate 2: Stratified 3-fold CV splits are mutually disjoint and sum to 628."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    folds_data, metadata = generate_3fold_cv_splits(
        manifest_df=manifest_df,
        seed=42,
        n_splits=3,
        folds_dir="data/folds",
        excluded_path=EXCLUDED_PATH
    )

    assert len(folds_data) == 3
    all_val_imgs = set()
    overall_mal_ratio = (manifest_df["class"] == "malignant").mean()

    for fold_idx, (train_df, val_df) in enumerate(folds_data, start=1):
        # Intra-fold check
        train_set = set(train_df["image_path"].apply(os.path.normpath))
        val_set = set(val_df["image_path"].apply(os.path.normpath))
        
        assert len(train_set.intersection(val_set)) == 0, f"Fold {fold_idx} train and val overlap!"
        assert len(train_set) + len(val_set) == 628, f"Fold {fold_idx} total count is not 628!"

        # Inter-fold disjoint validation check
        assert len(all_val_imgs.intersection(val_set)) == 0, f"Fold {fold_idx} validation overlaps with previous folds!"
        all_val_imgs.update(val_set)

        # Class ratio check (+- 2%)
        val_mal_ratio = (val_df["class"] == "malignant").mean()
        assert abs(val_mal_ratio - overall_mal_ratio) <= 0.02, f"Fold {fold_idx} class ratio deviation > 2%"

    # Total unique validation images must equal 628
    assert len(all_val_imgs) == 628, f"Total unique validation images = {len(all_val_imgs)}, expected 628."


def test_gate3_dataloader_smoke():
    """Verify Quality Gate 3: Tensor shapes, dtypes, ranges, and absence of NaNs."""
    config = load_config(CONFIG_PATH)
    folds_dir = config.get("data", {}).get("folds_dir", "data/folds")
    fold1_train_csv = os.path.join(folds_dir, "fold_1_train.csv")

    if not os.path.exists(fold1_train_csv):
        pytest.skip("Folds not yet written to disk. Run src/split.py first.")

    train_loader, val_loader = get_fold_dataloaders(config, fold_idx=1)

    # Fetch 1 batch from train loader
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    class_ids = batch["class_id"]

    # Assert shapes: (B, 1, 256, 256)
    expected_size = tuple(config["data"]["image_size"])
    assert images.dim() == 4
    assert images.shape[1] == 1
    assert images.shape[2:] == expected_size
    assert masks.shape == images.shape

    # Assert values
    assert not torch.isnan(images).any()
    assert not torch.isinf(images).any()
    assert images.min() >= 0.0
    assert images.max() <= 1.0

    # Mask values must strictly be {0.0, 1.0}
    unique_mask_vals = torch.unique(masks).tolist()
    for v in unique_mask_vals:
        assert v in [0.0, 1.0]

    # Check class IDs {0, 1}
    assert torch.all((class_ids == 0) | (class_ids == 1))


def test_val_determinism():
    """Verify that validation datasets produce identical deterministic outputs."""
    config = load_config(CONFIG_PATH)
    folds_dir = config.get("data", {}).get("folds_dir", "data/folds")
    raw_root = config.get("data", {}).get("raw_root", "")
    val_csv = os.path.join(folds_dir, "fold_1_val.csv")

    if not os.path.exists(val_csv):
        pytest.skip("Val fold not yet written to disk.")

    val_transform = build_transforms(config, is_train=False)
    ds = BUSIDataset(val_csv, transform=val_transform, raw_root=raw_root)

    # Fetch sample 0 twice and compare tensors
    sample_a = ds[0]
    sample_b = ds[0]

    assert torch.equal(sample_a["image"], sample_b["image"])
    assert torch.equal(sample_a["mask"], sample_b["mask"])


def test_paired_transform_synchronization():
    """Verify that geometric augmentations apply identically and synchronously to image and mask."""
    img_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    
    img_arr[10:30, 10:30] = 255
    mask_arr[10:30, 10:30] = 255

    img_pil = Image.fromarray(img_arr, mode="L")
    mask_pil = Image.fromarray(mask_arr, mode="L")

    aug_cfg = {
        "hflip_prob": 1.0,
        "vflip_prob": 1.0,
        "rotation_deg": 0.0,
        "brightness": 0.0,
        "contrast": 0.0
    }
    pipeline = DualTransformPipeline(image_size=(100, 100), is_train=True, augmentation_cfg=aug_cfg)
    
    img_t, mask_t = pipeline(img_pil, mask_pil)
    
    img_np = img_t.squeeze(0).numpy()
    mask_np = mask_t.squeeze(0).numpy()

    assert np.all((img_np > 0.5) == (mask_np > 0.5))
    assert mask_np[70:90, 70:90].sum() > 0
