"""
Smoke and Integrity Unit Tests for BUSI Data Pipeline and Quality Gates.

Validates:
- Quality Gate 1: Manifest counts, file existence, exclusion absence.
- Quality Gate 2: Split row counts, mutual disjointness, stratification.
- Quality Gate 3: Preprocessing, DataLoader shapes, tensor ranges, lack of NaN/Inf, binary mask values.
- Reproducibility & Synchronization: Paired transform sync and val/test determinism.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import BUSIDataset, get_dataloaders
from src.manifest import validate_manifest
from src.split import generate_splits
from src.transforms import DualTransformPipeline, build_transforms
from src.utils import load_config, set_seed


MANIFEST_PATH = "data/manifests/BUSI_phase1_manifest.csv"
EXCLUDED_PATH = "data/manifests/BUSI_phase1_excluded.csv"
CONFIG_PATH = "configs/split.yaml"


def test_gate1_manifest_contract():
    """Verify Quality Gate 1: 628 single-mask cases, 420 benign, 208 malignant, 0 missing files."""
    result = validate_manifest(MANIFEST_PATH, EXCLUDED_PATH)
    assert result["status"] == "PASS"
    assert result["total_count"] == 628
    assert result["benign_count"] == 420
    assert result["malignant_count"] == 208
    assert result["missing_files"] == 0
    assert result["excluded_overlap"] == 0
    assert result["empty_masks"] == 0
    assert result["dimension_mismatches"] == 0


def test_gate2_split_disjointness_and_counts():
    """Verify Quality Gate 2: Stratified splits are mutually disjoint and sum to 628."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    train_df, val_df, test_df, metadata = generate_splits(
        manifest_df=manifest_df,
        seed=42,
        ratios=(0.70, 0.15, 0.15),
        excluded_path=EXCLUDED_PATH
    )

    # Check total counts
    assert len(train_df) + len(val_df) + len(test_df) == 628

    # Check disjointness
    train_set = set(train_df["image_path"].apply(os.path.normpath))
    val_set = set(val_df["image_path"].apply(os.path.normpath))
    test_set = set(test_df["image_path"].apply(os.path.normpath))

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Check class ratios preservation
    overall_mal_ratio = (manifest_df["class"] == "malignant").mean()
    assert abs((train_df["class"] == "malignant").mean() - overall_mal_ratio) <= 0.02
    assert abs((val_df["class"] == "malignant").mean() - overall_mal_ratio) <= 0.02
    assert abs((test_df["class"] == "malignant").mean() - overall_mal_ratio) <= 0.02


def test_gate3_dataloader_smoke():
    """Verify Quality Gate 3: Tensor shapes, dtypes, ranges, and absence of NaNs."""
    config = load_config(CONFIG_PATH)
    splits_dir = config["data"]["splits_dir"]
    train_csv = os.path.join(splits_dir, "train.csv")

    if not os.path.exists(train_csv):
        pytest.skip("Splits not yet written to disk. Run src/split.py first.")

    train_loader, val_loader, test_loader = get_dataloaders(config)

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


def test_val_and_test_determinism():
    """Verify that validation and test datasets produce identical deterministic outputs."""
    config = load_config(CONFIG_PATH)
    splits_dir = config["data"]["splits_dir"]
    val_csv = os.path.join(splits_dir, "val.csv")

    if not os.path.exists(val_csv):
        pytest.skip("Val split not yet written to disk.")

    val_transform = build_transforms(config, is_train=False)
    ds = BUSIDataset(val_csv, transform=val_transform)

    # Fetch sample 0 twice and compare tensors
    sample_a = ds[0]
    sample_b = ds[0]

    assert torch.equal(sample_a["image"], sample_b["image"])
    assert torch.equal(sample_a["mask"], sample_b["mask"])


def test_paired_transform_synchronization():
    """Verify that geometric augmentations apply identically and synchronously to image and mask."""
    # Create a synthetic image with a recognizable quadrant marker
    img_arr = np.zeros((100, 100), dtype=np.uint8)
    mask_arr = np.zeros((100, 100), dtype=np.uint8)
    
    # Place a square in the top-left quadrant (rows 10..30, cols 10..30)
    img_arr[10:30, 10:30] = 255
    mask_arr[10:30, 10:30] = 255

    img_pil = Image.fromarray(img_arr, mode="L")
    mask_pil = Image.fromarray(mask_arr, mode="L")

    aug_cfg = {
        "hflip_prob": 1.0,  # Force horizontal flip
        "vflip_prob": 1.0,  # Force vertical flip
        "rotation_deg": 0.0,
        "brightness": 0.0,
        "contrast": 0.0
    }
    pipeline = DualTransformPipeline(image_size=(100, 100), is_train=True, augmentation_cfg=aug_cfg)
    
    img_t, mask_t = pipeline(img_pil, mask_pil)
    
    # Both image and mask should now have their active region flipped to bottom-right
    img_np = img_t.squeeze(0).numpy()
    mask_np = mask_t.squeeze(0).numpy()

    # Image active region (normalized 1.0) and Mask active region (1.0) must align exactly
    assert np.all((img_np > 0.5) == (mask_np > 0.5))
    assert mask_np[70:90, 70:90].sum() > 0  # Flipped to bottom right
