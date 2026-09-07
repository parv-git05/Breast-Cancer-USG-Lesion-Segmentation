"""
Smoke and Integrity Unit Tests for BUSI Data Pipeline and Quality Gates.

Validates:
- Quality Gate 1: Manifest counts, file existence, exclusion absence.
- Quality Gate 2: Two-stage split (63 locked test + 565 dev, 3 dev CV folds, mutual disjointness, stratification).
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
from src.dataset import BUSIDataset, get_development_dataloader, get_fold_dataloaders
from src.manifest import validate_manifest
from src.split import generate_exp_seg_02_splits
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


def test_gate2_exp_seg_02_disjointness_and_counts():
    """Verify Quality Gate 2: Stratified holdout test (63) and 3-fold dev CV (565) are disjoint."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    dev_df, test_df, folds_data, metadata = generate_exp_seg_02_splits(
        manifest_df=manifest_df,
        seed=42,
        test_size=63,
        n_splits=3,
        raw_root="..",
        manifests_dir="data/manifests",
        folds_dir="data/folds/exp_seg_02",
        test_dir="data/test_set",
        excluded_path=EXCLUDED_PATH
    )

    assert len(test_df) == 63
    assert len(dev_df) == 565
    assert len(folds_data) == 3

    test_imgs = set(test_df["image_path"].apply(os.path.normpath))
    dev_imgs = set(dev_df["image_path"].apply(os.path.normpath))

    # Test and Dev disjointness
    assert len(test_imgs.intersection(dev_imgs)) == 0

    all_val_imgs = set()
    for fold_idx, (train_df, val_df) in enumerate(folds_data, start=1):
        tr_set = set(train_df["image_path"].apply(os.path.normpath))
        va_set = set(val_df["image_path"].apply(os.path.normpath))

        # Intra-fold disjointness
        assert len(tr_set.intersection(va_set)) == 0
        assert len(tr_set) + len(va_set) == 565

        # Zero test leakage
        assert len(tr_set.intersection(test_imgs)) == 0
        assert len(va_set.intersection(test_imgs)) == 0

        # Inter-fold disjointness
        assert len(all_val_imgs.intersection(va_set)) == 0
        all_val_imgs.update(va_set)

    # 100% development coverage across folds
    assert len(all_val_imgs) == 565


def test_gate3_dataloader_smoke():
    """Verify Quality Gate 3: Tensor shapes, dtypes, ranges, and absence of NaNs."""
    config = load_config(CONFIG_PATH)
    folds_dir = config.get("data", {}).get("folds_dir", "data/folds/exp_seg_02")
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
    folds_dir = config.get("data", {}).get("folds_dir", "data/folds/exp_seg_02")
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


def test_development_dataloader_hard_gate():
    """Verify pre-training hard data gate for final model: 565 development cases, 378 benign, 187 malignant, 0 test leakage."""
    config = load_config("configs/exp_seg_02.yaml")
    dev_loader, dev_df = get_development_dataloader(config)

    assert len(dev_df) == 565
    assert int((dev_df["class"] == "benign").sum()) == 378
    assert int((dev_df["class"] == "malignant").sum()) == 187
    assert len(dev_loader.dataset) == 565

    # Check 1 batch from development loader
    batch = next(iter(dev_loader))
    assert batch["image"].shape == (8, 1, 256, 256)
    assert batch["mask"].shape == (8, 1, 256, 256)
    assert not torch.isnan(batch["image"]).any()
    assert not torch.isinf(batch["image"]).any()

