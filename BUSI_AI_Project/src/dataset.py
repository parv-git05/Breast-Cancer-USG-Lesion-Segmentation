"""
PyTorch Dataset and DataLoader Module for Breast Ultrasound (BUSI) Dataset.

Key Capabilities:
- Robust image & binary mask loading with file existence & integrity checks.
- Dual transformation integration.
- Graceful validation: flags empty masks, verifies binary range {0.0, 1.0}, checks dimension parity.
- Leakage-safe DataLoader constructors:
    * Deterministic non-shuffled loaders for val and test splits.
    * Train-only dataset statistics computation (mean/std) when needed.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transforms import DualTransformPipeline, build_transforms


class BUSIDataset(Dataset):
    """
    PyTorch Dataset wrapping a BUSI Phase 1 split CSV or DataFrame.
    """
    def __init__(
        self,
        split_source: Union[str, pd.DataFrame],
        transform: Optional[Callable] = None,
        allow_empty_masks: bool = False,
        raw_root: str = "",
    ):
        """
        Args:
            split_source: Path to split CSV (train.csv / val.csv / test.csv) or DataFrame.
            transform: DualTransformPipeline callable taking (image_pil, mask_pil).
            allow_empty_masks: If False, raises ValueError when a mask has 0 lesion pixels.
            raw_root: Directory prefix for image and mask paths.
        """
        if isinstance(split_source, str):
            if not os.path.exists(split_source):
                alt = os.path.join(os.path.dirname(__file__), "..", split_source)
                if os.path.exists(alt):
                    split_source = alt
                else:
                    raise FileNotFoundError(f"Split CSV file not found: {split_source}")
            self.df = pd.read_csv(split_source)
        elif isinstance(split_source, pd.DataFrame):
            self.df = split_source.copy().reset_index(drop=True)
        else:
            raise TypeError(f"split_source must be a path (str) or pandas DataFrame, got {type(split_source)}")

        self.transform = transform
        self.allow_empty_masks = allow_empty_masks
        self.raw_root = raw_root

        # Pre-verify file existence
        self._verify_files()

    def _verify_files(self) -> None:
        required_cols = ["image_path", "mask_path"]
        for col in required_cols:
            if col not in self.df.columns:
                raise KeyError(f"Missing required column '{col}' in dataset dataframe.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        mask_path = str(row["mask_path"])
        
        if self.raw_root:
            img_path = os.path.join(self.raw_root, img_path)
            mask_path = os.path.join(self.raw_root, mask_path)

        class_name = str(row.get("class", "unknown"))
        class_id = int(row.get("class_id", 0 if class_name == "benign" else 1))

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Ultrasound image not found: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Ultrasound mask not found: {mask_path}")

        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            raise IOError(f"Corrupt or unreadable image file at {img_path}: {e}")

        try:
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            raise IOError(f"Corrupt or unreadable mask file at {mask_path}: {e}")

        # Check native dimension parity
        if image.size != mask.size:
            raise ValueError(
                f"Dimension mismatch between image ({image.size}) and mask ({mask.size}) for {img_path}"
            )

        # Apply dual transformation
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
        else:
            fallback_pipe = DualTransformPipeline(image_size=(256, 256), is_train=False)
            image_tensor, mask_tensor = fallback_pipe(image, mask)

        # Validate mask values {0.0, 1.0}
        unique_vals = torch.unique(mask_tensor)
        for v in unique_vals:
            if not (torch.isclose(v, torch.tensor(0.0)) or torch.isclose(v, torch.tensor(1.0))):
                raise ValueError(
                    f"Non-binary values detected in mask tensor for {img_path}: {unique_vals.tolist()}"
                )

        # Check for empty mask if disallowed
        if not self.allow_empty_masks and mask_tensor.sum() == 0:
            raise ValueError(f"Empty mask detected (0 nonzero pixels) for single-mask lesion case: {img_path}")

        # Check for NaN / Inf in image tensor
        if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
            raise ValueError(f"NaN or Inf encountered in preprocessed image tensor: {img_path}")

        return {
            "image": image_tensor,          # Shape: (1, H, W), float32
            "mask": mask_tensor,            # Shape: (1, H, W), float32 {0.0, 1.0}
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "class_name": class_name,
            "image_path": img_path,
            "mask_path": mask_path,
        }


def compute_train_statistics(train_csv_path: str, image_size: Tuple[int, int] = (256, 256)) -> Tuple[float, float]:
    """
    Compute mean and standard deviation of grayscale images strictly on the training set.
    Prevents data leakage into validation or test distributions.
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training split CSV not found: {train_csv_path}")

    train_df = pd.read_csv(train_csv_path)
    pixel_values = []

    for img_p in train_df["image_path"]:
        with Image.open(img_p).convert("L") as img:
            resized = img.resize(image_size, resample=Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.float32) / 255.0
            pixel_values.append(arr)

    all_pixels = np.concatenate([p.flatten() for p in pixel_values])
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))
    return mean, std


def get_fold_dataloaders(
    config: Dict[str, Any],
    fold_idx: int,
    train_mean: Optional[float] = None,
    train_std: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct DataLoaders for a specific cross-validation fold (train and val).
    Zero test leakage: test set is NOT loaded or evaluated.
    """
    folds_dir = config.get("data", {}).get("folds_dir", "data/folds")
    train_csv = os.path.join(folds_dir, f"fold_{fold_idx}_train.csv")
    val_csv = os.path.join(folds_dir, f"fold_{fold_idx}_val.csv")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Fold {fold_idx} training CSV not found at: {train_csv}")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Fold {fold_idx} validation CSV not found at: {val_csv}")

    batch_size = config.get("training", {}).get("batch_size", 8)
    num_workers = config.get("hardware", {}).get("num_workers", 0)
    raw_root = config.get("data", {}).get("raw_root", "")

    # Build transforms: augmentations ONLY for training, deterministic for validation
    train_transform = build_transforms(config, is_train=True, train_mean=train_mean, train_std=train_std)
    eval_transform = build_transforms(config, is_train=False, train_mean=train_mean, train_std=train_std)

    train_dataset = BUSIDataset(train_csv, transform=train_transform, raw_root=raw_root)
    val_dataset = BUSIDataset(val_csv, transform=eval_transform, raw_root=raw_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


def get_dataloaders(
    config: Dict[str, Any],
    fold_idx: int = 1,
    train_mean: Optional[float] = None,
    train_std: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Backward compatible wrapper delegating to get_fold_dataloaders."""
    return get_fold_dataloaders(config, fold_idx=fold_idx, train_mean=train_mean, train_std=train_std)

