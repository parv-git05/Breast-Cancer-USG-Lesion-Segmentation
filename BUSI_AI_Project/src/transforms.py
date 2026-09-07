"""
Preprocessing and Augmentation Transforms for Breast Ultrasound AI Pipeline.

Features:
- Dual-target synchronous transformations for paired Image and Binary Mask.
- Image resize using Bilinear interpolation.
- Mask resize using NEAREST-NEIGHBOR ONLY to preserve crisp binary values.
- Grayscale conversion.
- Intensity normalization: default 'minmax' ([0, 1]) without dataset leakage.
- Training augmentations: synchronized horizontal flip, vertical flip, rotation, brightness/contrast jitter.
- Val/test transforms: strictly deterministic (resize + normalize, no random ops).
"""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance


class DualTransformPipeline:
    """
    Applies paired, synchronous transformations to (image, mask) PIL/Tensor pairs.
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        augmentation_cfg: Optional[Dict[str, Any]] = None,
        normalize_method: str = "minmax",
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.image_size = tuple(image_size)
        self.is_train = is_train
        self.aug_cfg = augmentation_cfg or {}
        self.normalize_method = normalize_method.lower()
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: PIL Image (B-mode ultrasound)
            mask:  PIL Image (Binary ground-truth mask)
        Returns:
            image_tensor: torch.FloatTensor of shape (1, H, W)
            mask_tensor:  torch.FloatTensor of shape (1, H, W) with values in {0.0, 1.0}
        """
        # Ensure Grayscale ('L')
        if image.mode != "L":
            image = image.convert("L")
        if mask.mode != "L":
            mask = mask.convert("L")

        # 1. Resize: Image via Bilinear, Mask strictly via NEAREST NEIGHBOR
        image = image.resize(self.image_size, resample=Image.Resampling.BILINEAR)
        mask = mask.resize(self.image_size, resample=Image.Resampling.NEAREST)

        # 2. Synchronous Augmentation for Training only
        if self.is_train and self.aug_cfg:
            # Horizontal Flip (synchronized)
            hflip_prob = self.aug_cfg.get("hflip_prob", 0.0)
            if hflip_prob > 0 and random.random() < hflip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Vertical Flip (synchronized)
            vflip_prob = self.aug_cfg.get("vflip_prob", 0.0)
            if vflip_prob > 0 and random.random() < vflip_prob:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            # Random Rotation (synchronized identical angle)
            rot_deg = self.aug_cfg.get("rotation_deg", 0.0)
            if rot_deg > 0:
                angle = random.uniform(-rot_deg, rot_deg)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

            # Photometric Jitter (Image ONLY, does not modify mask geometry)
            brightness = self.aug_cfg.get("brightness", 0.0)
            if brightness > 0:
                b_factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
                image = ImageEnhance.Brightness(image).enhance(b_factor)

            contrast = self.aug_cfg.get("contrast", 0.0)
            if contrast > 0:
                c_factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
                image = ImageEnhance.Contrast(image).enhance(c_factor)

        # 3. Convert to NumPy / Tensor
        img_arr = np.array(image, dtype=np.float32)
        msk_arr = np.array(mask, dtype=np.float32)

        # 4. Normalization (Default: minmax [0, 1])
        if self.normalize_method == "minmax":
            img_arr = img_arr / 255.0
        elif self.normalize_method == "zscore":
            img_arr = img_arr / 255.0
            mean = self.mean if self.mean is not None else 0.5
            std = self.std if (self.std is not None and self.std > 1e-6) else 0.5
            img_arr = (img_arr - mean) / std
        else:
            img_arr = img_arr / 255.0

        # Mask strictly thresholded to binary {0.0, 1.0}
        msk_arr = (msk_arr > 127.5).astype(np.float32)

        # Convert to Torch Tensors with channel dimension (1, H, W)
        image_tensor = torch.from_numpy(img_arr).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(msk_arr).unsqueeze(0)   # (1, H, W)

        return image_tensor, mask_tensor


def build_transforms(
    config: Dict[str, Any],
    is_train: bool = True,
    train_mean: Optional[float] = None,
    train_std: Optional[float] = None,
) -> DualTransformPipeline:
    """
    Factory function to construct DualTransformPipeline from experiment YAML config.
    """
    image_size = config.get("data", {}).get("image_size", [256, 256])
    normalize_method = config.get("preprocessing", {}).get("normalize", "minmax")
    
    use_aug = config.get("preprocessing", {}).get("augmentation", True) if is_train else False
    aug_cfg = config.get("augmentation", {}) if use_aug else {}

    return DualTransformPipeline(
        image_size=image_size,
        is_train=is_train,
        augmentation_cfg=aug_cfg,
        normalize_method=normalize_method,
        mean=train_mean,
        std=train_std,
    )
