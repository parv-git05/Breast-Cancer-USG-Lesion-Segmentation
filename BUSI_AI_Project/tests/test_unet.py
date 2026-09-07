"""
Unit Tests for Plain U-Net Architecture.
"""

import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.unet import UNet, get_unet


def test_unet_forward_shape():
    model = UNet(in_channels=1, out_channels=1, base_channels=32, depth=3)
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_unet_gradient_flow():
    model = UNet(in_channels=1, out_channels=1, base_channels=32, depth=3)
    x = torch.randn(2, 1, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
