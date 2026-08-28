"""
Plain U-Net Architecture for Binary Lesion Segmentation.

Design:
- Standard 2D Encoder-Decoder with Skip Connections (Ronneberger et al., 2015).
- Input: 1-channel Grayscale B-mode Ultrasound (B, 1, H, W).
- Output: 1-channel Raw Logits (B, 1, H, W).
- Configurable base channels (default: 64) and depth (default: 4 levels).
- Double convolution blocks: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU.
- MaxPool2d downsampling in encoder; ConvTranspose2d upsampling in decoder.
- No attention gates, no transformers, no CBAM (Plain baseline per PRD specification).
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Plain U-Net for binary medical image segmentation.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        ch_out = base_channels
        for i in range(depth):
            self.encoders.append(DoubleConv(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = ch_out
            ch_out = ch_out * 2

        # Bottleneck
        self.bottleneck = DoubleConv(ch_in, ch_out)

        # Decoder path
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth):
            # Upsample from ch_out to ch_in
            self.upconvs.append(nn.ConvTranspose2d(ch_out, ch_in, kernel_size=2, stride=2))
            # After concat with skip connection: ch_in (from upconv) + ch_in (skip) = ch_in * 2
            self.decoders.append(DoubleConv(ch_in * 2, ch_in))
            ch_out = ch_in
            ch_in = ch_in // 2

        # 1x1 Conv output head to produce logits
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, H, W).
        Returns:
            logits: Output tensor of shape (B, 1, H, W).
        """
        skips: List[torch.Tensor] = []

        # Encoder
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections (reversed)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle potential odd spatial dimension padding
            if x.shape != skip.shape:
                diff_y = skip.size(2) - x.size(2)
                diff_x = skip.size(3) - x.size(3)
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        logits = self.final_conv(x)
        return logits


def get_unet(config: dict) -> UNet:
    """Factory helper to build UNet from config dict."""
    model_cfg = config.get("model", {})
    in_channels = 1 if config.get("preprocessing", {}).get("grayscale", True) else 3
    out_channels = 1
    base_channels = model_cfg.get("base_channels", 64)
    depth = model_cfg.get("depth", 4)

    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
    )
