"""
Core utilities for the Breast Ultrasound AI Pipeline.
Provides deterministic seeding, device selection, YAML configuration loading,
checkpoint I/O, and runtime timing helpers.
"""

import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set seeds across Python standard library, NumPy, PyTorch CPU, and PyTorch CUDA.
    Enforces deterministic algorithm execution when configured.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Determine execution device based on configuration and hardware availability.
    Supported options: 'auto', 'cuda', 'cpu'.
    """
    pref = (device_preference or "auto").lower()
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif pref == "cpu":
        return torch.device("cpu")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.
    Validates that the file exists and is well-formed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config at {config_path}: root must be a mapping/dict.")
    
    return config


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint with full config payload and training metadata.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = dict(state)
    if config is not None:
        payload["config"] = config
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a model checkpoint onto specified device.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device is None:
        device = get_device("auto")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


class Timer:
    """
    Context manager and helper for timing operations and epochs.
    """
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time

    def get_elapsed_str(self) -> str:
        mins, secs = divmod(self.elapsed, 60)
        return f"{int(mins)}m {secs:.2f}s"
