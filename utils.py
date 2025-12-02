import os
import random
from typing import Any, Optional

import numpy as np
import torch


def get_device(requested: Optional[str] = None) -> torch.device:
    """
    Resolve a torch.device without hard-coding CUDA.
    Priority: explicit argument -> CUDA -> MPS -> CPU.
    If the requested device is unavailable, falls back to CPU with a warning.
    """
    if requested:
        try:
            req = str(requested).lower()
            if req.startswith("cuda") and not torch.cuda.is_available():
                print(f"[WARN] Requested device '{requested}' unavailable; falling back to CPU.")
                return torch.device("cpu")
            if req.startswith("mps") and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print(f"[WARN] Requested device '{requested}' unavailable; falling back to CPU.")
                return torch.device("cpu")
            return torch.device(requested)
        except Exception:
            print(f"[WARN] Requested device '{requested}' is invalid; falling back to CPU.")
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(item: Any, device: torch.device) -> Any:
    """
    Recursively moves tensors inside nested structures to the target device.
    """
    if isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return type(item)(move_to_device(v, device) for v in item)
    if torch.is_tensor(item):
        return item.to(device)
    return item


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
