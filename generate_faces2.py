# John Cosby
# Image Generation Module
# Generate faces from latent noise using trained weights.
#
# Usage (example):
#   python generate_faces.py \
#       --checkpoint models/vae_latest.pt \
#       --num_samples 16 \
#       --noise_dim 512 \
#       --out_dir outputs/generated
#
# This script:
#   - Loads a trained model checkpoint saved by Erick
#   - Samples random latent noise z ~ N(0, I)
#   - Calls model.generate(z) to produce images in [-1, 1]
#   - Saves a grid + individual PNGs into outputs/generated/
#
# The saved images are picked up automatically by:
#   - Da Marc's dashboard.py (Latest Generated Samples panel)

import argparse
import math
import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from pathlib import Path
import torch
import logging

logging.getLogger(__name__).setLevel(logging.INFO)

def _looks_like_state_dict(d: dict) -> bool:
    # heuristic: many keys and values are tensors / numpy arrays
    if not isinstance(d, dict) or len(d) == 0:
        return False
    sample = list(d.values())[:8]
    return all(hasattr(v, "shape") for v in sample if not isinstance(v, (int, float, str, bytes)))

def load_model_from_checkpoint(checkpoint_path, device=None, nz=100):
    """
    Robust loader:
    - finds nested state dicts under common keys
    - strips 'module.' prefixes
    - tries strict=False, then falls back to loading matching param keys by name+shape
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(nz=nz).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # collect candidate state dicts
    candidates = []
    if isinstance(ckpt, dict):
        for key in ("generator_state_dict", "model_state_dict", "state_dict", "model", "netG"):
            if key in ckpt and isinstance(ckpt[key], dict):
                candidates.append(ckpt[key])
        # also look for any dict-valued entry that resembles a state_dict
        for v in ckpt.values():
            if isinstance(v, dict) and _looks_like_state_dict(v):
                candidates.append(v)
    else:
        candidates.append(ckpt)

    # pick first plausible candidate
    state = None
    for cand in candidates:
        if isinstance(cand, dict):
            state = cand
            break
    if state is None:
        raise RuntimeError(f"No valid state_dict found in checkpoint: {checkpoint_path}")

    # normalize keys (remove 'module.' or 'net.' prefixes if present)
    norm_state = {}
    for k, v in state.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        # some checkpoints from other code use "net.[...]" vs "net.[...]" -> keep as-is
        norm_state[nk] = v

    # attempt best-effort load
    try:
        G.load_state_dict(norm_state, strict=False)
        logging.info("Loaded checkpoint into Generator (strict=False).")
        return G.eval()
    except Exception as e:
        logging.warning("load_state_dict(strict=False) failed: %s", e)

    # fallback: only keep keys that match in name and shape
    g_state = G.state_dict()
    filtered = {}
    for k, v in norm_state.items():
        if k in g_state and hasattr(v, "shape") and g_state[k].shape == v.shape:
            filtered[k] = v

    if not filtered:
        # helpful debug: list top-level keys found
        raise RuntimeError(f"No compatible parameter keys found between checkpoint and model. "
                           f"Checkpoint top keys: {list(state.keys())[:30]}")

    # merge and load
    merged = {**g_state}
    merged.update(filtered)
    G.load_state_dict(merged)
    logging.info("Loaded matching subset of checkpoint keys into Generator.")
    return G.eval()


def sample_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    """Sample latent noise z ~ N(0, I). Shape: [B, noise_dim]."""
    return torch.randn(batch_size, noise_dim, device=device)


def generate_from_model(G, num_images=8, nz=100, device=None, seed=None):
    """
    Generate num_images PIL images from G.
    Returns list of PIL.Image objects (RGB).
    """
    if device is None:
        device = next(G.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(num_images, nz, 1, 1, device=device)
        imgs = G(z)  # BxCxHxW in [-1,1]
    arr = _denorm_tensor(imgs)  # B,H,W,C uint8
    pil_images = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
    return pil_images


def denormalize_to_unit_range(imgs: torch.Tensor) -> torch.Tensor:
    """
    Convert from [-1, 1] to [0, 1] and clamp.

    This keeps consistency with:
      - Tycho's loader which normalizes to [-1, 1]
      - Dashboard which displays 0..255 images after rescaling
    """
    imgs = (imgs + 1.0) / 2.0
    return imgs.clamp(0.0, 1.0)


def save_generated_images(imgs: torch.Tensor, out_dir: Path, prefix: str = "gen"):
    """
    Save both:
      - A single grid image (prefix_grid.png)
      - Individual PNGs (prefix_000.png, prefix_001.png, ...)

    The dashboard reads from outputs/generated/ and shows the latest images.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save grid
    n = imgs.size(0)
    grid_cols = int(math.ceil(math.sqrt(n)))
    grid_path = out_dir / f"{prefix}_grid.png"
    save_image(imgs, grid_path, nrow=grid_cols)
    print(f"[INFO] Saved grid image: {grid_path}")

    # Save individual samples
    for i in range(n):
        img_path = out_dir / f"{prefix}_{i:03d}.png"
        save_image(imgs[i], img_path)
    print(f"[INFO] Saved {n} individual images to: {out_dir}")


# ---- CLI / Main -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate faces from latent noise using trained weights."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=None,
        help="Dimensionality of the latent noise. "
             "If omitted, will try to read from checkpoint['noise_dim'], "
             "otherwise falls back to 512.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/generated",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = _pick_device(force_cpu=args.cpu)
    print(f"[INFO] Using device: {device}")

    print(f"[INFO] Loading model from checkpoint: {checkpoint_path}")
    model, ckpt_noise_dim = load_model_from_checkpoint(checkpoint_path, device)

    # Decide noise dimension: CLI > checkpoint > default 512
    noise_dim = args.noise_dim or ckpt_noise_dim or 512
    if args.noise_dim and ckpt_noise_dim and args.noise_dim != ckpt_noise_dim:
        print(
            f"[WARN] noise_dim={args.noise_dim} "
            f"differs from checkpoint noise_dim={ckpt_noise_dim}. "
            "Using CLI value."
        )
    print(f"[INFO] Noise dimension: {noise_dim}")

    print(f"[INFO] Generating {args.num_samples} samples...")
    raw_imgs = generate_from_model(model, args.num_samples, noise_dim, device=device)

    # Convert [-1,1] -> [0,1] before saving as PNG
    imgs = denormalize_to_unit_range(raw_imgs)

    out_dir = Path(args.out_dir)
    save_generated_images(imgs, out_dir, prefix="gen")

    print("[SUCCESS] Image generation complete.")


# move any demo / script-level code into a guard (see data_loader patch below)
if __name__ == "__main__":
    main()
