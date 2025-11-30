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
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
from pathlib import Path
import torch
import logging
import sys

logging.getLogger(__name__).setLevel(logging.INFO)

def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or len(d) == 0:
        return False
    sample = list(d.values())[:8]
    return all(hasattr(v, "shape") for v in sample if not isinstance(v, (int, float, str, bytes)))

def _extract_state_dict(ckpt: dict):
    # common candidate keys
    for key in ("generator_state_dict", "model_state_dict", "state_dict", "netG", "net", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            cand = ckpt[key]
            # sometimes the nested dict itself wraps the real state under another key
            if any(k in cand for k in ("state_dict", "model_state_dict", "net", "generator_state_dict")):
                return _extract_state_dict(cand)
            return cand
    # otherwise look for any nested dict that looks like a state_dict
    for v in ckpt.values():
        if isinstance(v, dict) and _looks_like_state_dict(v):
            return v
    # fallback: if the top-level looks like a state_dict, return it
    if _looks_like_state_dict(ckpt):
        return ckpt
    return None

def load_model_from_checkpoint(checkpoint_path, device=None, nz=100):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(nz=nz).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = None
    if isinstance(ckpt, dict):
        state = _extract_state_dict(ckpt)
    else:
        state = ckpt

    if state is None:
        raise RuntimeError(f"No valid state_dict found in checkpoint: {checkpoint_path}. Top keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # normalize keys (remove "module." prefix)
    norm_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        norm_state[nk] = v

    # try best-effort load
    try:
        G.load_state_dict(norm_state, strict=False)
        logging.info("Loaded checkpoint into Generator with strict=False.")
        return G.eval()
    except Exception as e:
        logging.warning("load_state_dict(strict=False) failed: %s", e)

    # fallback: load only matching keys by name+shape
    g_state = G.state_dict()
    filtered = {}
    for k, v in norm_state.items():
        if k in g_state and hasattr(v, "shape") and g_state[k].shape == v.shape:
            filtered[k] = v

    if not filtered:
        raise RuntimeError(f"No compatible parameter keys found between checkpoint and model. Checkpoint top keys: {list(state.keys())[:40]}")

    merged = {**g_state}
    merged.update(filtered)
    G.load_state_dict(merged)
    logging.info("Loaded matching subset of checkpoint keys into Generator.")
    return G.eval()


def sample_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    """Sample latent noise z ~ N(0, I). Shape: [B, noise_dim]."""
    return torch.randn(batch_size, noise_dim, device=device)


def _robust_denorm_to_uint8(imgs: torch.Tensor) -> np.ndarray:
    """
    Convert generator output (B,C,H,W) with unknown range to uint8 HWC numpy array.
    Heuristics:
      - If float and range looks like [-1,1] => map to [0,255]
      - If float and range looks like [0,1] => map to [0,255]
      - If already uint8 => pass through
      - If single-channel => repeat to 3 channels
    Returns: np.ndarray shape (B,H,W,3) dtype uint8
    """
    if not isinstance(imgs, torch.Tensor):
        raise TypeError("expected torch.Tensor")

    # move to CPU float for inspection
    t = imgs.detach().cpu()
    B, C, H, W = t.shape
    info = f"tensor shape={t.shape} dtype={t.dtype} min={t.min().item():.4f} max={t.max().item():.4f} mean={t.mean().item():.4f}"
    logging.info("[DENORM] %s", info)
    # If already uint8
    if t.dtype == torch.uint8:
        arr = t.permute(0, 2, 3, 1).numpy()
    else:
        t = t.float()
        mn = float(t.min().item())
        mx = float(t.max().item())

        if mn >= -1.1 and mx <= 1.1:
            # assume tanh [-1,1]
            t = (t + 1.0) / 2.0
            logging.info("[DENORM] Assuming generator output in [-1,1] (tanh).")
        elif mn >= -0.01 and mx <= 1.01:
            # assume sigmoid / [0,1]
            logging.info("[DENORM] Assuming generator output in [0,1] (sigmoid).")
        elif mx > 1.0 and mx <= 255.0 and mn >= 0.0:
            # already 0..255 float
            t = t / 255.0
            logging.info("[DENORM] Generator output appears to be in 0..255 floats; scaling down.")
        else:
            # fallback: scale using min/max to [0,1]
            logging.warning("[DENORM] Unexpected range (min=%f max=%f). Applying linear min/max rescale.", mn, mx)
            t = (t - mn) / (mx - mn + 1e-8)

        t = t.clamp(0.0, 1.0)
        t = (t * 255.0).round().to(torch.uint8)
        arr = t.permute(0, 2, 3, 1).numpy()

    # Ensure 3 channels for PIL display
    if arr.shape[3] == 1:
        arr = np.repeat(arr, 3, axis=3)
    elif arr.shape[3] == 4:
        # keep RGBA but convert to RGB by discarding alpha for dashboard
        arr = arr[:, :, :, :3]

    return arr

def generate_from_model(G, num_images=8, nz=100, device=None, seed=None):
    """
    Generate num_images images from G and return a torch.Tensor BxCxHxW (float32, range approx [-1,1]).
    Heuristics:
      - Try conv-style z (B,nz,1,1). If output looks degenerate, retry with dense z (B,nz).
      - Ensure model in eval() and on correct device.
    """
    if device is None:
        device = next(G.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)

    G.eval()
    with torch.no_grad():
        B = int(num_images)
        # try conv z first (common for DCGAN-style generators)
        z_conv = torch.randn(B, nz, 1, 1, device=device)
        out = None
        try:
            out = G(z_conv)
        except Exception as e:
            logging.info("Generator rejected conv-z shape: %s. Will try dense-z. Err: %s", z_conv.shape, e)

        # if conv attempt failed or produced unexpected shape, try dense z
        if out is None or (isinstance(out, torch.Tensor) and out.dim() == 2):
            z_dense = torch.randn(B, nz, device=device)
            try:
                out = G(z_dense)
            except Exception as e:
                logging.error("Generator failed for dense z too: %s", e)
                raise

        if not isinstance(out, torch.Tensor):
            raise TypeError("Generator did not return a torch.Tensor. Got: %s" % type(out))

        # ensure shape is BxCxHxW
        if out.dim() == 4:
            imgs = out
        elif out.dim() == 3:
            # maybe returned CxHxW for single sample
            imgs = out.unsqueeze(0)
        else:
            raise RuntimeError(f"Unexpected generator output shape: {tuple(out.shape)}")

        # move to CPU for inspection but keep a float copy for downstream
        t = imgs.detach().cpu().float()
        B2, C, H, W = t.shape
        logging.info("Generator raw output shape=%s dtype=%s min=%.4f max=%.4f",
                     tuple(t.shape), str(t.dtype), float(t.min()), float(t.max()))

        # if single-channel, expand to 3 channels
        if C == 1:
            logging.info("Generator returned single channel; repeating to RGB.")
            t = t.repeat(1, 3, 1, 1)
            C = 3
        elif C == 4:
            logging.info("Generator returned 4 channels; dropping alpha.")
            t = t[:, :3, :, :]
            C = 3

        mn = float(t.min().item())
        mx = float(t.max().item())

        # Normalize output to approx [-1,1] for downstream consistency
        if mn >= -1.1 and mx <= 1.1:
            # already approx [-1,1]
            out_t = t
            logging.info("Assuming generator outputs in [-1,1].")
        elif mn >= -0.01 and mx <= 1.01:
            # [0,1] -> [-1,1]
            out_t = (t * 2.0) - 1.0
            logging.info("Assuming generator outputs in [0,1], converting to [-1,1].")
        elif mn >= 0.0 and mx > 1.0 and mx <= 255.0:
            # 0..255 -> [-1,1]
            out_t = (t / 255.0) * 2.0 - 1.0
            logging.info("Assuming generator outputs in [0,255], converting to [-1,1].")
        else:
            # fallback linear rescale to [-1,1]
            logging.warning("Generator output had unexpected range (min=%f max=%f). Linearly rescaling to [-1,1].", mn, mx)
            out_t = (t - mn) / (mx - mn + 1e-8)  # [0,1]
            out_t = out_t * 2.0 - 1.0

        # quick sanity: check per-image variance; if extremely low, warn
        per_img_std = out_t.view(out_t.size(0), -1).std(dim=1)
        for i, s in enumerate(per_img_std.tolist()):
            if s < 1e-3:
                logging.warning("Generated image %d has very low std=%.6f -> likely degenerate.", i, s)

        # return tensor on CPU (float32) in range [-1,1]
        return out_t.to(torch.float32)
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


# ensure Generator is available
try:
    # common places: train.py, models.py, model.py
    from train import Generator
except Exception:
    try:
        from models import Generator
    except Exception:
        try:
            from model import Generator
        except Exception as e:
            raise ImportError(
                "Generator class not found. Define it in this repo or export it from train.py / models.py / model.py"
            ) from e

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
