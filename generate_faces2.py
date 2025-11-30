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


# Simple DCGAN-like Generator (must match train.py)
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


def _pick_device(force_cpu: bool = False) -> torch.device:
    """Selects CUDA if available, otherwise CPU, unless CPU is forced."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _denorm_tensor(img_tensor):
    # img_tensor: CxHxW or BxCxHxW in [-1,1] -> uint8 HxWxC
    img = (img_tensor + 1.0) / 2.0  # [0,1]
    img = img.clamp(0, 1)
    img = (img * 255).to(torch.uint8).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # C,H,W -> H,W,C
    else:
        img = np.transpose(img, (0, 2, 3, 1))  # B,C,H,W -> B,H,W,C
    return img


def load_model_from_checkpoint(checkpoint_path, device=None, nz=100):
    """
    Loads generator from checkpoint saved by train.py (keys: generator_state_dict or whole state_dict).
    Returns generator (in eval mode) on requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(nz=nz).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    # support both formats: {generator_state_dict: ...} or direct state_dict
    if isinstance(ckpt, dict) and "generator_state_dict" in ckpt:
        state = ckpt["generator_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    try:
        G.load_state_dict(state)
    except RuntimeError:
        # attempt to filter missing prefixes (e.g., "module.")
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        G.load_state_dict(new_state)

    G.eval()
    return G


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
