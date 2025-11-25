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
from pathlib import Path

import torch
from torchvision.utils import save_image


# ---- Helpers ---------------------------------------------------------------

def _pick_device(force_cpu: bool = False) -> torch.device:
    """Selects CUDA if available, otherwise CPU, unless CPU is forced."""
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """
    Load a trained model from a checkpoint.

    Expected checkpoint format (set by Erick's training code):
        {
            "model_class": <class reference>,
            "model_args":  { ... kwargs ... },
            "weights":     <state_dict>,
            # optional: "noise_dim": int, "epoch": int, ...
        }

    The model_class must implement:
        model = model_class(**model_args)
        out = model.generate(noise)  # returns images in [-1, 1]
    """
    state = torch.load(checkpoint_path, map_location=device)

    if not all(k in state for k in ["model_class", "model_args", "weights"]):
        raise KeyError(
            "Checkpoint is missing one of the required keys: "
            "'model_class', 'model_args', 'weights'."
        )

    model_class = state["model_class"]
    model_args = state["model_args"]
    weights = state["weights"]

    model = model_class(**model_args)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # Optional: if Erick stores noise_dim in the checkpoint,
    # we can pull it out here; but we also accept a CLI override.
    noise_dim = state.get("noise_dim", None)

    return model, noise_dim


def sample_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    """Sample latent noise z ~ N(0, I). Shape: [B, noise_dim]."""
    return torch.randn(batch_size, noise_dim, device=device)


def generate_from_model(model, num_samples: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    """
    Generate a batch of images using the model's .generate() method.

    The model is expected to output images in [-1, 1] with shape [B, 3, H, W].
    """
    noise = sample_noise(num_samples, noise_dim, device=device)
    with torch.no_grad():
        out = model.generate(noise)

    if out.ndim != 4 or out.shape[1] != 3:
        raise ValueError(
            f"Expected output shape [B, 3, H, W], got {tuple(out.shape)} instead."
        )

    return out


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


if __name__ == "__main__":
    main()
