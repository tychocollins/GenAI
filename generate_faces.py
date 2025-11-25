"""
generate_faces.py
Image Generation Module - John Cosby

This script:
- loads a trained generative model checkpoint (e.g., VAE),
- samples latent noise,
- generates face images in [-1, 1] tensor format,
- converts them to PNG files in outputs/generated/.

It is designed to be compatible with:
- the [-1, 1] normalization used by the dataset loader,
- Cesar's metrics.py (FID and mode-collapse),
- Da Marc's dashboard (reads images from outputs/generated/).
"""

import os
import argparse
from datetime import datetime

import torch
from torch import nn
from torchvision.utils import save_image


# ====== MODEL IMPORT ======
# You may need to change this to match Erick's model file / class name.
# Example assumes a convolutional VAE defined as VAE in models_vae.py
try:
    from models_vae import VAE  # TODO: confirm with Erick
except ImportError:
    VAE = None
    print("Warning: Could not import VAE from models_vae. "
          "Please update the import to match the training code.")


def build_model(model_type: str, latent_dim: int, img_channels: int = 3) -> nn.Module:
    """
    Factory function to build the generative model.
    For now, assume VAE is the default model type.
    You can extend this to support 'diffusion', 'flow', etc.
    """
    if model_type.lower() == "vae":
        if VAE is None:
            raise ImportError(
                "VAE model class not found. "
                "Update build_model() to use the correct model from Erick's code."
            )
        model = VAE(latent_dim=latent_dim, img_channels=img_channels)
        return model
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         "Supported: 'vae' (extend as needed).")


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load model weights from a checkpoint file.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Depending on Erick's training script, this might be:
    #   checkpoint["state_dict"] or checkpoint["model_state_dict"] or the raw state dict.
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def rescale_from_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from [-1, 1] range to [0, 1] range for saving as images.
    """
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    return x


def generate_samples(model: nn.Module,
                     num_samples: int,
                     latent_dim: int,
                     device: torch.device) -> torch.Tensor:
    """
    Sample latent vectors and generate images using the model.

    Returns:
        images: Tensor of shape [num_samples, C, H, W] in [-1, 1].
    """
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        # For a VAE, assume model has a 'decode' method.
        if hasattr(model, "decode"):
            images = model.decode(z)
        else:
            # If Erick's model uses a different interface, update this.
            images = model(z)

    return images


def main():
    parser = argparse.ArgumentParser(description="Generate faces from a trained model checkpoint.")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint (.pt or .pth).")
    parser.add_argument("--out_dir", type=str, default="outputs/generated",
                        help="Directory where generated images will be saved.")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of images to generate.")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="Dimension of the latent vector. Must match training.")
    parser.add_argument("--model_type", type=str, default="vae",
                        help="Type of generative model: e.g. 'vae', 'diffusion' (extend as needed).")
    parser.add_argument("--grid", action="store_true",
                        help="If set, also save a grid image of all samples.")

    args = parser.parse_args()

    # ====== Setup ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Timestamped subdirectory for easier tracking by metrics/UI
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Saving generated images to: {run_dir}")

    # ====== Build and load model ======
    model = build_model(args.model_type, latent_dim=args.latent_dim)
    model = load_checkpoint(model, args.checkpoint, device)

    # ====== Generate images ======
    images_minus1_1 = generate_samples(
        model=model,
        num_samples=args.num_samples,
        latent_dim=args.latent_dim,
        device=device
    )

    # Save tensors in [-1, 1] for metrics if needed
    # (Cesar can load them directly from files or from the saved PNGs)
    images_0_1 = rescale_from_minus1_1(images_minus1_1)

    # Save each image individually
    for idx in range(args.num_samples):
        img_path = os.path.join(run_dir, f"sample_{idx:03d}.png")
        save_image(images_0_1[idx], img_path)
        # Images are saved in [0, 1] range; metrics and UI can read from this folder.
        print(f"Saved {img_path}")

    # Optionally, also save a grid for quick visual inspection
    if args.grid:
        grid_path = os.path.join(run_dir, "grid.png")
        save_image(images_0_1, grid_path, nrow=int(args.num_samples ** 0.5))
        print(f"Saved image grid to {grid_path}")

    print("Generation complete.")


if __name__ == "__main__":
    main()
