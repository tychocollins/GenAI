import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import transforms


class DemoGenerator(nn.Module):
    """
    Minimal generator stub for demo purposes.
    Maps noise z -> image tensor in [-1, 1], shape [B, 3, 64, 64].
    """

    def __init__(self, noise_dim=128, hidden=256):
        super().__init__()
        self.noise_dim = noise_dim
        self.proj = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * 64 * 64),
            nn.Tanh(),  # keep outputs in [-1, 1]
        )

    def forward(self, z):
        return self.generate(z)

    def generate(self, z):
        out = self.proj(z)
        out = out.view(z.size(0), 3, 64, 64)
        return out


def create_demo_checkpoint(path="models/ckpt.pt", noise_dim=128):
    model = DemoGenerator(noise_dim=noise_dim)
    state = {
        "model_class": DemoGenerator,
        "model_args": {"noise_dim": noise_dim, "hidden": 256},
        "weights": model.state_dict(),
        "noise_dim": noise_dim,
    }
    torch.save(state, path)


class RealImageGenerator(nn.Module):
    """
    Uses a real image (or mean of a few) as the base output, with small noise.
    """

    def __init__(self, base_img: torch.Tensor, noise_scale: float = 0.05):
        super().__init__()
        # base_img expected shape [3, 64, 64], in [-1, 1]
        self.register_buffer("base_img", base_img)
        self.noise_scale = noise_scale

    def generate(self, z):
        # z: [B, noise_dim]; add light noise to the base image
        b = z.size(0)
        noise = torch.randn_like(self.base_img).unsqueeze(0).repeat(b, 1, 1, 1) * self.noise_scale
        return (self.base_img.unsqueeze(0).repeat(b, 1, 1, 1) + noise).clamp(-1, 1)


def _load_mean_image(real_dir: Path, take=32):
    imgs = []
    tfm = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    files = [p for p in real_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    files = files[:take]
    if not files:
        raise ValueError(f"No images found under {real_dir}")
    for p in files:
        img = Image.open(p).convert("RGB")
        imgs.append(tfm(img))
    stack = torch.stack(imgs, dim=0)
    return stack.mean(dim=0)


def create_real_checkpoint(real_dir="outputs/real_images", path="models/ckpt.pt", noise_scale=0.05):
    real_dir = Path(real_dir)
    base = _load_mean_image(real_dir)
    model = RealImageGenerator(base_img=base, noise_scale=noise_scale)
    state = {
        "model_class": RealImageGenerator,
        "model_args": {"base_img": base, "noise_scale": noise_scale},
        "weights": model.state_dict(),
        "noise_dim": 128,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"Saved real-image-based checkpoint to {path}")


if __name__ == "__main__":
    create_demo_checkpoint()
    print("Saved demo checkpoint to models/ckpt.pt")
