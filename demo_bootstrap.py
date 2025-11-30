"""
Cesar Cabrera - bootstrap demo artifacts so the dashboard runs without a checkpoint.

What it does:
- Seeds outputs/generated/ with random images (grid + individual PNGs).
- Seeds outputs/logs/training_log.pt with a dummy loss curve.
- Seeds outputs/logs/metrics.pt using synthetic data (FID + collapse).

Run: python demo_bootstrap.py
"""

from pathlib import Path

import torch
from torchvision.utils import save_image

from metrics import compute_and_store_metrics, _NoiseImages
from torch.utils.data import DataLoader


def seed_generated(out_dir: Path, num=8, seed=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    imgs = torch.randn(num, 3, 64, 64, generator=gen).clamp(-2, 2)
    imgs = torch.tanh(imgs)  # push into [-1, 1]

    grid_path = out_dir / "demo_grid.png"
    save_image((imgs + 1) / 2, grid_path, nrow=4)  # save in [0,1]

    for i in range(num):
        save_image((imgs[i] + 1) / 2, out_dir / f"demo_{i:03d}.png")

    print(f"[demo] Seeded {num} generated images in {out_dir}")


def seed_logs(log_path: Path, length=50):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    losses = [max(0.05, 3.0 * (0.98**i)) for i in range(length)]
    torch.save({"losses": losses}, log_path)
    print(f"[demo] Seeded training_log.pt at {log_path}")


def seed_metrics(save_path: Path, batch_size=16):
    real = DataLoader(_NoiseImages(count=96, seed=1), batch_size=batch_size)
    fake = DataLoader(_NoiseImages(count=96, seed=2), batch_size=batch_size)
    metrics = compute_and_store_metrics(real, fake, save_path=save_path, device="cpu", max_batches=2)
    print(f"[demo] Seeded metrics at {save_path} | FID: {metrics['fid']:.2f} collapse: {metrics['collapse']}")


def main():
    base = Path("outputs")
    generated_dir = base / "generated"
    logs_dir = base / "logs"

    seed_generated(generated_dir, num=8, seed=0)
    seed_logs(logs_dir / "training_log.pt", length=50)
    seed_metrics(logs_dir / "metrics.pt", batch_size=16)

    print("[demo] Bootstrap complete. Run: python dashboard.py")


if __name__ == "__main__":
    main()
