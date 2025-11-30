"""
Cesar Cabrera - Minimal DCGAN training script to produce a usable checkpoint.

Usage (from repo root):
python train_dcgan.py --data_dir data/celeba/celeba-dataset/img_align_celeba --epochs 1 --batch_size 64 --device cuda

Outputs:
- models/ckpt.pt with keys: model_class, model_args, weights, noise_dim
- outputs/generated/sample_*.png for quick visual checks
- outputs/logs/training_log.pt with loss history
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Generator(nn.Module):
    def __init__(self, noise_dim=128, feature_g=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, 3, 4, 2, 1, bias=False),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, z):
        return self.generate(z)

    def generate(self, z):
        # Expect z shape [B, noise_dim]
        z = z.view(z.size(0), self.noise_dim, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, feature_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def get_loader(data_dir, batch_size):
    tfm = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    ds = datasets.ImageFolder(root=data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


def save_checkpoint(gen, noise_dim, path="models/ckpt.pt"):
    state = {
        "model_class": Generator,
        "model_args": {"noise_dim": noise_dim, "feature_g": 64},
        "weights": gen.state_dict(),
        "noise_dim": noise_dim,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image root (folder with class subfolders).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--noise_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[INFO] Using device: {device}")

    loader = get_loader(args.data_dir, args.batch_size)
    gen = Generator(noise_dim=args.noise_dim).to(device)
    disc = Discriminator().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(16, args.noise_dim, device=device)
    losses = []

    out_dir = Path("outputs/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for i, (imgs, _) in enumerate(loader):
            real = imgs.to(device)
            bsz = real.size(0)
            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            # Train Discriminator
            opt_d.zero_grad()
            logits_real = disc(real)
            loss_real = criterion(logits_real, real_labels)

            noise = torch.randn(bsz, args.noise_dim, device=device)
            fake = gen(noise)
            logits_fake = disc(fake.detach())
            loss_fake = criterion(logits_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            logits_fake = disc(fake)
            loss_g = criterion(logits_fake, real_labels)
            loss_g.backward()
            opt_g.step()

            losses.append((loss_d.item(), loss_g.item()))

            if (i + 1) % 50 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Step {i+1} | D: {loss_d.item():.4f} G: {loss_g.item():.4f}")

        # Save samples each epoch
        with torch.no_grad():
            samples = gen(fixed_noise).detach().cpu()
            save_image((samples + 1) / 2, out_dir / f"sample_epoch{epoch+1}.png", nrow=4)

    # Save checkpoint and loss log
    save_checkpoint(gen, args.noise_dim, path="models/ckpt.pt")
    torch.save({"losses": losses}, log_dir / "training_log.pt")
    print("[DONE] Training complete. Checkpoint at models/ckpt.pt")


if __name__ == "__main__":
    main()
