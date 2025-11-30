import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from data_loader import get_celeba_loader

# Simple DCGAN-like Generator
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

# Simple DCGAN-like Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

def denorm(img):
    return (img + 1.0) / 2.0

def save_sample_images(G, nz, device, out_dir, step, n_samples=64):
    z = torch.randn(n_samples, nz, 1, 1, device=device)
    with torch.no_grad():
        imgs = G(z).cpu()
    imgs = denorm(imgs)
    os.makedirs(out_dir, exist_ok=True)
    save_image(imgs, os.path.join(out_dir, f"sample_step_{step:06d}.png"), nrow=8)

def save_checkpoint(G, D, optG, optD, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "optimizer_G_state_dict": optG.state_dict(),
        "optimizer_D_state_dict": optD.state_dict(),
    }, path)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_celeba_loader(batch_size=args.batch_size, resolution=args.resolution)

    nz = args.nz
    G = Generator(nz=nz).to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    real_label = 1.0
    fake_label = 0.0

    step = 0
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(loader):
            # handle different batch formats
            if isinstance(batch, dict) and "pixel_values" in batch:
                imgs = batch["pixel_values"]
            else:
                # assume loader yields raw tensors or lists
                imgs = batch
            if isinstance(imgs, list):
                imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            b_size = imgs.size(0)
            if b_size == 0:
                continue

            # Train Discriminator
            D.zero_grad()
            labels = torch.full((b_size,), real_label, device=device)
            output = D(imgs)
            errD_real = criterion(output, labels)
            errD_real.backward()

            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = G(noise)
            labels.fill_(fake_label)
            output = D(fake.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            optD.step()

            # Train Generator
            G.zero_grad()
            labels.fill_(real_label)
            output = D(fake)
            errG = criterion(output, labels)
            errG.backward()
            optG.step()

            step += 1

            if step % args.log_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{step}] Loss_D: {(errD_real+errD_fake).item():.4f} Loss_G: {errG.item():.4f}")

            if step % args.sample_interval == 0:
                save_sample_images(G, nz, device, os.path.join(args.out_dir, "samples"), step, n_samples=min(64, args.batch_size*4))

            if step >= args.steps_per_epoch:
                break

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(G, D, optG, optD, epoch, ckpt_path)
            save_checkpoint(G, D, optG, optD, epoch, os.path.join(args.save_dir, "latest_checkpoint.pt"))
            print(f"Saved checkpoint: {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000,
                        help="max steps per epoch (useful for streaming loaders)")
    parser.add_argument("--nz", type=int, default=100, help="latent dim")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--sample-interval", type=int, default=200,
                        help="save sample images every N steps")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    train(args)