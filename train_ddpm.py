import argparse
import math
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data_loader import get_celeba_loader
from ddpm import DDPMConfig, save_ddpm_checkpoint, load_ddpm_checkpoint, sample_ddpm
from utils import ensure_dir, get_device, seed_everything


def _denorm(imgs: torch.Tensor) -> torch.Tensor:
    return ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)


def add_ddpm_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion steps.")
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Starting beta for scheduler.")
    parser.add_argument("--beta-end", type=float, default=2e-2, help="Ending beta for scheduler.")
    parser.add_argument("--base-channels", type=int, default=64, help="Base channel width for UNet.")
    parser.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 4, 8], help="Channel multipliers.")
    parser.add_argument("--num-res-blocks", type=int, default=2, help="ResBlocks per resolution.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in UNet.")
    parser.add_argument("--no-attention", action="store_true", help="Disable attention blocks in UNet.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable).")
    parser.add_argument("--sample-interval", type=int, default=200, help="Steps between saving DDPM samples.")
    parser.add_argument("--sample-batch", type=int, default=8, help="Number of samples to generate per preview.")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume DDPM checkpoint.")
    return parser


def train_ddpm(args):
    device = get_device(args.device)
    if args.seed is not None:
        seed_everything(args.seed)

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        if not isinstance(resume_ckpt, dict) or resume_ckpt.get("model_type") != "ddpm":
            raise RuntimeError(f"Resume checkpoint {args.resume} is not a DDPM checkpoint (missing model_type='ddpm').")
        if "config" not in resume_ckpt:
            raise RuntimeError(f"Resume checkpoint {args.resume} is missing required DDPM config.")
        resume_config = DDPMConfig(**resume_ckpt["config"])

        def _assert_match(arg_val, ckpt_val, flag):
            if arg_val != ckpt_val:
                raise ValueError(f"CLI {flag}={arg_val} conflicts with checkpoint value {ckpt_val}.")

        _assert_match(args.resolution, resume_config.image_size, "--resolution")
        _assert_match(args.timesteps, resume_config.timesteps, "--timesteps")
        _assert_match(args.beta_start, resume_config.beta_start, "--beta-start")
        _assert_match(args.beta_end, resume_config.beta_end, "--beta-end")
        _assert_match(args.base_channels, resume_config.base_channels, "--base-channels")
        _assert_match(tuple(args.channel_mults), tuple(resume_config.channel_mults), "--channel-mults")
        _assert_match(args.num_res_blocks, resume_config.num_res_blocks, "--num-res-blocks")
        _assert_match(args.dropout, resume_config.dropout, "--dropout")
        _assert_match(not args.no_attention, resume_config.use_attention, "--no-attention")
        _assert_match(3, resume_config.in_channels, "--in-channels (fixed to 3)")
        config = resume_config
    else:
        config = DDPMConfig(
            in_channels=3,
            image_size=args.resolution,
            base_channels=args.base_channels,
            channel_mults=tuple(args.channel_mults),
            num_res_blocks=args.num_res_blocks,
            dropout=args.dropout,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            use_attention=not args.no_attention,
        )

    model = config.to_unet().to(device)
    scheduler = config.to_scheduler().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_epoch = 1
    global_step = 0
    if args.resume:
        ckpt = load_ddpm_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch} step {global_step}.")

    loader = get_celeba_loader(
        batch_size=args.batch_size,
        resolution=args.resolution,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
    )

    sample_dir = ensure_dir(os.path.join(args.out_dir, "samples", "ddpm"))
    ckpt_dir = ensure_dir(os.path.join(args.save_dir, "ddpm"))
    loss_history = []
    log_path = os.path.join(ensure_dir(os.path.join(args.out_dir, "logs")), "training_log.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        for i, batch in enumerate(loader):
            imgs = batch["pixel_values"] if isinstance(batch, dict) else batch
            if isinstance(imgs, list):
                imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            bsz = imgs.size(0)
            if bsz == 0:
                continue

            timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(imgs)
            noisy = scheduler.add_noise(imgs, timesteps, noise)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1

            if global_step % args.log_interval == 0:
                print(
                    f"[DDPM] Epoch {epoch} Step {global_step} "
                    f"Loss: {loss.item():.4f} | Batch: {bsz} | Device: {device}"
                )
            loss_history.append(float(loss.item()))

            if global_step % args.sample_interval == 0:
                with torch.no_grad():
                    samples, _ = sample_ddpm(
                        model,
                        scheduler,
                        num_samples=args.sample_batch,
                        shape=(3, args.resolution, args.resolution),
                        device=device,
                    )
                samples = _denorm(samples)
                grid_cols = max(1, int(math.sqrt(samples.size(0))))
                sample_path = os.path.join(sample_dir, f"ddpm_step_{global_step:06d}.png")
                save_image(samples, sample_path, nrow=grid_cols, normalize=False)
                print(f"[DDPM] Saved samples to {sample_path}")
                model.train()

            if args.steps_per_epoch and (i + 1) >= args.steps_per_epoch:
                break

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(ckpt_dir, f"ddpm_epoch_{epoch}.pt")
            save_ddpm_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step, config)
            latest_path = os.path.join(args.save_dir, "ddpm_latest.pt")
            save_ddpm_checkpoint(latest_path, model, optimizer, scheduler, epoch, global_step, config)
            print(f"[DDPM] Saved checkpoints: {ckpt_path} and {latest_path}")
            torch.save({"losses": loss_history}, log_path)

    print("[DDPM] Training complete.")


def build_parser():
    parser = argparse.ArgumentParser(description="Train a DDPM on CelebA.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", type=str, default=None, help="torch device string")
    parser.add_argument("--seed", type=int, default=None)
    add_ddpm_arguments(parser)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_ddpm(args)
