import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    # choose the largest divisor up to 32 to avoid GroupNorm crashes
    while groups > 1 and channels % groups != 0:
        groups -= 1
    if channels % groups != 0:
        raise ValueError(
            f"Channel count {channels} not divisible by any group <=32. Adjust base_channels or channel_multipliers."
        )
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        qkv = self.qkv(h_)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)

        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        time_dim: Optional[int] = None,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        time_dim = time_dim or base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
            layer_dict = {"blocks": blocks}
            if use_attention:
                layer_dict["attn"] = AttentionBlock(ch)
            if mult != channel_mults[-1]:
                layer_dict["down"] = Downsample(ch)
            self.downs.append(nn.ModuleDict(layer_dict))

        self.mid_block1 = ResidualBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch) if use_attention else None
        self.mid_block2 = ResidualBlock(ch, ch, time_dim, dropout)

        self.ups = nn.ModuleList()
        for idx, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch + out_ch, out_ch, time_dim, dropout))
                ch = out_ch
            layer_dict = {"blocks": blocks}
            if use_attention:
                layer_dict["attn"] = AttentionBlock(ch)
            if idx != len(channel_mults) - 1:
                layer_dict["up"] = Upsample(ch)
            self.ups.append(nn.ModuleDict(layer_dict))

        self.final_norm = _group_norm(ch)
        self.final_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_embedding(timesteps, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        h = self.init_conv(x)
        residuals: List[torch.Tensor] = []

        for layer in self.downs:
            for block in layer["blocks"]:
                h = block(h, t_emb)
                if "attn" in layer:
                    attn = layer["attn"]
                    h = h + attn(h)
                residuals.append(h)
            if "down" in layer:
                h = layer["down"](h)

        h = self.mid_block1(h, t_emb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.ups:
            for block in layer["blocks"]:
                res = residuals.pop()
                h = torch.cat([h, res], dim=1)
                h = block(h, t_emb)
            if "attn" in layer:
                attn = layer["attn"]
                h = h + attn(h)
            if "up" in layer:
                h = layer["up"](h)

        h = F.silu(self.final_norm(h))
        return self.final_conv(h)


def _extract(coeff: torch.Tensor, timesteps: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    b = timesteps.shape[0]
    idx = timesteps.to(coeff.device)
    out = coeff.gather(-1, idx)
    return out.view(b, *([1] * (len(x_shape) - 1)))


class LinearNoiseScheduler(nn.Module):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
        )

        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

    def add_noise(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = noise if noise is not None else torch.randn_like(x0)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x0.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0.shape)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def step(self, noise_pred: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        beta = _extract(self.betas, timesteps, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape)
        sqrt_recip = _extract(self.sqrt_recip_alphas, timesteps, x.shape)

        model_mean = sqrt_recip * (x - beta / (sqrt_one_minus + 1e-8) * noise_pred)
        posterior_var = _extract(self.posterior_variance, timesteps, x.shape)

        noise = torch.randn_like(x)
        nonzero_mask = (timesteps > 0).float().view(x.shape[0], *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_var + 1e-8) * noise

    def to_config(self) -> Dict:
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
        }


@dataclass
class DDPMConfig:
    in_channels: int = 3
    image_size: int = 64
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    dropout: float = 0.0
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    use_attention: bool = True

    def to_unet(self) -> UNetModel:
        return UNetModel(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            channel_mults=self.channel_mults,
            num_res_blocks=self.num_res_blocks,
            dropout=self.dropout,
            time_dim=self.base_channels * 4,
            use_attention=self.use_attention,
        )

    def to_scheduler(self) -> LinearNoiseScheduler:
        return LinearNoiseScheduler(
            num_train_timesteps=self.timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )


def save_ddpm_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LinearNoiseScheduler,
    epoch: int,
    step: int,
    config: DDPMConfig,
) -> None:
    torch.save(
        {
            "model_type": "ddpm",
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler.to_config(),
            "config": asdict(config),
        },
        path,
    )


def load_ddpm_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LinearNoiseScheduler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict:
    device = torch.device(map_location)
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or ckpt.get("model_type") != "ddpm":
        raise RuntimeError(f"Checkpoint {path} is not a DDPM checkpoint (missing model_type='ddpm').")
    if "config" not in ckpt:
        raise RuntimeError(f"Checkpoint {path} is missing required DDPM config.")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler" in ckpt:
        sched_cfg = ckpt["scheduler"]
        scheduler.__init__(
            num_train_timesteps=sched_cfg.get("num_train_timesteps", scheduler.num_train_timesteps),
            beta_start=sched_cfg.get("beta_start", scheduler.beta_start),
            beta_end=sched_cfg.get("beta_end", scheduler.beta_end),
        )
        scheduler.to(device)
    return ckpt


def load_ddpm_for_inference(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or ckpt.get("model_type") != "ddpm":
        raise RuntimeError(f"Checkpoint {path} is not a DDPM checkpoint (missing model_type='ddpm').")
    if "config" not in ckpt:
        raise RuntimeError(f"Checkpoint {path} is missing required DDPM config.")

    config = DDPMConfig(**ckpt["config"])
    model = config.to_unet().to(device)
    scheduler = config.to_scheduler().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if "scheduler" in ckpt:
        scheduler.__init__(
            num_train_timesteps=ckpt["scheduler"].get("num_train_timesteps", scheduler.num_train_timesteps),
            beta_start=ckpt["scheduler"].get("beta_start", scheduler.beta_start),
            beta_end=ckpt["scheduler"].get("beta_end", scheduler.beta_end),
        )
        scheduler.to(device)
    return model.eval(), scheduler, config


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    scheduler: LinearNoiseScheduler,
    num_samples: int,
    shape: Tuple[int, int, int],
    device: torch.device,
    progress: bool = False,
    callback: Optional[callable] = None,
    store_intermediate: bool = False,
    max_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    model.eval()
    x = torch.randn(num_samples, *shape, device=device)
    intermediates: List[torch.Tensor] = []

    total = scheduler.num_train_timesteps
    steps = total if max_steps is None else min(max_steps, total)
    for idx, t in enumerate(reversed(range(total))):
        if idx >= steps:
            break
        timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, timesteps)
        x = scheduler.step(noise_pred, timesteps, x)

        if store_intermediate and (t % max(total // 10, 1) == 0 or t == 0):
            intermediates.append(x.detach().cpu())
        if callback:
            callback(t, total, x)
        if progress and (idx % 50 == 0 or t == 0):
            print(f"Sampling step {t}/{total}")

    x = x.clamp(-1.0, 1.0)
    return x, intermediates


def save_samples(imgs: torch.Tensor, out_dir: str, prefix: str = "ddpm") -> str:
    out_dir = out_dir.rstrip("/\\")
    save_path = f"{out_dir}/{prefix}_grid.png"
    save_image((imgs + 1.0) * 0.5, save_path, normalize=False, nrow=int(math.sqrt(imgs.size(0))) or 1)
    return save_path
