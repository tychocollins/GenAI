# app_streamlit.py

import os
<<<<<<< HEAD
import requests
import streamlit as st
import torch
from generate_faces2 import load_model_from_checkpoint, generate_from_model
from pathlib import Path

# ensure models dir exists
Path("models").mkdir(parents=True, exist_ok=True)

def download_file(url: str, dst: str):
=======
from pathlib import Path

import requests
import streamlit as st
import torch

from ddpm import load_ddpm_for_inference, sample_ddpm
from generate_faces2 import load_model_from_checkpoint, generate_from_model
from utils import get_device

# ensure models dir exists
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
NZ = 100


def download_file(url: str, dst: Path):
>>>>>>> cesar/main
    if not url:
        return False
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Failed to download checkpoint: {e}")
        return False

<<<<<<< HEAD
# Locate checkpoint: prefer local latest, else download from CHECKPOINT_URL env var
def find_or_get_checkpoint():
    # prefer explicit latest file
    latest = Path("models") / "latest_checkpoint.pt"
    if latest.exists():
        return str(latest)
    # fallback: pick newest .pt in models/
    pts = sorted(Path("models").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return str(pts[0])
    # attempt download from env var
    url = os.environ.get("CHECKPOINT_URL", "").strip()
    if url:
        dst = str(latest)
        st.info("Downloading checkpoint for first-time use (may take a while)...")
        ok = download_file(url, dst)
        return dst if ok else None
    return None

MODEL_PATH = find_or_get_checkpoint()
NZ = 100

@st.cache_resource
def load_generator():
    if not MODEL_PATH or not os.path.exists(MODEL_PATH):
        st.warning(f"Checkpoint not found in models/. Run train.py first or point MODEL_PATH to a valid checkpoint.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = load_model_from_checkpoint(MODEL_PATH, device=device, nz=NZ)
    return G

st.title("Generated Faces (DCGAN)")

G = load_generator()

cols = st.columns([1, 1, 1, 1])
with cols[0]:
    n = st.slider("Number of images", 1, 16, 8)
with cols[1]:
    seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2**31-1, value=0)
with cols[2]:
    generate_btn = st.button("Generate")
with cols[3]:
    st.write("")  # spacer

if G is None:
    st.stop()

if generate_btn:
    s = None if seed == 0 else int(seed)
    imgs = generate_from_model(G, num_images=n, nz=NZ, seed=s)
    cols = st.columns(min(n, 8))
    for i, img in enumerate(imgs):
        cols[i % len(cols)].image(img, use_column_width=True)
    st.success("Done — images are denormalized from [-1,1] to displayable RGB.")
else:
    st.info("Click Generate to produce faces. If you see colored blocks, ensure the checkpoint is a trained generator with Tanh output and that the generator architecture matches the checkpoint.")
=======

def find_or_get_checkpoint(model_type: str):
    base = MODELS_DIR
    candidates = []
    if model_type == "dcgan":
        candidates = [base / "dcgan_latest.pt", base / "latest_checkpoint.pt"]
    else:
        candidates = [base / "ddpm_latest.pt", base / model_type / f"{model_type}_latest.pt"]

    # newest .pt matching the model_type
    candidates.extend(sorted(base.rglob(f"{model_type}*.pt"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    seen = set()
    for p in candidates:
        if p and p.exists() and p not in seen:
            return str(p)
        seen.add(p)

    env_var = "CHECKPOINT_URL_DDPM" if model_type == "ddpm" else "CHECKPOINT_URL"
    url = os.environ.get(env_var, "").strip()
    if url:
        dst = base / f"{model_type}_latest.pt"
        st.info(f"Downloading {model_type.upper()} checkpoint for first-time use (may take a while)...")
        ok = download_file(url, dst)
        return str(dst) if ok else None
    return None


@st.cache_resource
def load_dcgan(selected_device: str):
    path = find_or_get_checkpoint("dcgan")
    if not path or not os.path.exists(path):
        st.warning("DCGAN checkpoint not found in models/. Run train.py --model dcgan or add a checkpoint.")
        return None
    device = torch.device(selected_device)
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        st.warning(f"Failed to read DCGAN checkpoint at {path}: {e}")
        return None

    nz = ckpt.get("nz")
    if nz is None:
        st.warning(f"DCGAN checkpoint missing latent size ('nz'); defaulting to {NZ}.")
        nz = NZ
    elif nz != NZ:
        st.info(f"Using checkpoint latent size nz={nz} (was configured default {NZ}).")

    try:
        G = load_model_from_checkpoint(path, device=device, nz=nz)
        return {"model": G, "nz": nz}
    except Exception as e:
        st.error(f"Failed to load DCGAN checkpoint: {e}")
        return None


@st.cache_resource
def load_ddpm_model(selected_device: str):
    path = find_or_get_checkpoint("ddpm")
    if not path or not os.path.exists(path):
        st.warning("DDPM checkpoint not found. Train with train.py --model ddpm or provide ddpm_latest.pt.")
        return None
    device = torch.device(selected_device)
    try:
        model, scheduler, config = load_ddpm_for_inference(path, device)
        return {"model": model, "scheduler": scheduler, "config": config, "device": device, "path": path}
    except Exception as e:
        st.warning(f"Failed to load DDPM checkpoint: {e}")
        return None


def tensor_to_images(imgs: torch.Tensor):
    imgs = ((imgs + 1.0) / 2.0).clamp(0.0, 1.0)
    imgs = (imgs * 255.0).round().to(torch.uint8)
    arr = imgs.permute(0, 2, 3, 1).cpu().numpy()
    return [arr[i] for i in range(arr.shape[0])]


st.title("Generated Faces")
device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_options.append("mps")
selected_device = st.selectbox("Device", device_options, index=0)

cols = st.columns([1, 1, 1, 1])
with cols[0]:
    model_choice = st.radio("Model", ["DCGAN", "DDPM"], horizontal=True)
with cols[1]:
    n = st.slider("Number of images", 1, 16, 8)
with cols[2]:
    seed = st.number_input("Seed (0 = random)", min_value=0, max_value=2**31 - 1, value=0)
with cols[3]:
    generate_btn = st.button("Generate")

if model_choice == "DCGAN":
    bundle = load_dcgan(selected_device)
    if bundle is None:
        st.stop()
    G = bundle["model"]
    nz = bundle["nz"]

    if generate_btn:
        s = None if seed == 0 else int(seed)
        imgs = generate_from_model(G, num_images=n, nz=nz, seed=s)
        cols = st.columns(min(n, 8))
        for i, img in enumerate(imgs):
            cols[i % len(cols)].image(img, use_column_width=True)
        st.success("DCGAN generation complete.")
    else:
        st.info("Click Generate to produce faces with DCGAN.")
else:
    bundle = load_ddpm_model(selected_device)
    if bundle is None:
        st.stop()

    model = bundle["model"]
    scheduler = bundle["scheduler"]
    config = bundle["config"]
    device = bundle["device"]

    if generate_btn:
        s = None if seed == 0 else int(seed)
        if s is not None:
            torch.manual_seed(s)
        progress = st.progress(0.0)
        status = st.empty()

        def _cb(t, total, _x):
            done = total - t
            progress.progress(min(1.0, done / total))
            status.write(f"Sampling {done}/{total} steps")

        with torch.no_grad():
            max_steps = scheduler.num_train_timesteps if device.type != "cpu" else min(250, scheduler.num_train_timesteps)
            if device.type == "cpu" and max_steps < scheduler.num_train_timesteps:
                st.info(f"CPU fast mode: sampling truncated to {max_steps} steps (of {scheduler.num_train_timesteps}).")
            samples, _ = sample_ddpm(
                model,
                scheduler,
                num_samples=n,
                shape=(3, config.image_size, config.image_size),
                device=device,
                callback=_cb,
                max_steps=max_steps,
            )

        progress.empty()
        status.empty()
        imgs = tensor_to_images(samples)
        cols = st.columns(min(n, 8))
        for i, img in enumerate(imgs):
            cols[i % len(cols)].image(img, use_column_width=True)
        st.success("DDPM sampling complete.")
    else:
        st.info("Click Generate to run DDPM sampling (iterative denoising shown below).")
>>>>>>> cesar/main
