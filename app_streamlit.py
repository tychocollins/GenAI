# app_streamlit.py

import os
import requests
import streamlit as st
import torch
from generate_faces2 import load_model_from_checkpoint, generate_from_model
from pathlib import Path

# ensure models dir exists
Path("models").mkdir(parents=True, exist_ok=True)

def download_file(url: str, dst: str):
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