# app_streamlit.py

import streamlit as st
import os
import torch
from generate_faces2 import load_model_from_checkpoint, generate_from_model

st.set_page_config(page_title="GAN Face Generator", layout="wide")

MODEL_PATH = os.path.join("models", "latest_checkpoint.pt")
NZ = 100

@st.cache_resource
def load_generator():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Checkpoint not found at {MODEL_PATH}. Run train.py first or point MODEL_PATH to a valid checkpoint.")
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