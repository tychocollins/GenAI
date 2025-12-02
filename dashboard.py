import streamlit as st
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The deprecated import 'from torch.serialization import add_safe_globals' 
# has been REMOVED to fix the final runtime error.

# Define the directory where your PyTorch models are stored
MODELS_DIR = "models"


# --- Helper Functions (Your Project's Logic) ---

def load_latest_generated(n=8):
    """
<<<<<<< HEAD
    Loads the file paths for the latest N generated images from the outputs folder.
    """
    try:
        gen_dir = "outputs/generated/"
        # Check if the directory exists (important for the initial deployment state)
        if not os.path.exists(gen_dir): 
            # If no folder or files, we return an empty list.
            return []
        
        # Get file list and sort by modification time to get the 'latest'
        files = sorted(
            [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.jpeg'))],
            key=os.path.getmtime,
            reverse=True
        )
        return files[:n]
    except Exception:
=======
    Loads the file paths for the latest N generated images from outputs/samples/<model>/,
    falling back to outputs/generated/ for legacy runs.
    """
    try:
        candidate_dirs = [
            "outputs/samples/dcgan/",
            "outputs/samples/ddpm/",
            "outputs/generated/",
        ]
        files = []
        for gen_dir in candidate_dirs:
            if not os.path.exists(gen_dir):
                continue
            files.extend(
                [os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
            )
        files = sorted(files, key=os.path.getmtime, reverse=True)
        if not files:
            st.warning("No generated samples found under outputs/samples/<model>/ or outputs/generated/.")
        return files[:n]
    except Exception as e:
        st.warning(f"Failed to list generated images: {e}")
>>>>>>> cesar/main
        return []

def load_training_log():
    """
    Loads the training log data (e.g., loss values) from a JSON/text file.
    """
<<<<<<< HEAD
    # Placeholder Data: Matches the graph seen in the web browser screenshot.
    return {"losses": [1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.55]}
=======
    path = "outputs/logs/training_log.pt"
    if not os.path.exists(path):
        st.warning("Training log not found (outputs/logs/training_log.pt). Run training to populate it.")
        return None
    try:
        log = torch.load(path, map_location="cpu")
        if not isinstance(log, dict) or "losses" not in log:
            st.warning("Training log is missing 'losses'; cannot render chart.")
            return None
        return log
    except Exception as e:
        st.warning(f"Failed to read training log: {e}")
        return None
>>>>>>> cesar/main

def load_metrics():
    """
    Loads evaluation metrics (e.g., FID score, mode collapse status).
    """
<<<<<<< HEAD
    # Placeholder Data: Matches the metrics seen in the web browser screenshot.
    return {"fid": 12.45, "collapse": None}
=======
    path = "outputs/logs/metrics.pt"
    if not os.path.exists(path):
        st.warning("Metrics file not found (outputs/logs/metrics.pt). Run metrics_runner.py to generate it.")
        return None
    try:
        metrics = torch.load(path, map_location="cpu")
        if not isinstance(metrics, dict) or "fid" not in metrics:
            st.warning("Metrics file missing 'fid'; cannot render metrics.")
            return None
        return metrics
    except Exception as e:
        st.warning(f"Failed to read metrics: {e}")
        return None
>>>>>>> cesar/main

def load_model_state(model_path):
    """
    Loads a PyTorch model checkpoint from the given path.
    """
    try:
        if not os.path.exists(model_path):
            return None
<<<<<<< HEAD
        # Your actual model loading code goes here.
        return True # Placeholder to unblock the dashboard.
    except Exception:
=======
        ckpt = torch.load(model_path, map_location="cpu")
        if not isinstance(ckpt, dict) or "model_type" not in ckpt:
            st.warning(f"Checkpoint at {model_path} missing 'model_type'; may be incompatible.")
        return ckpt
    except Exception as e:
        st.warning(f"Failed to load checkpoint {model_path}: {e}")
>>>>>>> cesar/main
        return None

def generate_sample(model):
    """
    Generates a sample image using the loaded model.
    """
    # Placeholder: Creates a simple placeholder image for visual confirmation
    img = Image.new('RGB', (256, 256), color = 'lightblue')
<<<<<<< HEAD
    return img
=======
    return img

# --- Minimal UI wiring ---
st.title("GAN/DDPM Dashboard")

st.header("Latest Generated Samples")
latest = load_latest_generated(n=8)
if latest:
    st.image(latest, caption=[os.path.basename(p) for p in latest], width=256)

st.header("Training Loss")
log = load_training_log()
if log and "losses" in log:
    st.line_chart(log["losses"])

st.header("Metrics")
metrics = load_metrics()
if metrics:
    st.json(metrics)
>>>>>>> cesar/main
