# app_streamlit.py

import streamlit as st
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import core dashboard functions from your existing file
from dashboard import load_latest_generated, load_training_log, load_metrics, load_model_state, generate_sample, MODELS_DIR


# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI Face Generation Dashboard")
GENERATED_DIR = "outputs/generated/"


# --- Dashboard Component Functions ---

def display_generated_samples():
    """Displays the latest generated faces."""
    st.header("🖼️ Latest Generated Samples")
    
    # Use the existing function to get file paths
    gen_images = load_latest_generated(n=8)

    if not gen_images:
        st.warning("No generated images found. Run `demoboostrap.py` first!")
        return

    # Display images in a simple horizontal row using Streamlit columns
    cols = st.columns(len(gen_images))
    for col, img_path in zip(cols, gen_images):
        try:
            image = Image.open(img_path)
            col.image(image, use_column_width=True)
        except Exception as e:
            col.error(f"Error loading image: {e}")


def display_training_metrics():
    """Displays loss curve and FID/Collapse metrics side-by-side."""
    col1, col2 = st.columns([2, 1])

    # Plot 1: Training Loss Curve
    with col1:
        st.subheader("📉 Training Loss Curve")
        logs = load_training_log()
        if logs and "losses" in logs:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(logs["losses"], color="blue")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            st.pyplot(fig) # Use st.pyplot to display the Matplotlib figure
        else:
            st.info("No training logs found.")
    
    # Plot 2: Evaluation Metrics
    with col2:
        st.subheader("📊 Evaluation Metrics")
        metrics = load_metrics()
        if metrics:
            fid = metrics.get("fid", "N/A")
            collapse = metrics.get("collapse", "N/A")
            
            st.markdown(f"**FID Score:** `{fid:.2f}`")
            if collapse:
                 st.error(f"**Mode Collapse:** ⚠️ **{collapse}**")
            else:
                 st.success(f"**Mode Collapse:** ✅ **{collapse}**")
        else:
            st.info("No metrics found.")


def display_model_comparison():
    """Displays a sample generated image from the latest model checkpoint."""
    st.header("🤖 Model Sample Test")
    
    # Find the latest model file (or simply look for ckpt.pt)
    if os.path.exists(MODELS_DIR):
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    else:
        model_files = []

    if model_files:
        model_path = os.path.join(MODELS_DIR, model_files[0])
        model = load_model_state(model_path)

        if model:
            st.caption(f"Loaded Checkpoint: {model_files[0]}")
            sample_image = generate_sample(model)
            st.image(sample_image, caption="Sample Generation (Noise: 512)", width=256)
        else:
            st.error("Model failed to load. Check `demogenerator.py` imports.")
    else:
        st.info("No models available in the `/models` directory.")


# --- Main Run Logic ---
def main():
    st.title("👨‍🔬 GenAI Project Dashboard")
    st.markdown("---")
    
    # Run the display components
    display_generated_samples()
    st.markdown("---")
    display_training_metrics()
    st.markdown("---")
    display_model_comparison()
    
    st.sidebar.title("Team Status")
    st.sidebar.write("Project components integrated successfully.")
    st.sidebar.markdown(
        "* **Tycho:** Data Pipeline\n"
        "* **Erick:** Training Logs/Checkpoints\n"
        "* **John:** Generated Output\n"
        "* **Cesar:** Metric Calculation\n"
        "* **Da Marc:** Streamlit Interface"
    )

if __name__ == "__main__":
    main()