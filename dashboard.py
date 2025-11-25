# Da Marc Henry
# Dashboard for AI Facial Generation Project
# Display generated samples, training curves
# Displayevaluation metrics (FID / collapse), basic model comparison visualizations
# Dependencies: matplotlib, torch, pillow, os
# Run with: python dashboard.py in cmd.exe

import os 
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Config - Folder Structure
GENERATED_DIR = "outputs/generated/"
LOG_FILE = "outputs/logs/training_log.pt"
METRICS_FILE = "outputs/logs/metrics.pt"
MODELS_DIR = "models/"

# Helper Functions
def load_latest_generated(n=8):
    # Load the latest N generated images from John's 
    if not os.path.exists(GENERATED_DIR):
        return[]
    
    files = sorted(os.listdir(GENERATED_DIR))[-n:]
    return [os.path.join (GENERATED_DIR, f) for f in files]


def load_training_log():
    # Load Erick's training loss curves
    if os.path.exists(LOG_FILE):
        return torch.load(LOG_FILE)
    return None


def load_metrics():
    # Load Cesar's FID and collapse metrics
    if os.path.exists(METRICS_FILE):
        return torch.load(METRICS_FILE)
    return None


def load_model_state(path):
    # Load any trained model for comparison
    try: 
        state = torch.load(path, map_location = "cpu")
        model_class = state["model_class"]
        model_args = state["model_args"]
        model = model_class(**model_args)
        model.load_state_dict(state["weights"])
        model.eval()
        return model
    except:
        return None
    
    
def generate_sample(model, noise_dim = 512):
    # Generate one sample image for model comparsion
    noise = torch.randn(1, noise_dim)
    with torch.no_grad():
        out = model.generate(noise)
        
    img = out.squeeze().permute(1, 2 ,0).numpy()
    img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)



# Main Dashboard GUI
def run_dashboard():
    print("[INFO] Launching Dashboard...")
    
    fig = plt.figure(figsize = (14, 10))
    fig.suptitle("AI Facial Generation Project - Dashboard", fontsize = 20)
    
    grid = plt.GridSpec(3, 3, wspace = 0.4, hspace = 0.5)
    
    # 1. Generated Samples
    ax1 = fig.add_subplot(grid[0, :])
    ax1.set_title("Latest Generated Samples")
    
    gen_images = load_latest_generated()
    
    if not gen_images:
        ax1.text(0.5, 0.5, "No generated images found.", ha="center")
    else:
        imgs = [np.array(Image.open(p).resize((128, 128))) for p in gen_images]
        montage = np.hstack(imgs)
        ax1.imshow(montage)
        ax1.axis("off")
        
    # Training loss curvve
    ax2 = fig.add_subplot(grid[1, 0])
    ax2.set_title("Training Loss Curve")
    
    logs = load_training_log()
    if logs and "losses" in logs:
        ax2.plot(logs["losses"], color = "blue")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Loss")
    else:
        ax2.text(0.5, 0.5, "No training logs found.", ha = "center")
    
    
    # Metrics Display (FID + Collapse)    
    ax3 = fig.add_subplot(grid[1, 1])
    ax3.set_title("Evaluation Metrics")
    ax3.axis("off")
    metrics = load_metrics()
    if metrics:
        fid = metrics.get("fid", "N/A")
        collapse = metrics.get("collapse", "N/A")
        ax3.text(0.1, 0.8, f"FID Score: {fid}", fontsize = 12)
        ax3.text(0.1, 0.6, f"Mode Collapse: {collapse}", fontsize = 12)
    else:
        ax3.text(0.5, 0.5, "No metrics found.", ha = "center")
        
    
    # Model Comparison (VAE vs DDPM vs Flow)
    ax4 = fig.add_subplot(grid[1, 2])
    ax4.set_title("Model Comparison")
    ax4.axis("off")
        
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    
    if model_files:
        # Load the first model as example (or loop if needed)    
        model_path = os.path.join(MODELS_DIR, model_files[0])
        model = load_model_state(model_path)
        
        if model:
            sample = generate_sample(model)
            ax4.imshow(sample)
            ax4.axis("off")
        else:
            ax4.text(0.5, 0.5, "Model failed to load.", ha = "center")
    else:
        ax4.text(0.5, 0.5, "No models available.", ha = "center")
        
    
    # System Notes
    ax5 = fig.add_subplot(grid[2, :])
    ax5.set_title("Team Integration Notes")
    ax5.axis("off")
    
    notes = (
        "• Tycho: Preprocessing pipeline\n"
        "• Erick: Training progress pulled automatically from logs\n"
        "• John: Generated faces displayed from /outputs/generated/\n"
        "• Cesar: FID + Collapse metrics displayed dynamically\n"
        "• Da Marc: Dashboard integrates all components visually")
        
    ax5.text(0.02, 0.8, notes, fontsize = 12, va = "top")
    
    
    # Display the dashboard window
    plt.show()
    print("[SUCCESS!!!] Dashboard displayed successfully")
    

# Run Dashboard
if __name__ == "__main__":
    run_dashboard()