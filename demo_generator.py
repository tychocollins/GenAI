import torch
import torch.nn as nn
import os
import random
from PIL import Image
import numpy as np

# --- Configuration (ADJUST THESE TO MATCH YOUR PROJECT) ---
<<<<<<< HEAD
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs/generated"
# Use the name of your actual model file here
CHECKPOINT_FILENAME = "your_actual_checkpoint.pt" 
=======
MODELS_DIR = "models/demo"
OUTPUTS_DIR = "outputs/generated"
# Use the name of your actual model file here
CHECKPOINT_FILENAME = "demo_checkpoint.pt" 
>>>>>>> cesar/main
CHECKPOINT_PATH = os.path.join(MODELS_DIR, CHECKPOINT_FILENAME)
NUM_SAMPLES = 8
Z_DIM = 512 # Latent space dimension 
IMAGE_SIZE = 64 # Output image size 

# --- The Generator Model (***REPLACE THIS WITH YOUR REAL MODEL CLASS***) ---
class MyGeneratorModel(nn.Module):
    """
    ***PLACEHOLDER: REPLACE THIS ENTIRE CLASS DEFINITION*** ***WITH YOUR ACTUAL PyTorch Generator Model Definition (e.g., GAN or VAE Gen).***
    """
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        # This placeholder output layer just generates random noise
        self.output_layer = nn.Sequential(
            nn.Conv2d(z_dim, 3, 1),
            nn.Tanh() 
        )

    def forward(self, z):
        # Reshape noise tensor for the placeholder model
        z = z.view(z.size(0), z.size(1), 1, 1).expand(-1, -1, self.img_size, self.img_size)
        return self.output_layer(z)


# --- Core Functions ---

def ensure_directories_exist():
    """Checks for and creates necessary directories (THE FIX for RuntimeError)."""
    # Create models directory
    if not os.path.exists(MODELS_DIR):
        print(f"Creating missing directory: {MODELS_DIR}")
        os.makedirs(MODELS_DIR)
    
    # Create outputs directory
    if not os.path.exists(OUTPUTS_DIR):
        print(f"Creating missing directory: {OUTPUTS_DIR}")
        os.makedirs(OUTPUTS_DIR)

def create_demo_checkpoint():
    """
    ***ADJUST THIS FUNCTION TO LOAD YOUR REAL, TRAINED MODEL***
    This function currently creates a DUMMY model and saves a placeholder checkpoint.
    """
    print("Preparing model checkpoint...")
    
    # 1. Create the necessary directories (The FIX)
    ensure_directories_exist() 
    
    # 2. Instantiate the model
    # ***Make sure the Z_DIM and IMAGE_SIZE match your real model's requirements***
    model = MyGeneratorModel(z_dim=Z_DIM, img_size=IMAGE_SIZE)
    
    # 3. Save the placeholder checkpoint
    state = {
        'model_state_dict': model.state_dict(),
        'iteration': 1000,
<<<<<<< HEAD
        'loss': 0.5
=======
        'loss': 0.5,
        'model_type': 'demo',
        'nz': Z_DIM,
>>>>>>> cesar/main
    }
    
    # Only save the checkpoint if a real, trained one is not already present
    if not os.path.exists(CHECKPOINT_PATH):
        torch.save(state, CHECKPOINT_PATH)
        print(f"Placeholder checkpoint saved to: {CHECKPOINT_PATH}")
    else:
        print(f"Checkpoint already exists at {CHECKPOINT_PATH}. Loading existing model.")
        
    # Load the state dictionary back into the model for use
    # ***IMPORTANT: You should use your actual model loading/unwrapping logic here***
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
<<<<<<< HEAD
=======
    if "model_type" not in checkpoint:
        checkpoint["model_type"] = "dcgan"
        torch.save(checkpoint, CHECKPOINT_PATH)
>>>>>>> cesar/main
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def generate_and_save_samples(model):
    """Generates sample images from the model and saves them."""
    print(f"Generating {NUM_SAMPLES} samples...")
    
    # 1. Ensure output directory exists
    ensure_directories_exist()

    # 2. Generate latent vectors (noise)
    latent_vectors = torch.randn(NUM_SAMPLES, Z_DIM)

    # 3. Generate images (turn off gradient calculation)
    with torch.no_grad():
        model.eval()
        generated_images = model(latent_vectors)

    # 4. Process and save images
    for i in range(NUM_SAMPLES):
        # Convert tensor (C, H, W) to NumPy array (H, W, C)
        img_tensor = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize (e.g., from [-1, 1] to [0, 255])
        img_array = (img_tensor * 127.5 + 127.5).astype(np.uint8)
        
        # Save as PIL Image
        img = Image.fromarray(img_array)
        output_path = os.path.join(OUTPUTS_DIR, f"face_{i+1}_{random.randint(1000, 9999)}.png")
        img.save(output_path)
        print(f"Sample saved to: {output_path}")

    print("Sample generation complete.")


# --- Main Execution ---
if __name__ == "__main__":
    # This will now create the 'models' folder if it is missing.
    model = create_demo_checkpoint()
    
    # This will now create the 'outputs/generated' folder if it is missing.
<<<<<<< HEAD
    generate_and_save_samples(model)
=======
    generate_and_save_samples(model)
>>>>>>> cesar/main
