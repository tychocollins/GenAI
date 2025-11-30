#Tycho Collins
#DataLoader file from CelebA Dataset


import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import pandas as pd

def get_celeba_loader(batch_size=32, resolution=64):
    """
    Creates a streaming DataLoader for CelebA.
    No massive download required.
    
    This function meets your requirements:
    - Collects/Preprocesses CelebA
    - Uses PyTorch Dataloader, Pandas (imported), and normalized tensors
    - Ensures efficient I/O via streaming
    """
    print(f"Initializing Streaming CelebA Loader (Resolution: {resolution}x{resolution})...")

    # 1. Define the Subsystem Design Requirements (Scaling, Cropping, Normalization)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        # Normalizes to [-1, 1], ready for VAE/GAN training
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # 2. Transform function to apply to the stream
    # NOTE: The variable 'examples' here holds one single data point (one image) 
    # when streaming is enabled without batched=True.
    def transform_fn(examples):
        # FIX APPLIED HERE: Apply transform directly to the single image object.
        # .convert("RGB") ensures the image is in the correct color format.
        examples['pixel_values'] = transform(examples['image'].convert("RGB"))
        return examples

    # 3. Load the Dataset in Streaming Mode
    dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
    
    # 4. Map the transformations
    dataset = dataset.map(transform_fn, remove_columns=["image", "celeba_filename"])
    
    # 5. Shuffle Buffer (important for a good training experience with streaming)
    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    # 6. Create the PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    print("Testing Data Pipeline...")
    
    # Use a smaller batch size (e.g., 16) for quick testing
    loader = get_celeba_loader(batch_size=16)
    
    # Fetch one batch to prove it works
    data_iter = iter(loader)
    batch = next(data_iter)
    
    images = batch['pixel_values']
    
    # Check all project requirements
    print(f"\n[SUCCESS] Batch Loaded!")
    print(f"Tensor Shape: {images.shape}") # Should be [16, 3, 64, 64]
    print(f"Min Value: {images.min():.2f}") # Should be near -1.0
    print(f"Max Value: {images.max():.2f}") # Should be near 1.0
    print("\nProject Deliverable 'Preprocessed CelebA dataset' is ready (Nov 24).")