#Tycho Collins
#DataLoader file from CelebA Dataset


import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from torchvision import transforms
import pandas as pd

class HFDatasetIterable(IterableDataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.hf_dataset = hf_dataset

    def __iter__(self):
        # HF streaming dataset is already an iterator
        for example in self.hf_dataset:
            yield example

def get_celeba_loader(batch_size=32, resolution=64):
    """
    Creates a streaming DataLoader for CelebA.
    """
    print(f"Initializing Streaming CelebA Loader (Resolution: {resolution}x{resolution})...")

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def transform_fn(examples):
        examples['pixel_values'] = transform(examples['image'].convert("RGB"))
        return examples

    try:
        dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
    except Exception:
        # fallback to non-streaming (downloads dataset locally; slower but stable)
        print("Streaming failed — falling back to non-streaming download (this may take time).")
        dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=False)
    dataset = dataset.map(transform_fn, remove_columns=["image", "celeba_filename"])
    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    iterable = HFDatasetIterable(dataset)
    dataloader = DataLoader(iterable, batch_size=batch_size)
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
    print(f"Max Value: {images.max():.2f}")

    # Demo generation (only run when file executed directly)
    try:
        from generate_faces2 import load_model_from_checkpoint, generate_from_model
        import os, torch, numpy as np
        ckpt = "models/latest_checkpoint.pt"
        if not os.path.exists(ckpt):
            print("Checkpoint missing:", ckpt)
        else:
            G = load_model_from_checkpoint(ckpt, device=torch.device("cpu"), nz=100)
            imgs = generate_from_model(G, num_images=8, nz=100, device=torch.device("cpu"), seed=42)

            # compute pixel stats and save debug images
            arrs = [np.array(im).astype(np.float32)/255.0 for im in imgs]
            stack = np.stack(arrs)
            print("shape:", stack.shape)
            print("mean:", stack.mean(), "std:", stack.std(), "min:", stack.min(), "max:", stack.max())

            for i, im in enumerate(imgs):
                im.save(f"debug_gen_{i}.png")
                print("Saved debug_gen_%d.png" % i)
    except Exception as e:
        print("Demo generation skipped (error):", e)