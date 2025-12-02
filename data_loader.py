<<<<<<< HEAD
#Tycho Collins
#DataLoader file from CelebA Dataset


=======
"""
CelebA data loader with shared transforms for GAN, VAE, and DDPM pipelines.
"""

import os
>>>>>>> cesar/main
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

<<<<<<< HEAD
=======
# Centralized transform configuration
DEFAULT_IMAGE_MEAN = (0.5, 0.5, 0.5)
DEFAULT_IMAGE_STD = (0.5, 0.5, 0.5)


def build_transforms(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD),
        ]
    )


def _transform_record(examples, transform):
    examples["pixel_values"] = transform(examples["image"].convert("RGB"))
    return examples


>>>>>>> cesar/main
class HFDatasetIterable(IterableDataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.hf_dataset = hf_dataset

    def __iter__(self):
<<<<<<< HEAD
        # HF streaming dataset is already an iterator
        for example in self.hf_dataset:
            yield example

def get_celeba_loader(batch_size=32, resolution=64):
    """
    Creates a streaming DataLoader for CelebA.
=======
        for example in self.hf_dataset:
            if "pixel_values" not in example:
                raise ValueError(f"Dataset example missing 'pixel_values'; keys={list(example.keys())}")
            img = example["pixel_values"]
            if not torch.is_tensor(img) or img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"Unexpected shape for 'pixel_values': expected [3,H,W], got {getattr(img, 'shape', None)}")
            yield example


def _has_local_images(root: str) -> bool:
    return os.path.isdir(root) and any(
        f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(root)
    )


def get_celeba_loader(
    batch_size: int = 32,
    resolution: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    transform: transforms.Compose | None = None,
    shuffle_buffer: int = 1000,
    local_dir: str | None = "data/celeba",
):
    """
    Creates a DataLoader for CelebA with consistent transforms.
    Tries local_dir first (if images exist), else HuggingFace streaming; fails cleanly if neither available.
    Returns batches shaped [B, 3, resolution, resolution] in [-1, 1] under key 'pixel_values'.
>>>>>>> cesar/main
    """
    print(f"Initializing CelebA loader (resolution={resolution})...")

<<<<<<< HEAD
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def transform_fn(examples):
        examples['pixel_values'] = transform(examples['image'].convert("RGB"))
        return examples
=======
    transform = transform or build_transforms(resolution)

    if local_dir and _has_local_images(local_dir):
        print(f"Using local dataset from {local_dir}")
        ds = ImageFolder(local_dir, transform=transform)
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
>>>>>>> cesar/main

    try:
        dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
    except Exception:
<<<<<<< HEAD
        # fallback to non-streaming (downloads dataset locally; slower but stable)
        print("Streaming failed — falling back to non-streaming download (this may take time).")
        dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=False)
    dataset = dataset.map(transform_fn, remove_columns=["image", "celeba_filename"])
    dataset = dataset.shuffle(seed=42, buffer_size=1000)

    iterable = HFDatasetIterable(dataset)
    dataloader = DataLoader(iterable, batch_size=batch_size)
    return dataloader
=======
        print("Streaming failed; falling back to non-streaming download (this may take time).")
        try:
            dataset = load_dataset("nielsr/CelebA-faces", split="train", streaming=False)
        except Exception as e:
            if local_dir and _has_local_images(local_dir):
                ds = ImageFolder(local_dir, transform=transform)
                return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
            raise RuntimeError(
                f"Could not load CelebA via streaming or local dir ({local_dir}). "
                "Provide a local dataset folder or internet access."
            ) from e

    print(f"Dataset columns before mapping: {dataset.column_names}")
    to_remove = [c for c in ("image", "celeba_filename") if c in dataset.column_names]
    dataset = (
        dataset.map(lambda ex: _transform_record(ex, transform), remove_columns=to_remove)
        if to_remove
        else dataset.map(lambda ex: _transform_record(ex, transform))
    )
    if shuffle_buffer:
        dataset = dataset.shuffle(seed=42, buffer_size=shuffle_buffer)

    iterable = HFDatasetIterable(dataset)
    return DataLoader(iterable, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

>>>>>>> cesar/main

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    print("Testing data pipeline...")

    loader = get_celeba_loader(batch_size=16)
<<<<<<< HEAD
    
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
=======
    batch = next(iter(loader))
    images = batch["pixel_values"]

    print(f"\n[SUCCESS] Batch loaded.")
    print(f"Tensor Shape: {images.shape}")  # Expect [16, 3, 64, 64]
    print(f"Min Value: {images.min():.2f}")
    print(f"Max Value: {images.max():.2f}")

    try:
        from generate_faces2 import load_model_from_checkpoint, generate_from_model
        import os
        import numpy as np

>>>>>>> cesar/main
        ckpt = "models/latest_checkpoint.pt"
        if not os.path.exists(ckpt):
            print("Checkpoint missing:", ckpt)
        else:
            G = load_model_from_checkpoint(ckpt, device=torch.device("cpu"), nz=100)
            imgs = generate_from_model(G, num_images=8, nz=100, device=torch.device("cpu"), seed=42)

<<<<<<< HEAD
            # compute pixel stats and save debug images
            arrs = [np.array(im).astype(np.float32)/255.0 for im in imgs]
=======
            arrs = [np.array(im).astype(np.float32) / 255.0 for im in imgs]
>>>>>>> cesar/main
            stack = np.stack(arrs)
            print("shape:", stack.shape)
            print("mean:", stack.mean(), "std:", stack.std(), "min:", stack.min(), "max:", stack.max())

            for i, im in enumerate(imgs):
                im.save(f"debug_gen_{i}.png")
                print("Saved debug_gen_%d.png" % i)
    except Exception as e:
<<<<<<< HEAD
        print("Demo generation skipped (error):", e)
=======
        print("Demo generation skipped (error):", e)
>>>>>>> cesar/main
