import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from metrics import compute_and_store_metrics, _NoiseImages


ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}


class ImageFolderDataset(Dataset):
    def __init__(self, root, resolution=64):
        root = Path(root)
        self.files = [p for p in root.rglob("*") if p.suffix.lower() in ALLOWED_EXTS]
        if not self.files:
            raise ValueError(f"No images found under {root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return {"pixel_values": self.transform(img)}


def build_loader(root, batch_size=16, resolution=64):
    ds = ImageFolderDataset(root, resolution=resolution)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def build_noise_loader(count=96, seed=0, batch_size=16):
    return DataLoader(_NoiseImages(count=count, seed=seed), batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="Run FID + collapse metrics and store for the dashboard.")
    parser.add_argument("--real_dir", type=str, default=None, help="Directory with real images (optional).")
    parser.add_argument("--fake_dir", type=str, default=None, help="Directory with generated images.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for loaders.")
    parser.add_argument("--resolution", type=int, default=64, help="Resize images to this resolution before scoring.")
    parser.add_argument("--noise_count", type=int, default=96, help="If dirs are missing, use synthetic noise of this size.")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches (for quick runs).")
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/logs/metrics.pt",
        help="Where to save metrics for the dashboard.",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device, e.g., cpu or cuda.")
    args = parser.parse_args()

    if args.real_dir and args.fake_dir:
        print(f"[INFO] Using image folders for metrics: real={args.real_dir}, fake={args.fake_dir}")
        real_loader = build_loader(args.real_dir, batch_size=args.batch_size, resolution=args.resolution)
        fake_loader = build_loader(args.fake_dir, batch_size=args.batch_size, resolution=args.resolution)
    else:
        print("[INFO] No image folders provided. Using synthetic noise loaders for a smoke test.")
        real_loader = build_noise_loader(count=args.noise_count, seed=1, batch_size=args.batch_size)
        fake_loader = build_noise_loader(count=args.noise_count, seed=2, batch_size=args.batch_size)

    metrics = compute_and_store_metrics(
        real_loader,
        fake_loader,
        save_path=args.save_path,
        device=args.device,
        max_batches=args.max_batches,
    )

    print(f"[DONE] Saved metrics to {args.save_path}")
    print(f"FID: {metrics['fid']:.2f} | Collapse flag: {metrics['collapse']}")
    if "collapse_detail" in metrics:
        detail = metrics["collapse_detail"]
        print(
            f"Variance: {detail.get('feature_variance'):.4f}, "
            f"Mean L2: {detail.get('mean_pairwise_distance'):.2f}"
        )


if __name__ == "__main__":
    main()
