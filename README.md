# GenAI

<<<<<<< HEAD
# AI-Based Facial Generation System

## Project Summary
This project develops an **AI facial generation system** using modern generative models such as **Variational Autoencoders (VAE)**, **Diffusion Models**, and **Flow Matching**. It converts random latent noise into realistic human-face images through advanced deep-learning architecture trained on the **CelebA/FFHQ datasets**. The final objective is to provide a web-based visualization tool that allows users to compare generated faces, analyze model metrics (FID score, loss plots), and document AI performance across iterations.

***

## 👨‍💻 Team & Responsibilities
This project is currently transitioning from the **Dataset Setup** phase to the **Model Training** phase.

| Role | Member | Key Task | Status |
| :--- | :--- | :--- | :--- |
| Dataset and Preprocessing//Deployment| **Tycho Collins** | Collect, scale, crop, and normalize the CelebA/FFHQ dataset. | **✅ Completed (Nov 24)** |
| Model Training Lead | Erick Chavez | Set up and train VAE/DDPM/Flow Matching models; tune hyperparameters. |  **✅ Completed**  |
| Image Generation Module | John Cosby | Generate faces from latent noise using trained weights. |  **✅ Completed**  |
| Assessment and Metrics | Cesar Cabrera | Compute FID scores and detect overfitting/mode collapse. | **✅ Completed**  |
| User Interface & Visualization | Da Marc Henry | Create the dashboard that displays outputs, training graphs, and comparisons. |  **✅ Completed**  |

***

## 🚀 Setup & Installation (For Model Training Lead)

### 1. Clone the Repository
```bash
git clone [https://github.com/tychocollins/GenAI.git](https://github.com/tychocollins/GenAI.git)
cd GenAI

```

### Assessment & Metrics (Cesar Cabrera)
- Added `metrics.py` for Frechet Inception Distance and a quick mode-collapse check.
- Expect loaders that return `pixel_values` normalized to `[-1, 1]` (see `data_loader.py`).
- Usage:
  - `fid_from_loaders(real_loader, fake_loader, device="cuda", max_batches=50)`
  - `collapse_report(fake_loader, device="cuda", max_batches=50)`
- To feed Da Marc's dashboard: `compute_and_store_metrics(real_loader, fake_loader, save_path="outputs/logs/metrics.pt")`
- CLI helper: `python metrics_runner.py --real_dir <path_to_real_images> --fake_dir <path_to_generated_images>` (falls back to synthetic noise if dirs are omitted).
- A small offline smoke test is included in `metrics.py` under `__main__`.

### Quick Demo (no checkpoints needed)
- `python demo_bootstrap.py` seeds `outputs/generated/`, `outputs/logs/training_log.pt`, and `outputs/logs/metrics.pt` with synthetic data so `python dashboard.py` can run end-to-end.







=======
AI-based facial generation system with a DCGAN baseline and a DDPM diffusion model. The project now includes device-agnostic training, centralized data transforms, and a Streamlit app that can sample from either generator.

## What's Inside
- DCGAN generator/discriminator for quick baselines (`train.py --model dcgan`).
- DDPM UNet with a linear noise scheduler plus iterative sampling (`train.py --model ddpm` or `train_ddpm.py`).
- Streaming CelebA loader with shared transforms (resize -> tensor -> normalize to `[-1, 1]`).
- Metrics for FID/mode collapse and a Streamlit UI that can toggle between DCGAN and DDPM with sampling progress.

## Device Agnostic by Design
The code selects a device in the order CUDA -> MPS -> CPU and never hard-codes `.cuda()`:
```python
from utils import get_device
device = get_device()  # respects AMD/ROCm and CPU-only setups
```
Pass `--device cpu` to force CPU if ROCm is not available on Windows.

## Training
- DCGAN: `python train.py --model dcgan --batch-size 64 --resolution 64` (DCGAN fixed to 64×64)
- DDPM: `python train.py --model ddpm --batch-size 32 --timesteps 1000 --resolution 64`
- Dedicated DDPM entrypoint (same args): `python train_ddpm.py --batch-size 32 --timesteps 1000`

Checkpoints:
- DCGAN -> `models/dcgan_latest.pt` (legacy alias: `models/latest_checkpoint.pt`)
- DDPM -> `models/ddpm_latest.pt`
Samples land in `outputs/samples/<model>/`.

## Data & Preprocessing
- Source: `nielsr/CelebA-faces` via HuggingFace streaming (fallback to local folder `data/celeba/` when offline).
- Transforms centralized in `data_loader.py` using `build_transforms(resolution)`.
- Loader always emits `batch["pixel_values"]` shaped `[B, 3, H, W]` in `[-1, 1]`, so GAN, VAE, and DDPM pipelines share the same input contract.

## Deployment (Streamlit)
Run `streamlit run app_streamlit.py`.
- Toggle DCGAN/DDPM generation.
- DDPM shows iterative sampling progress.
- Gracefully handles missing checkpoints and optional env downloads (`CHECKPOINT_URL` for DCGAN, `CHECKPOINT_URL_DDPM` for DDPM).

## Responsibilities & Status
- Dataset & preprocessing - streaming loader + normalization.
- Model training - DCGAN baseline and DDPM pipeline/support.
- Debugging & stability - metrics, logging, safer checkpoints.
- Deployment - Streamlit/dashboard updated for both models.
- Device portability - AMD/CPU friendly; no CUDA-only calls.

## Metrics & Evaluation
- FID and collapse heuristics in `metrics.py`; run `python metrics_runner.py --fake_dir outputs/samples/<model>/ --real_dir <reference> --device cpu --weights <local_inception_weights>`.
- Outputs stored under `outputs/logs/metrics.pt` for dashboard consumption.

## Smoke Tests (CPU-only)
- Data pipeline: `python data_loader.py` (runs verification block, prints shape/min/max).
- DCGAN quick run: `python train.py --model dcgan --batch-size 16 --epochs 1 --steps-per-epoch 10 --device cpu`. Checkpoints in `models/`, samples in `outputs/samples/dcgan/`.
- DDPM quick run: `python train.py --model ddpm --batch-size 8 --epochs 1 --steps-per-epoch 5 --device cpu`. Checkpoints in `models/ddpm_latest.pt`, samples in `outputs/samples/ddpm/`.
- Metrics: `python metrics_runner.py --device cpu --max_batches 2` (uses synthetic noise if dirs missing).
- Streamlit: `streamlit run app_streamlit.py` (warns if checkpoints missing; toggles should not crash).
>>>>>>> cesar/main
