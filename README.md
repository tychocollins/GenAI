# GenAI

# AI-Based Facial Generation System

## Project Summary
This project develops an **AI facial generation system** using modern generative models such as **Variational Autoencoders (VAE)**, **Diffusion Models**, and **Flow Matching**. It converts random latent noise into realistic human-face images through advanced deep-learning architecture trained on the **CelebA/FFHQ datasets**. The final objective is to provide a web-based visualization tool that allows users to compare generated faces, analyze model metrics (FID score, loss plots), and document AI performance across iterations.

***

## 👨‍💻 Team & Responsibilities
This project is currently transitioning from the **Dataset Setup** phase to the **Model Training** phase.

| Role | Member | Key Task | Status |
| :--- | :--- | :--- | :--- |
| **Dataset and Preprocessing** | **Tycho Collins** | Collect, scale, crop, and normalize the CelebA/FFHQ dataset. | **✅ Completed (Nov 24)** |
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







