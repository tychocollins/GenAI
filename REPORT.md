# GenAI Project Report (latest update)

## Scope Completed
- Added DDPM architecture: UNet denoiser, linear noise scheduler, forward diffusion, and iterative sampling utilities (`ddpm.py`).
- Built dedicated DDPM training pipeline with checkpoints, previews, and resume support (`train_ddpm.py`), and wired `train.py` to choose DCGAN or DDPM via `--model`.
- Centralized data transforms for CelebA streaming (`data_loader.py`) to guarantee `[B, 3, H, W]` in `[-1, 1]`.
- Updated Streamlit deployment with DCGAN/DDPM toggle, DDPM progress display, and safer checkpoint handling.
- Device portability audit: removed CUDA hard-coding, standardized `get_device()` (CUDA -> MPS -> CPU) for AMD/ROCm and CPU fallback.

## Responsibilities (current owners)
- Dataset & preprocessing: streaming loader and normalization.
- Model training: DCGAN baseline plus DDPM support scripts and checkpoints.
- Debugging & stability: metrics, logging, and checkpoint sanity.
- Deployment: Streamlit experience and dashboard updates.
- Device portability: shared - AMD/CPU-friendly code paths, no CUDA-only calls.

## Usage Quicklinks
- Train DCGAN: `python train.py --model dcgan --batch-size 64 --resolution 64`
- Train DDPM: `python train.py --model ddpm --batch-size 32 --timesteps 1000 --resolution 64`
- Streamlit app: `streamlit run app_streamlit.py` (toggle DCGAN/DDPM)
- Metrics: `python metrics_runner.py --fake_dir outputs/samples/<model>/ --real_dir data/real --device cpu --weights <local_inception_weights>`
