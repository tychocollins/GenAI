# AI-Based Facial Generation System (GenAI)

This project develops an **AI facial generation system** using modern generative models such as **DCGAN**, and **Denoising Diffusion Probabilistic Models (DDPM)**. It converts random latent noise into realistic human-face images through advanced deep-learning architecture trained on the **CelebA** dataset.

The system includes a Streamlit-based web tool that allows users to generate faces, analyze model metrics (FID score), and observe the iterative sampling process of the diffusion model.

***

## 👨‍💻 Team & Responsibilities

| Role | Member | Key Task | Status |
| :--- | :--- | :--- | :--- |
| Dataset & Preprocessing | **Tycho Collins** | Collect, scale, crop, and normalize the CelebA dataset. | **✅ Completed** |
| Model Training Lead | Erick Chavez | Set up and train DCGAN and DDPM models; tune hyperparameters. |  **✅ Completed**  |
| Image Generation Module | John Cosby | Generate faces from latent noise using trained weights. |  **✅ Completed**  |
| Assessment & Metrics | Cesar Cabrera | Compute FID scores and detect overfitting/mode collapse. | **✅ Completed**  |
| User Interface | Da Marc Henry | Create the Streamlit app for visualization and sampling progress. |  **✅ Completed**  |

***

## 🚀 What's Inside
- **DCGAN**: Generator/discriminator for quick baselines (`train.py --model dcgan`).
- **DDPM**: UNet with a linear noise scheduler plus iterative sampling (`train.py --model ddpm` or `train_ddpm.py`).
- **Streaming Loader**: CelebA loader with shared transforms (resize -> tensor -> normalize to `[-1, 1]`).
- **Metrics**: FID and mode collapse heuristics in `metrics.py`.
- **Streamlit UI**: Toggle between DCGAN and DDPM with real-time sampling progress.

## 🛠 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tychocollins/GenAI.git
cd GenAI
```

### 2. Device Agnostic by Design
The code automatically selects the best available device (CUDA -> MPS -> CPU):
```python
from utils import get_device
device = get_device()
```

## 🏋️‍♂️ Training

- **DCGAN**: `python train.py --model dcgan --batch-size 64 --resolution 64`
- **DDPM**: `python train.py --model ddpm --batch-size 32 --timesteps 1000 --resolution 64`
- **Dedicated DDPM entrypoint**: `python train_ddpm.py --batch-size 32 --timesteps 1000`

Checkpoints are saved to `models/` (e.g., `models/ddpm_latest.pt`).

## 🌐 Deployment (Streamlit)
Run the following command to start the web interface:
```bash
streamlit run app_streamlit.py
```

## 📊 Metrics & Evaluation
Run the metrics suite to evaluate model performance:
```bash
python metrics_runner.py --fake_dir outputs/samples/ddpm/ --real_dir <reference_path> --device cpu
```

## 🧪 Smoke Tests (CPU-only)
- **Data pipeline**: `python data_loader.py`
- **DCGAN quick run**: `python train.py --model dcgan --batch-size 16 --epochs 1 --steps-per-epoch 10 --device cpu`
- **DDPM quick run**: `python train_ddpm.py --batch-size 8 --epochs 1 --steps-per-epoch 5 --device cpu`
- **Streamlit**: `streamlit run app_streamlit.py`
